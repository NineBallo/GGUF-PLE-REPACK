#!/usr/bin/env python3
# Repack a split GGUF so one tensor (default: the qwen4exp engram table
# per_layer_token_embd.weight) sits ALONE in the final shard, with no other tensors
# sharing that file. Motivation: on Metal, llama.cpp wires the mmap'd regions that
# back GPU tensors; a tensor interleaved with them in the same file gets wired along
# for the ride (measured +24 GiB on Qwen3.8-Flash-Next). A file containing only
# CPU-side tensors keeps its own mapping and stays pageable.
#
# GRAFT mode (--graft-from): the isolated tensor's bytes, GGUF type, and n_bytes are
# taken from a same-named tensor in ANOTHER split GGUF (e.g. the Q8 table shard from
# UD-Q5_K_XL) instead of from the input. Repack + graft in one write pass; every
# other tensor is copied verbatim from the input.
#
# Run with the gguf-py matching the model's llama.cpp tree:
#   uv run --no-project --with ~/AI/llama.cpp-pr27742/gguf-py python repack-ple-shard.py \
#       ~/AI/models/Qwen3.8-Flash-Next-Unsloth/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
#       --graft-from ~/AI/models/.../UD-Q5_K_XL-00004-of-00004.gguf
#   ... --dry-run  (prints the layout incl. grafted type/size; writes nothing)
#   ... --verify [--graft-from ...]  (byte-compares every tensor against its source)

import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType

SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
SPLIT_KV_TYPES = {
    "split.no": GGUFValueType.UINT16,
    "split.tensors.count": GGUFValueType.INT32,
    "split.count": GGUFValueType.UINT16,
}


def shard_paths(first_shard: Path) -> tuple[str, list[Path]]:
    m = SPLIT_RE.match(first_shard.name)
    if not m or int(m.group(2)) != 1:
        sys.exit(f"ERROR: {first_shard.name} is not a '-00001-of-NNNNN.gguf' first shard")
    stem, count = m.group(1), int(m.group(3))
    paths = [first_shard.parent / f"{stem}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]
    for p in paths:
        if not p.is_file():
            sys.exit(f"ERROR: missing input shard {p}")
    return stem, paths


def out_paths(first_shard: Path, stem: str, suffix: str, count: int) -> list[Path]:
    return [first_shard.parent / f"{stem}-{suffix}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)]


def read_all(paths: list[Path]) -> list[GGUFReader]:
    return [GGUFReader(p) for p in paths]


def kv_int(reader: GGUFReader, key: str) -> int:
    field = reader.fields.get(key)
    if field is None:
        sys.exit(f"ERROR: {key} missing from {reader.data.filename if hasattr(reader.data, 'filename') else 'shard'} — not a split GGUF?")
    return int(field.contents())


def kv_val(reader: GGUFReader, key: str):
    field = reader.fields.get(key)
    return None if field is None else field.contents()


def load_graft_tensor(path: Path, name: str) -> tuple[GGUFReader, GGUFReader, "object"]:
    # Returns (meta_reader_of_source_shard1, holder_reader, tensor). The holder must
    # outlive any use of tensor.data — keep it referenced through the write pass.
    # Accepts any shard of a split set (the Q5_K_XL table lives in its own shard,
    # usually NOT shard 1) or a standalone GGUF. Opens shards lazily and stops at
    # the first one containing the tensor.
    m = SPLIT_RE.match(path.name)
    if not m:
        r = GGUFReader(path)
        cands = [t for t in r.tensors if t.name == name]
        if len(cands) != 1:
            sys.exit(f"ERROR: graft source {path.name}: found {len(cands)} tensors named {name!r}")
        meta = r if not m else None
        return meta, r, cands[0]
    stem, count = m.group(1), int(m.group(3))
    meta = None
    for i in range(1, count + 1):
        p = path.parent / f"{stem}-{i:05d}-of-{count:05d}.gguf"
        if not p.is_file():
            sys.exit(f"ERROR: graft source set: missing shard {p}")
        r = GGUFReader(p)
        if i == 1:
            meta = r
        cands = [t for t in r.tensors if t.name == name]
        if cands:
            if len(cands) != 1:
                sys.exit(f"ERROR: graft source {p.name}: found {len(cands)} tensors named {name!r}")
            return meta if meta is not None else r, r, cands[0]
    sys.exit(f"ERROR: tensor {name!r} not found anywhere in graft source set of {path.name}")


def kv_val(reader, key):
    field = reader.fields.get(key)
    return None if field is None else field.contents()


def check_graft_compat(target_meta, src_meta, iso, src) -> None:
    # Hard gate: identical shape (element count/layout) is what makes a graft valid.
    if list(iso.shape) != list(src.shape):
        sys.exit(f"ERROR: graft shape mismatch: target {list(iso.shape)} vs source {list(src.shape)}")
    b_arch = kv_val(src_meta, "general.architecture")
    if b_arch is None:
        print("WARNING: graft source has no metadata (single-tensor shard) — skipping "
              "architecture/ngram KV cross-check. Acceptable when both files come from "
              "the same repo/snapshot; shape equality is still enforced.")
        return
    a_arch = kv_val(target_meta, "general.architecture")
    if a_arch != b_arch:
        sys.exit(f"ERROR: architecture mismatch: {a_arch!r} vs {b_arch!r}")
    # Only cross-check ngram/PLE KV when BOTH sides carry them; a partial download
    # simply has nothing to compare against.
    keys = {k for r in (target_meta, src_meta) for k in r.fields
            if "ngram" in k.lower() or "ple" in k.lower()}
    for k in sorted(keys):
        va, vb = kv_val(target_meta, k), kv_val(src_meta, k)
        if va is not None and vb is not None and va != vb:
            sys.exit(f"ERROR: metadata KV {k!r} differs (target {va!r} vs graft source {vb!r})")

def type_name(t) -> str:
    try:
        return str(t.tensor_type)
    except Exception:
        return str(int(t.tensor_type))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def bytes_equal(a: np.ndarray, b: np.ndarray, chunk: int = 1 << 28) -> bool:
    # Compare as raw bytes, chunked: a whole-tensor `a == b` materializes a bool array
    # the size of the tensor (26.8 GiB for the PLE), and value comparison on floats
    # treats +0.0 == -0.0 and NaN != NaN — byte view avoids both.
    av = a.reshape(-1).view(np.uint8)
    bv = b.reshape(-1).view(np.uint8)
    if av.nbytes != bv.nbytes:
        return False
    return all(np.array_equal(av[off:off + chunk], bv[off:off + chunk]) for off in range(0, av.nbytes, chunk))


def plan_shards(readers: list[GGUFReader], isolate: str, n_data_shards: int):
    # Global tensor order = file order across data shards; that order is preserved in
    # the output so nothing but file membership changes.
    all_tensors = [t for r in readers for t in r.tensors]
    iso = [t for t in all_tensors if t.name == isolate]
    if len(iso) != 1:
        sys.exit(f"ERROR: expected exactly one tensor named {isolate!r}, found {len(iso)}")
    rest = [t for t in all_tensors if t.name != isolate]

    def padded(t):  # on-disk footprint, 32-byte GGUF alignment
        return (int(t.n_bytes) + 31) // 32 * 32

    total = sum(padded(t) for t in rest)
    shards: list[list] = [[] for _ in range(n_data_shards)]
    target = total / n_data_shards
    idx, acc = 0, 0
    for t in rest:
        # advance once the current shard is at target; the last shard takes the tail
        if acc >= target and idx < n_data_shards - 1 and shards[idx]:
            idx += 1
            acc = 0
        shards[idx].append(t)
        acc += padded(t)
    if any(not s for s in shards):
        sys.exit("ERROR: shard packing produced an empty data shard — lower the shard count")
    return all_tensors, shards, iso[0]


def write_data_shard(path: Path, tensors, split_no: int, split_count: int, split_tensors_total: int) -> None:
    # arch is a constructor formality: add_architecture() is never called, so the only
    # KVs in the file are the three split.* keys, exactly like llama-gguf-split output.
    w = GGUFWriter(path=None, arch="")
    w.add_key_value("split.no", split_no, SPLIT_KV_TYPES["split.no"])
    w.add_key_value("split.tensors.count", split_tensors_total, SPLIT_KV_TYPES["split.no"].__class__ and GGUFValueType.INT32)
    w.add_key_value("split.count", split_count, SPLIT_KV_TYPES["split.count"])
    for t in tensors:
        # Contract with gguf-py: ReaderTensor.shape is GGUF file order (ne); the writer
        # serializes reversed(shape), so pass reversed dims. The dummy non-uint8 dtype
        # skips the writer's byte-shape-to-quant-shape conversion (we already have
        # element dims), and raw_dtype carries the real quant type through untouched —
        # in graft mode this is the SOURCE tensor's type, copied, never hardcoded.
        w.add_tensor_info(
            t.name,
            tuple(reversed(t.shape.tolist())),
            np.dtype(np.float32),
            int(t.n_bytes),
            raw_dtype=t.tensor_type,
        )
    w.write_header_to_file(path=path)
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for i, t in enumerate(tensors):
        w.write_tensor_data(t.data)  # memmap-backed: streams, no full-tensor RAM copy
        if (i + 1) % 200 == 0 or i + 1 == len(tensors):
            print(f"    {path.name}: {i + 1}/{len(tensors)} tensors", flush=True)
    w.close()


def do_repack(args, first: Path, stem: str, in_paths: list[Path]) -> None:
    readers = read_all(in_paths)
    split_count = kv_int(readers[0], "split.count")
    split_total = kv_int(readers[0], "split.tensors.count")
    if split_count != len(in_paths):
        sys.exit(f"ERROR: split.count={split_count} but {len(in_paths)} files found")
    if readers[0].tensors:
        sys.exit("ERROR: input shard 1 is not metadata-only; this script relies on copying it verbatim")

    n_data_shards = len(in_paths) - 2  # everything but the metadata shard and the isolate shard
    if n_data_shards < 1:
        sys.exit("ERROR: need at least 3 input shards (metadata + 1 data + isolate)")
    all_tensors, shards, iso = plan_shards(readers[1:], args.isolate, n_data_shards)
    if len(all_tensors) != split_total:
        sys.exit(f"ERROR: read {len(all_tensors)} tensors but split.tensors.count={split_total}")

    graft_holder = None
    final_tensor = iso
    if args.graft_from:
        src_meta, graft_holder, src = load_graft_tensor(args.graft_from.resolve(), args.isolate)
        check_graft_compat(readers[0], src_meta, iso, src)
        final_tensor = src
        print(f"Graft source: {type_name(src)}, {int(src.n_bytes):,} bytes "
              f"({8 * int(src.n_bytes) / int(np.prod(src.shape)):.2f} bpv) "
              f"replacing {type_name(iso)}, {int(iso.n_bytes):,} bytes "
              f"({8 * int(iso.n_bytes) / int(np.prod(iso.shape)):.2f} bpv)")
        if int(src.n_bytes) == int(iso.n_bytes) and str(src.tensor_type) == str(iso.tensor_type):
            print("WARNING: graft source is identical type+size to the existing table — no-op graft")

    outs = out_paths(first, stem, args.suffix, len(in_paths))
    existing = [p for p in outs if p.exists()]
    if existing and not args.force:
        sys.exit(f"ERROR: output exists (use --force to overwrite): {existing[0]}")

    # Graft changes the byte count of the final shard; size the free-space check on
    # what will ACTUALLY be written, not on the input tensors.
    need = sum(int(t.n_bytes) for t in all_tensors) - int(iso.n_bytes) + int(final_tensor.n_bytes)
    need += in_paths[0].stat().st_size
    free = shutil.disk_usage(first.parent).free
    if free < need * 1.02:
        sys.exit(f"ERROR: needs ~{need / 2**30:.1f} GiB free in {first.parent}, only {free / 2**30:.1f} GiB available")

    print(f"Plan: {len(all_tensors)} tensors -> {len(outs)} files (suffix -{args.suffix})")
    for i, s in enumerate(shards):
        print(f"  shard {i + 2}: {len(s)} tensors, {sum(int(t.n_bytes) for t in s) / 2**30:.2f} GiB")
    src_lbl = "GRAFTED" if args.graft_from else "isolated"
    print(f"  shard {len(outs)}: 1 tensor ({args.isolate}), "
          f"{int(final_tensor.n_bytes) / 2**30:.2f} GiB  <- {src_lbl}")
    if args.dry_run:
        print("Dry run: nothing written.")
        return

    # Stage to .tmp names and publish only when every shard is complete: a failure
    # mid-write must neither leave partial final-named files nor (with --force)
    # destroy a previously good output set.
    tmps = [p.with_name(p.name + ".tmp") for p in outs]
    try:
        print(f"  shard 1: verbatim copy of {in_paths[0].name}")
        shutil.copyfile(in_paths[0], tmps[0])
        for i, s in enumerate(shards):
            write_data_shard(tmps[i + 1], s, split_no=i + 1, split_count=split_count, split_tensors_total=split_total)
        # final shard: the grafted tensor if requested (its metadata + bytes come from
        # the source reader, which graft_holder keeps alive), else the input's table.
        write_data_shard(tmps[-1], [final_tensor], split_no=len(outs) - 1, split_count=split_count, split_tensors_total=split_total)
    except BaseException:
        for t in tmps:
            t.unlink(missing_ok=True)
        raise
    for t, p in zip(tmps, outs):
        os.replace(t, p)
    print("Repack written. Run with --verify before using it.")


def do_verify(args, first: Path, stem: str, in_paths: list[Path]) -> None:
    outs = out_paths(first, stem, args.suffix, len(in_paths))
    for p in outs:
        if not p.is_file():
            sys.exit(f"ERROR: missing output shard {p}")
    fails = 0
    if file_sha256(in_paths[0]) != file_sha256(outs[0]):
        print("FAIL: shard 1 differs from input shard 1")
        fails += 1
    in_readers = read_all(in_paths[1:])
    out_readers = read_all(outs[1:])

    graft_holder = None
    graft_src = None
    if args.graft_from:
        src_meta, graft_holder, graft_src = load_graft_tensor(args.graft_from.resolve(), args.isolate)

    for i, r in enumerate(out_readers):
        want_no = i + 1
        if kv_int(r, "split.no") != want_no or kv_int(r, "split.count") != len(outs):
            print(f"FAIL: bad split KVs in {outs[i + 1].name}")
            fails += 1
        for key, want_type in SPLIT_KV_TYPES.items():
            # llama.cpp reads these with fixed types; the right value in the wrong
            # encoding still fails to load
            types = r.fields[key].types if key in r.fields else []
            if types != [want_type]:
                print(f"FAIL: {key} in {outs[i + 1].name} has types {types}, want [{want_type}]")
                fails += 1
        # llama.cpp requires offsets equal to the cumulative 32-byte-padded sizes; a
        # file with relocated data would read back fine in Python but be rejected there
        prev_end = None
        for t in r.tensors:
            off = int(t.data_offset)
            if off % 32 != 0 or (prev_end is not None and off != prev_end):
                print(f"FAIL: non-canonical data offset for {t.name} in {outs[i + 1].name}")
                fails += 1
                break
            prev_end = (off + int(t.n_bytes) + 31) // 32 * 32
    iso_shard = out_readers[-1]
    if len(iso_shard.tensors) != 1 or iso_shard.tensors[0].name != args.isolate:
        print(f"FAIL: final shard does not contain exactly [{args.isolate}]")
        fails += 1

    src = {t.name: t for r in in_readers for t in r.tensors}
    dst = {}
    for r in out_readers:
        for t in r.tensors:
            if t.name in dst:
                print(f"FAIL: duplicate tensor {t.name} across output shards")
                fails += 1
            dst[t.name] = t
    split_total = kv_int(out_readers[0], "split.tensors.count")
    if len(dst) != split_total:
        print(f"FAIL: {len(dst)} tensors across output shards, split.tensors.count says {split_total}")
        fails += 1
    if src.keys() != dst.keys():
        print(f"FAIL: tensor sets differ (in-only: {sorted(src.keys() - dst.keys())[:3]}, out-only: {sorted(dst.keys() - dst.keys())[:3]})")
        fails += 1
    checked = 0
    for name in sorted(src.keys() & dst.keys()):
        if graft_src is not None and name == args.isolate:
            continue  # the grafted tensor is checked against the graft source, below
        a, b = src[name], dst[name]
        if a.tensor_type != b.tensor_type or a.shape.tolist() != b.shape.tolist() or int(a.n_bytes) != int(b.n_bytes):
            print(f"FAIL: meta mismatch on {name}")
            fails += 1
            continue
        if not bytes_equal(a.data, b.data):
            print(f"FAIL: byte mismatch on {name}")
            fails += 1
        checked += 1
        if checked % 200 == 0:
            print(f"    verified {checked}/{len(src)} tensors", flush=True)
    if graft_src is not None:
        # The isolated table must match the GRAFT source, not the input table.
        b = dst[args.isolate]
        if (graft_src.tensor_type != b.tensor_type
                or graft_src.shape.tolist() != b.shape.tolist()
                or int(graft_src.n_bytes) != int(b.n_bytes)):
            print(f"FAIL: grafted table meta mismatch (source {type_name(graft_src)}/{int(graft_src.n_bytes)} "
                  f"vs output {type_name(b)}/{int(b.n_bytes)})")
            fails += 1
        elif not bytes_equal(graft_src.data, b.data):
            print(f"FAIL: grafted table byte mismatch vs {args.graft_from}")
            fails += 1
        else:
            print(f"    grafted table verified against source: {type_name(b)}, {int(b.n_bytes):,} bytes")
    if fails:
        sys.exit(f"VERIFY FAILED: {fails} problem(s)")
    print(f"VERIFY PASSED: {checked} tensors byte-identical"
          + (", table grafted from source" if graft_src is not None else "")
          + ", shard 1 identical, split KVs correct.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolate one tensor into its own GGUF split shard (byte-exact repack), "
                                             "optionally grafting a replacement from another GGUF in the same pass")
    ap.add_argument("first_shard", type=Path, help="path to the -00001-of-NNNNN.gguf input shard")
    ap.add_argument("--isolate", default="per_layer_token_embd.weight")
    ap.add_argument("--graft-from", type=Path, default=None,
                    help="any shard of a split GGUF (or a standalone GGUF) containing the replacement tensor")
    ap.add_argument("--suffix", default="PLESHARD", help="inserted into output filenames before the shard numbering")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="verify a previous repack instead of writing one")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    first = args.first_shard.resolve()
    stem, in_paths = shard_paths(first)
    if args.verify:
        do_verify(args, first, stem, in_paths)
    else:
        do_repack(args, first, stem, in_paths)


if __name__ == "__main__":
    main()
