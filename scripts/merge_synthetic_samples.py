#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import sys
from typing import List, Tuple


SAMPLE_DIR_RE = re.compile(r"^s\d{6}$")


def list_sample_dirs(root_dir: str) -> List[str]:
    # Non-lazy, full list
    try:
        entries = os.listdir(root_dir)
    except FileNotFoundError:
        raise SystemExit(f"Directory does not exist: {root_dir}")
    except Exception as ex:
        raise SystemExit(f"Failed to list directory {root_dir}: {ex}")

    dir_names: List[str] = []
    for name in entries:
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and SAMPLE_DIR_RE.match(name):
            dir_names.append(name)
    dir_names.sort()
    return dir_names


def ensure_sample_files(root_dir: str, dir_name: str) -> Tuple[str, str]:
    base = dir_name
    dpath = os.path.join(root_dir, dir_name)
    json_path = os.path.join(dpath, f"{base}.json")
    npz_path = os.path.join(dpath, f"{base}.npz")
    if not os.path.isfile(json_path):
        raise SystemExit(f"Missing JSON file: {json_path}")
    if not os.path.isfile(npz_path):
        raise SystemExit(f"Missing NPZ file: {npz_path}")
    return json_path, npz_path


def rename_files_in_dir(dir_path: str, old_base: str, new_base: str, dry_run: bool) -> None:
    if old_base == new_base:
        return
    old_json = os.path.join(dir_path, f"{old_base}.json")
    old_npz = os.path.join(dir_path, f"{old_base}.npz")
    new_json = os.path.join(dir_path, f"{new_base}.json")
    new_npz = os.path.join(dir_path, f"{new_base}.npz")

    if not os.path.isfile(old_json) or not os.path.isfile(old_npz):
        raise SystemExit(f"Expected files not found in {dir_path}: {old_base}.json/.npz")

    if os.path.exists(new_json) or os.path.exists(new_npz):
        raise SystemExit(f"Target files already exist in {dir_path}: {new_base}.json/.npz")

    print(f"REN FILE {old_json} -> {new_json}")
    print(f"REN FILE {old_npz} -> {new_npz}")
    if not dry_run:
        os.rename(old_json, new_json)
        os.rename(old_npz, new_npz)


def rename_dir(parent_dir: str, old_name: str, new_name: str, dry_run: bool) -> None:
    src = os.path.join(parent_dir, old_name)
    dst = os.path.join(parent_dir, new_name)
    if os.path.exists(dst):
        raise SystemExit(f"Target directory already exists: {dst}")
    print(f"REN DIR  {src} -> {dst}")
    if not dry_run:
        os.rename(src, dst)


def move_dir(src_dir: str, dst_parent: str, new_name: str, dry_run: bool) -> None:
    if not os.path.isdir(src_dir):
        raise SystemExit(f"Source directory missing: {src_dir}")
    final_dst = os.path.join(dst_parent, new_name)
    if os.path.exists(final_dst):
        raise SystemExit(f"Destination already exists: {final_dst}")
    print(f"MOVE DIR {src_dir} -> {final_dst}")
    if not dry_run:
        shutil.move(src_dir, final_dst)


def to_basename(index: int) -> str:
    if index < 0 or index > 999999:
        raise SystemExit(f"Index out of supported range [0, 999999]: {index}")
    return f"s{index:06d}"


def parse_index_from_name(name: str) -> int:
    if SAMPLE_DIR_RE.match(name):
        return int(name[1:])
    if name.startswith("__tmp__s") and SAMPLE_DIR_RE.match(name.replace("__tmp__", "", 1)):
        return int(name.replace("__tmp__", "", 1)[1:])
    return -1


def compute_next_index(dir_path: str) -> int:
    try:
        names = os.listdir(dir_path)
    except Exception as ex:
        raise SystemExit(f"Failed to list {dir_path} to compute next index: {ex}")
    max_idx = -1
    for name in names:
        full = os.path.join(dir_path, name)
        if not os.path.isdir(full):
            continue
        idx = parse_index_from_name(name)
        if idx >= 0 and idx > max_idx:
            max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0


def stage_src1_rename_in_place(src1: str, start_index: int, dry_run: bool) -> int:
    # Gather current well-formed sample dirs
    dir_names = list_sample_dirs(src1)
    count_regular = len(dir_names)

    # Also account for any pre-existing temp-staged dirs from an earlier aborted run
    pre_tmp_dirs = [d for d in os.listdir(src1) if d.startswith("__tmp__s") and os.path.isdir(os.path.join(src1, d))]
    pre_tmp_dirs.sort()

    # Map current name -> final basename, stable order by current name
    mapping = []  # (current_dir_name, final_basename)
    for i, name in enumerate(dir_names):
        mapping.append((name, to_basename(start_index + i)))

    # Phase A: rename files and move dir to temp name to avoid collisions
    for cur_name, final_base in mapping:
        cur_dir = os.path.join(src1, cur_name)
        ensure_sample_files(src1, cur_name)
        # Update file basenames first (within current dir)
        rename_files_in_dir(cur_dir, cur_name, final_base, dry_run)
        # Rename directory to temporary unique name under same parent
        tmp_name = f"__tmp__{final_base}"
        rename_dir(src1, cur_name, tmp_name, dry_run)

    # Phase B: drop temp prefix to reach final names
    # Re-list to find all tmp dirs (including pre-existing and the ones we just created)
    tmp_dirs = [d for d in os.listdir(src1) if d.startswith("__tmp__s") and os.path.isdir(os.path.join(src1, d))]
    tmp_dirs.sort()
    for tmp_name in tmp_dirs:
        final_name = tmp_name.replace("__tmp__", "", 1)
        rename_dir(src1, tmp_name, final_name, dry_run)

    # Total number of src1 samples equals regular sample dirs + pre-existing tmp-staged dirs
    return count_regular + len(pre_tmp_dirs)


def process_src2_move(src2: str, dst: str, start_index: int, dry_run: bool) -> int:
    dir_names = list_sample_dirs(src2)
    count = len(dir_names)
    for i, name in enumerate(dir_names):
        final_base = to_basename(start_index + i)
        cur_dir = os.path.join(src2, name)
        ensure_sample_files(src2, name)
        # Rename files in-place within src2 dir first
        rename_files_in_dir(cur_dir, name, final_base, dry_run)
        # Move directory into destination with final name
        move_dir(cur_dir, dst, final_base, dry_run)
    return count


def main():
    parser = argparse.ArgumentParser("Merge and reindex synthetic samples across two directories")
    parser.add_argument("--src1", required=True, help="Primary source dir (will be reindexed in-place starting at s000000)")
    parser.add_argument("--src2", required=True, help="Secondary source dir (will be moved into dst continuing the index)")
    parser.add_argument("--dst", required=True, help="Destination dir for all samples (src1 can equal dst)")
    parser.add_argument("--dry_run", action="store_true", help="Print actions only; do not modify filesystem")
    args = parser.parse_args()

    src1 = os.path.abspath(os.path.expanduser(args.src1))
    src2 = os.path.abspath(os.path.expanduser(args.src2))
    dst = os.path.abspath(os.path.expanduser(args.dst))

    if not os.path.isdir(src1):
        raise SystemExit(f"src1 is not a directory: {src1}")
    if not os.path.isdir(src2):
        raise SystemExit(f"src2 is not a directory: {src2}")
    if not os.path.isdir(dst):
        if args.dry_run:
            print(f"MKDIR    {dst}")
        else:
            os.makedirs(dst, exist_ok=True)

    print(f"SRC1: {src1}")
    print(f"SRC2: {src2}")
    print(f"DST : {dst}")
    print(f"DRY : {args.dry_run}")

    # Step 1: Reindex src1 in place starting at 0
    print("-- Reindexing SRC1 in place starting at s000000 --")
    n1 = stage_src1_rename_in_place(src1, start_index=0, dry_run=args.dry_run)
    # Compute next index robustly from current DST (handles partial previous runs and gaps)
    next_index = compute_next_index(dst)
    print(f"SRC1 count: {n1}; next index for SRC2: {next_index} -> {to_basename(next_index)}")

    # Step 2: Move and reindex src2 into dst starting at next_index
    print("-- Moving SRC2 into DST continuing the index --")
    n2 = process_src2_move(src2, dst, start_index=next_index, dry_run=args.dry_run)

    final_first = to_basename(0)
    final_last = to_basename(n1 + n2 - 1) if (n1 + n2) > 0 else "s000000"
    print(f"Done. Total merged: {n1 + n2}. Range: {final_first} .. {final_last}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

