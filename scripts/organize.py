"""
Dataset Structure Flattening Utility
Author: İrem Ölçün

Description:
    - Flattens nested dataset directory structure
    - Moves or copies all files from carpet subfolders
      directly into train/ or test/ root directories
    - Renames files to preserve source identity
    - Supports dry-run mode for safe preview

Use Case:
    Designed for reorganizing texture datasets prior to
    feature extraction or decoder training.
"""

import os
import shutil
from pathlib import Path

# =======================
# CONFIGURATION
# =======================
SPLIT_ROOT = Path("/home/irem/tile_decoder_training/dataset_root_split")
SUBSETS = ("train", "test")
MODE = "move"            # "move" or "copy"
DRY_RUN = False          # True → preview only
VALID_EXTS = None        # Example: {".png",".jpg",".jpeg",".tif"}  # None = keep all files


# =======================
def should_keep_file(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.name.startswith("."):
        return False
    if VALID_EXTS is None:
        return True
    return p.suffix.lower() in VALID_EXTS


def unique_name(dst_dir: Path, name: str) -> Path:
    base, ext = os.path.splitext(name)
    candidate = dst_dir / name
    i = 1
    while candidate.exists():
        candidate = dst_dir / f"{base}_{i}{ext}"
        i += 1
    return candidate


def flatten_into_subset_root(subset_dir: Path, mode: str = "move"):
    """
    Moves or copies all files from nested carpet directories
    directly into the subset root directory.
    """

    print(f"\n=== Processing subset: {subset_dir} ===")
    moved = 0

    for carpet_dir in sorted([p for p in subset_dir.iterdir() if p.is_dir()]):
        carpet_name = carpet_dir.name

        for root, dirs, files in os.walk(carpet_dir):
            root_path = Path(root)

            if root_path == subset_dir:
                continue

            if root_path == carpet_dir:
                rel = ""
            else:
                rel = root_path.relative_to(carpet_dir).as_posix().replace("/", "_")

            for fname in files:
                src = root_path / fname
                if not should_keep_file(src):
                    continue

                base, ext = os.path.splitext(fname)

                if rel:
                    target_name = f"{carpet_name}_{rel}_{base}{ext}"
                else:
                    target_name = f"{carpet_name}_{base}{ext}"

                dst = subset_dir / target_name
                if dst.exists():
                    dst = unique_name(subset_dir, dst.name)

                print(f"{'[DRY] ' if DRY_RUN else ''}{mode.upper():4}  {src}  -->  {dst}")

                if not DRY_RUN:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if mode == "move":
                        shutil.move(str(src), str(dst))
                    elif mode == "copy":
                        shutil.copy2(str(src), str(dst))
                    else:
                        raise ValueError("MODE must be 'move' or 'copy'.")
                    moved += 1

        if not DRY_RUN:
            try:
                for r, d, f in os.walk(carpet_dir, topdown=False):
                    rp = Path(r)
                    if rp.exists() and not any(rp.iterdir()):
                        rp.rmdir()
            except Exception:
                pass

    print(f"Completed: {subset_dir.name} → {moved} files processed.")


def main():
    for subset in SUBSETS:
        subset_dir = SPLIT_ROOT / subset
        if not subset_dir.is_dir():
            print(f"Not found: {subset_dir} — skipping.")
            continue
        flatten_into_subset_root(subset_dir, mode=MODE)


if __name__ == "__main__":
    main()

