"""
Join MAE_data and MAE_data_nflow CSV files in the LFI_real_results tree.

For each folder under `results/LFI_real_results/...` this script finds files
matching these patterns:

  - MAE_data_{B}_{n}.csv
  - MAE_data_nflow_{B}_{n}.csv

        It parses B and n from the filename and either:

        - by default horizontally merges the Nflow column into the base
            `MAE_data_{B}_{n}.csv` as a new column named `nflow` and writes a
            new file named `MAE_data_with_nflow_{B}_{n}.csv` (preserves original),
            or
        - if no base file exists for a (B,n) pair, the script will skip
            horizontal merging for that pair.

        This preserves the same row index ordering. The script is resilient
        to the leading unnamed index column in the original CSVs by reading
        with index_col=0.

Usage:
    python join_mae_nflow.py [--root /path/to/results/LFI_real_results]

This script requires pandas.
"""

from pathlib import Path
import re
import argparse
import pandas as pd
from typing import Dict, List, Tuple

FILENAME_RE = re.compile(r"MAE_data(?:_nflow)?_(?P<B>\d+)_(?P<n>\d+)\.csv$")
BASE_RE = re.compile(r"MAE_data_(?P<B>\d+)_(?P<n>\d+)\.csv$")
NFLOW_RE = re.compile(r"MAE_data_nflow_(?P<B>\d+)_(?P<n>\d+)\.csv$")


def find_mae_files(root: Path) -> Dict[Path, List[Path]]:
    """Return mapping folder -> list of MAE csv files in that folder (recursive)."""
    folders: Dict[Path, List[Path]] = {}
    for p in root.rglob("*.csv"):
        if FILENAME_RE.search(p.name):
            folders.setdefault(p.parent, []).append(p)
    return folders


def parse_B_n_from_name(name: str) -> Tuple[int, int]:
    m = FILENAME_RE.search(name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {name}")
    return int(m.group("B")), int(m.group("n"))


def read_csv_indexed(path: Path) -> pd.DataFrame:
    """Read CSV using the first column as index if present.

    This handles files where the CSV was saved with an unnamed index
    as the first column (common in the provided data).
    """
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        df = pd.read_csv(path)
    return df


def process_folder(folder: Path) -> Tuple[int, int]:
    """Process a single folder: for each base MAE_data and matching
    MAE_data_nflow file, add a `nflow` column to the base and write
    a new file `MAE_data_with_nflow_{B}_{n}.csv`.

    Returns: (num_pairs_written, num_rows_written_total)
    """
    files = [p for p in folder.iterdir() if p.is_file() and FILENAME_RE.search(p.name)]
    if not files:
        return 0, 0

    # Map (B,n) -> paths for base and nflow
    base_map: Dict[Tuple[int, int], Path] = {}
    nflow_map: Dict[Tuple[int, int], Path] = {}
    for p in files:
        m_base = BASE_RE.search(p.name)
        m_nflow = NFLOW_RE.search(p.name)
        if m_base:
            key = (int(m_base.group("B")), int(m_base.group("n")))
            base_map[key] = p
        elif m_nflow:
            key = (int(m_nflow.group("B")), int(m_nflow.group("n")))
            nflow_map[key] = p

    total_pairs = 0
    total_rows = 0
    for key, base_path in base_map.items():
        B, n = key
        out_name = folder / f"MAE_data_with_nflow_{B}_{n}.csv"
        if key not in nflow_map:
            # No nflow file for this pair: copy base to new file (no change)
            try:
                df_base = read_csv_indexed(base_path)
                df_base.to_csv(out_name)
                total_pairs += 1
                total_rows += len(df_base)
            except Exception as e:
                print(f"Warning: failed to read/write {base_path}: {e}")
            continue

        nflow_path = nflow_map[key]
        try:
            df_base = read_csv_indexed(base_path)
            df_nflow = read_csv_indexed(nflow_path)
        except Exception as e:
            print(f"Warning: failed to read files for {key} in {folder}: {e}")
            continue

        # Extract the nflow series. nflow files often have a single column
        # with header like 'Nflow' and an index column; take the first data
        # column.
        if df_nflow.shape[1] == 0:
            print(f"Warning: nflow file {nflow_path} has no data columns")
            continue
        nflow_col = df_nflow.iloc[:, 0]

        # Align by index: if indexes differ, align by positional index
        if not df_base.index.equals(nflow_col.index):
            # Reindex nflow_col to base index by positional matching
            n = min(len(df_base), len(nflow_col))
            nflow_col_aligned = nflow_col.reset_index(drop=True).iloc[:n]
            df_base = df_base.reset_index(drop=True).iloc[:n]
            df_base["nflow"] = nflow_col_aligned.values
            # Restore default integer index on write
            df_base.to_csv(out_name)
            total_pairs += 1
            total_rows += len(df_base)
        else:
            # Same index, safe to assign by index
            df_base["nflow"] = nflow_col
            df_base.to_csv(out_name)
            total_pairs += 1
            total_rows += len(df_base)

    return total_pairs, total_rows


def main(root: Path):
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")
    folders = find_mae_files(root)
    print(f"Found {len(folders)} folders with MAE files under {root}")
    grand_pairs = 0
    grand_rows = 0
    for folder, _ in folders.items():
        pairs, rows = process_folder(folder)
        if pairs:
            print(f"Wrote {pairs} combined files in {folder} ({rows} rows total)")
        grand_pairs += pairs
        grand_rows += rows

    print(f"Done. Created/updated {grand_pairs} combined files with {grand_rows} total rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join MAE_data and MAE_data_nflow CSVs by B and n")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent / "LFI_real_results",
                        help="Root folder containing LFI_real_results (default: results/LFI_real_results)")
    args = parser.parse_args()
    main(args.root)
