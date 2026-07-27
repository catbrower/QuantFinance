#!/usr/bin/env python3
"""Unzip the CFTC report archives in data/ and combine each group into one CSV.

Three groups are recognised, by zip-name prefix:

    com_disagg_txt_*   Disaggregated, Futures-and-Options Combined   (191 columns)
    com_fin_txt_*  |   Traders in Financial Futures,                 ( 87 columns)
    fin_com_txt_*  |   Futures-and-Options Combined
    dea_cit_txt_*      Commodity Index Trader Supplement             ( 54 columns)

Note the financial group ships under two prefixes: the annual files are `com_fin_txt_*`
but the 2006-2016 historical bundle is `fin_com_txt_2006_2016` -- same schema, reversed
name. Both are folded into one group.

The three groups have DIFFERENT schemas, so they cannot be stacked into a single file.
The script checks this rather than assuming it, and only writes a combined-everything
file if the headers really do match.

Rows are streamed with the csv module rather than pandas so the original values survive
verbatim -- the 2006-2016 files store numbers as "112602.000000" while the annual files
use space-padded integers, and a dataframe round-trip would silently rewrite both.
Only surrounding whitespace is stripped, since padding carries no meaning in a CSV.

Two columns are appended to every row:
    report_date  -- the group's date column normalised to YYYY-MM-DD
    year         -- calendar year, so the data is still sliceable if a date fails to parse
    source_zip   -- provenance; drop it if you want the schema untouched

Usage:
    python combine_cftc.py [--data-dir data] [--out-dir data/combined] [--keep-extracted]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

csv.field_size_limit(10_000_000)

GROUP_PREFIXES = {
    "com_disagg_txt": "disaggregated",
    "com_fin_txt": "financial",
    "fin_com_txt": "financial",      # 2006-2016 bundle, same schema, reversed name
    "dea_cit_txt": "index_trader",
}

# Preferred date column per group, most specific first. Whichever appears is used.
DATE_COLUMNS = [
    "Report_Date_as_YYYY-MM-DD",
    "As_of_Date_In_Form_YYYY-MM-DD",
    "Report_Date_as_YYYY_MM_DD",
]
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y", "%Y%m%d"]

EXTRA_COLUMNS = ["report_date", "year", "source_zip"]


def group_of(zip_name: str):
    for prefix, group in GROUP_PREFIXES.items():
        if zip_name.startswith(prefix):
            return group
    return None


def parse_date(raw: str):
    s = (raw or "").strip().strip('"')
    if not s:
        return "", ""
    for fmt in DATE_FORMATS:
        try:
            d = datetime.strptime(s, fmt)
            return d.strftime("%Y-%m-%d"), str(d.year)
        except ValueError:
            continue
    return s, (s[:4] if s[:4].isdigit() else "")


def extract_all(data_dir: Path, work_dir: Path):
    """Unzip every archive into work_dir/<group>/<zipstem>/, return {group: [txt paths]}."""
    found = defaultdict(list)
    skipped = []
    for zip_path in sorted(data_dir.glob("*.zip")):
        group = group_of(zip_path.name)
        if group is None:
            skipped.append(zip_path.name)
            continue
        dest = work_dir / group / zip_path.stem
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".txt")]
            if not members:
                skipped.append(f"{zip_path.name} (no .txt inside)")
                continue
            for m in members:
                zf.extract(m, dest)
                found[group].append((dest / m, zip_path.name))
    return found, skipped


def combine_group(group, files, out_path):
    """Stream every file into one CSV. Returns a summary dict."""
    header = None
    date_col_idx = None
    date_col_name = None
    rows_written = 0
    per_file = []
    mismatches = []

    with out_path.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.writer(out_fh)
        for path, zip_name in sorted(files):
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.reader(fh)
                try:
                    cols = [c.strip().strip('"') for c in next(reader)]
                except StopIteration:
                    continue

                if header is None:
                    header = cols
                    date_col_name = next((c for c in DATE_COLUMNS if c in header), None)
                    date_col_idx = header.index(date_col_name) if date_col_name else None
                    writer.writerow(header + EXTRA_COLUMNS)   # header written exactly once
                elif cols != header:
                    mismatches.append((path.name, zip_name, len(cols)))
                    continue

                n = 0
                for row in reader:
                    if not row or all(not c.strip() for c in row):
                        continue
                    row = [c.strip() for c in row]
                    if len(row) < len(header):
                        row += [""] * (len(header) - len(row))
                    elif len(row) > len(header):
                        row = row[: len(header)]
                    raw = row[date_col_idx] if date_col_idx is not None else ""
                    iso, year = parse_date(raw)
                    writer.writerow(row + [iso, year, zip_name])
                    n += 1
                rows_written += n
                per_file.append((zip_name, path.name, n))

    return {
        "group": group, "out": out_path, "header": header,
        "date_col": date_col_name, "rows": rows_written,
        "files": per_file, "mismatches": mismatches,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data/combined")
    ap.add_argument("--keep-extracted", action="store_true",
                    help="leave the unzipped .txt files on disk")
    ap.add_argument("--groups", nargs="+", metavar="G",
                    choices=sorted(set(GROUP_PREFIXES.values())),
                    help="only process these groups (default: all present). "
                         "Other groups' existing output files are left untouched.")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    if not data_dir.is_dir():
        sys.exit(f"no such directory: {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_extracted"

    print(f"unzipping from {data_dir} ...")
    found, skipped = extract_all(data_dir, work_dir)
    if args.groups:
        dropped = [g for g in found if g not in args.groups]
        found = {g: f for g, f in found.items() if g in args.groups}
        if dropped:
            print(f"  (--groups) not processing: {', '.join(sorted(dropped))}")
        missing = [g for g in args.groups if g not in found]
        if missing:
            print(f"  (--groups) requested but no archives found: {', '.join(missing)}")
    for group, files in sorted(found.items()):
        print(f"  {group:<14} {len(files):>2} files")
    if skipped:
        print(f"  skipped (unrecognised prefix): {skipped}")
    if not found:
        sys.exit("no recognised archives found")

    print("\ncombining ...")
    results = []
    for group, files in sorted(found.items()):
        res = combine_group(group, files, out_dir / f"cftc_{group}.csv")
        results.append(res)
        size_mb = res["out"].stat().st_size / 1e6
        print(f"  {group:<14} {res['rows']:>8,} rows  {len(res['header']):>3} cols  "
              f"{size_mb:>7.1f} MB  date={res['date_col']}  -> {res['out'].name}")
        if res["mismatches"]:
            print(f"    !! header mismatch, files SKIPPED: {res['mismatches']}")

    # Only stack the groups if they genuinely share a schema.
    headers = {r["group"]: tuple(r["header"]) for r in results}
    distinct = set(headers.values())
    print("\nsingle combined file?")
    if args.groups and len(args.groups) < len(set(GROUP_PREFIXES.values())):
        print("  skipped - only a subset of groups was processed this run")
    elif len(distinct) == 1:
        big = out_dir / "cftc_all.csv"
        with big.open("w", encoding="utf-8", newline="") as out_fh:
            writer = csv.writer(out_fh)
            writer.writerow(list(next(iter(distinct))) + EXTRA_COLUMNS + ["report_group"])
            for r in results:
                with r["out"].open(newline="") as fh:
                    rd = csv.reader(fh)
                    next(rd)
                    for row in rd:
                        writer.writerow(row + [r["group"]])
        print(f"  yes - identical schemas, wrote {big.name}")
    else:
        print("  NO - the three groups have different schemas, so stacking them would")
        print("  mean padding ~140 empty columns onto most rows. Kept separate:")
        for r in results:
            print(f"    {r['group']:<14} {len(r['header']):>3} columns")

    if not args.keep_extracted and work_dir.exists():
        shutil.rmtree(work_dir)
        print(f"\nremoved {work_dir} (pass --keep-extracted to keep it)")


if __name__ == "__main__":
    main()
