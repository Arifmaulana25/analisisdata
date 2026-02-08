"""
Utility to merge duplicate rows in a DataFrame based on Asal, Tujuan, Komoditas
and summing Volume. Designed to be imported by `preprocess.py` or used as a CLI.
"""
import re
import pandas as pd
from typing import Tuple, Optional


def parse_volume(val) -> float:
    """Parse a volume value into float.

    Handles thousands separators like '63.500' and decimal commas like '1,5'.
    Returns 0.0 on failure.
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s == '':
        return 0.0
    s = s.replace(' ', '')

    # If both separators exist, assume '.' thousands and ',' decimal
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    else:
        # If multiple dots and no comma, likely dot is thousands separator
        if s.count('.') > 1 and ',' not in s:
            s = s.replace('.', '')
        else:
            # If single dot and three digits after it -> thousands sep
            if '.' in s and len(s.split('.')[-1]) == 3 and ',' not in s:
                s = s.replace('.', '')
            else:
                # replace comma with dot for decimal
                s = s.replace(',', '.')

    # strip any non numeric chars except minus and dot
    s = re.sub(r"[^0-9\.-]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ''
    s = str(s).strip().lower()
    # remove punctuation commonly seen in place names
    s = re.sub(r"[\.,]", '', s)
    # collapse whitespace
    s = re.sub(r"\s+", ' ', s)
    return s


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """Return names of columns to use as (asal, tujuan, komoditas, volume).

    The function searches for common permutations and falls back to column
    name heuristics if exact names not found.
    """
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]

    origin_keys = ['daerah asal', 'daerah_asal', 'asal', 'origin']
    destination_keys = ['tujuan', 'daerah tujuan', 'destination']
    commodity_keys = ['komoditas', 'komodity', 'komoditas/jenis', 'komoditi', 'komodities']
    volume_keys = ['volume', 'jumlah', 'qty', 'vol']

    def find_key(keys):
        for k in keys:
            if k in lower_cols:
                return cols[lower_cols.index(k)]
        # fuzzy match: look for key as substring
        for k in keys:
            for i, c in enumerate(lower_cols):
                if k in c:
                    return cols[i]
        return None

    asal_col = find_key(origin_keys) or cols[0]
    tujuan_col = find_key(destination_keys) or (cols[1] if len(cols) > 1 else cols[0])
    komoditas_col = find_key(commodity_keys) or (cols[2] if len(cols) > 2 else cols[0])
    volume_col = find_key(volume_keys) or cols[-1]

    return asal_col, tujuan_col, komoditas_col, volume_col


def merge_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate rows by (asal, tujuan, komoditas) and sum volume.

    Returns a new DataFrame with deduplicated rows. Non-key columns are kept
    using the first value (you can change this behaviour if needed).
    """
    if df.empty:
        return df

    asal_col, tujuan_col, komoditas_col, volume_col = detect_columns(df)

    # Create normalized keys
    df['_asal_norm'] = df[asal_col].apply(normalize_text)
    df['_tujuan_norm'] = df[tujuan_col].apply(normalize_text)
    df['_komoditas_norm'] = df[komoditas_col].apply(normalize_text)

    # Parse volumes safely
    df['_volume_num'] = df[volume_col].apply(parse_volume)

    # Group and aggregate
    grouped = df.groupby(['_asal_norm', '_tujuan_norm', '_komoditas_norm'], dropna=False)

    agg_dict = {volume_col: ('_volume_num', 'sum')}

    # For other columns, take first non-null value
    for c in df.columns:
        if c in ['_asal_norm', '_tujuan_norm', '_komoditas_norm', '_volume_num']:
            continue
        if c == volume_col:
            continue
        agg_dict[c] = (c, 'first')

    merged = grouped.agg(**agg_dict).reset_index(drop=True)

    # Rename summed volume
    merged = merged.rename(columns={volume_col: 'Volume_Sum'})

    # Optionally restore readable Asal/Tujuan/Komoditas from originals (first)
    # If those original columns were kept by 'first' they exist in merged already.

    # Clean helper columns if present
    for c in ['_asal_norm', '_tujuan_norm', '_komoditas_norm', '_volume_num']:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    return merged


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Merge duplicate rows in Excel/CSV')
    parser.add_argument('input', help='Input file (xlsx or csv)')
    parser.add_argument('--output', '-o', help='Output file path (xlsx or csv). If omitted, prints a summary')
    args = parser.parse_args()

    if args.input.lower().endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input)

    merged = merge_duplicates(df)

    if args.output:
        if args.output.lower().endswith('.csv'):
            merged.to_csv(args.output, index=False)
        else:
            merged.to_excel(args.output, index=False)
        print(f"Saved merged file to {args.output}")
    else:
        print(f"Original rows: {len(df)}, Merged rows: {len(merged)}")
