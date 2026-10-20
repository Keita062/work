import glob
import os
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "変換後データ")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "変換後データ.csv")

SRC_COLS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]

DST_COLS = [
    "区　分",
    "10人_勤続年数_年", "10人_超過実労働時間数_時間", "10人_所定内給与額_千円", "10人_労働者数_千円",
    "1,000人以上_勤続年数_年", "1,000人以上_超過実労働時間数_時間", "1,000人以上_所定内給与額_千円", "1,000人以上_労働者数_千円",
    "100～999人_勤続年数_年", "100～999人_超過実労働時間数_時間", "100～999人_所定内給与額_千円", "100～999人_労働者数_千円",
    "10～99人勤続年数_年", "10～99人_超過実労働時間数_時間", "10～99人_所定内給与額_千円", "10～99人_労働者数_千円",
]

FINAL_COLS = ["区　分", "産業", "年度"] + DST_COLS[1:]


def parse_filename(path):
    base = os.path.basename(path)
    stem = base.rsplit(".csv", 1)[0]
    parts = stem.split(" - ", 1)
    nendo = parts[0].replace(".xlsx", "")
    sangyo = parts[1]
    return nendo, sangyo


def transform_file(path):
    nendo, sangyo = parse_filename(path)
    df = pd.read_csv(path, header=None, encoding="utf-8-sig")
    data = df.iloc[11:].copy()
    result = data[SRC_COLS].copy()
    result.columns = DST_COLS
    result.insert(1, "産業", sangyo)
    result.insert(2, "年度", nendo)
    return result[FINAL_COLS]


def main():
    csv_files = glob.glob(os.path.join(BASE_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSVファイルが見つかりません: {BASE_DIR}")
    print(f"処理対象: {len(csv_files)}ファイル")
    frames = [transform_file(p) for p in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"保存完了: {OUTPUT_CSV}  ({len(combined)}行)")


if __name__ == "__main__":
    main()
