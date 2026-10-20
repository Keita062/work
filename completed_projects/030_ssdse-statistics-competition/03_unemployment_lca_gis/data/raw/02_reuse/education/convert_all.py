import glob
import os
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = r"C:\GitHub\Data-Analysis_competition\SSDSE\02_industry_labor_economics\Data\変換前データ\変換後データ"
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


def transform_file(path):
    ext = os.path.splitext(path)[1].lower()
    engine = "xlrd" if ext == ".xls" else "openpyxl"
    nendo = os.path.splitext(os.path.basename(path))[0]  # 例: "令和三年"
    xl = pd.ExcelFile(path, engine=engine)
    frames = []
    for sheet_name in xl.sheet_names:          # 産業シートを1枚ずつ処理
        df = xl.parse(sheet_name, header=None)
        if len(df) <= 11:                       # データ行なしのシートはスキップ
            continue
        data = df.iloc[12:].copy()             # 行13以降がデータ（行12は単位）
        result = data[SRC_COLS].copy()
        result.columns = DST_COLS
        result.insert(1, "産業", sheet_name)  # シート名 = 産業名
        result.insert(2, "年度", nendo)
        frames.append(result[FINAL_COLS])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main():
    # .xlsx と .xls を全件取得（サブディレクトリは含まない）
    files = sorted(
        glob.glob(os.path.join(SRC_DIR, "*.xlsx")) +
        glob.glob(os.path.join(SRC_DIR, "*.xls"))
    )
    if not files:
        raise FileNotFoundError(f"Excelファイルが見つかりません: {SRC_DIR}")
    print(f"処理対象: {len(files)}ファイル")
    all_frames = []
    for path in files:
        print(f"  処理中: {os.path.basename(path)}")
        df = transform_file(path)
        if df is not None:
            all_frames.append(df)
    combined = pd.concat(all_frames, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"保存完了: {OUTPUT_CSV}  ({len(combined)}行)")


if __name__ == "__main__":
    main()
