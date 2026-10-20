#!/usr/bin/env python3
"""
Phase 0: e-Stat からのデータ取得（Web直取得、API不使用）

取得メカニズム（実証済み）:
  https://www.e-stat.go.jp/stat-search/file-download?statInfId=<ID>&fileKind=0
  - fileKind=0 が実ファイル（多くは BIFF .xls / OLE2）。CSV(fileKind=1)は無い表が多い。
  - .xls は pandas + engine='xlrd' で読める。文字は cp932 だが pandas/xlrd が unicode 化する
    （コンソール表示は PYTHONUTF8=1 が必要）。

使い方:
  PYTHONUTF8=1 python scripts/collect/fetch_estat.py           # 登録済みを取得
  PYTHONUTF8=1 python scripts/collect/fetch_estat.py <key>      # 個別取得

各データの実在・粒度・対応構成概念・状況は docs/データ取得手順書.md を参照。
"""
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw" / "estat"
RAW.mkdir(parents=True, exist_ok=True)

DL = "https://www.e-stat.go.jp/stat-search/file-download?statInfId={sid}&fileKind=0"

# 確認済みデータ登録簿（key: (statInfId, 保存名, 内容メモ)）
DATASETS = {
    # 構成概念2：雇用の安定性（離職率）— 都道府県×産業(大分類)×企業規模×性別。実用可。
    "koyoudoukou_rishoku": ("000031612648", "koyoudoukou_table5_rishoku.xls",
                             "雇用動向 表5 都道府県・産業大分類別 離職率"),
    # 構成概念3候補：職業安定2-6は【新規学卒者】の求人・就職で一般労働需給には不適（要代替）。
    "shokugyo_shinsotsu_2_6": ("000031877861", "shokugyo_2-6_shinsotsu.xls",
                               "職業安定2-6 産業別・都道府県別 求人/就職（新規学卒者・不適）"),
}


def fetch(sid: str, dest: Path) -> bool:
    url = DL.format(sid=sid)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if data[:4] == b"\xd0\xcf\x11\xe0" or data[:2] == b"PK":  # OLE2(.xls) or zip(.xlsx)
            dest.write_bytes(data)
            print(f"  ✓ {dest.name}  ({len(data)//1024} KB)")
            return True
        print(f"  ✗ {dest.name}  想定外の中身（HTML等, {len(data)} bytes）")
        return False
    except Exception as e:
        print(f"  ✗ {dest.name}  {type(e).__name__}: {e}")
        return False


def main(keys):
    for k in keys:
        if k not in DATASETS:
            print(f"[skip] 未登録キー: {k}")
            continue
        sid, name, memo = DATASETS[k]
        print(f"[{k}] {memo}")
        fetch(sid, RAW / name)
        time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS))
