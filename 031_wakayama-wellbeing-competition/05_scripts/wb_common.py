"""地域幸福度データの読み込みと、M0の抽出条件（分析スクリプト共通）"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "04_data"
PROC = DATA / "processed"
WB_XLSX = DATA / "digital_wellbeing_shikuchoson_2022_2026.xlsx"

WAKAYAMA = ["和歌山市", "岩出市", "橋本市", "海南市", "田辺市", "紀の川市"]


def load_wb():
    """主観データ・客観データの2シートを返す"""
    subj = pd.read_excel(WB_XLSX, sheet_name="主観データ")
    obj = pd.read_excel(WB_XLSX, sheet_name="客観データ")
    return subj, obj


def categories(subj):
    """24分野名（「主観_」を外した名前）をデータの列順で返す"""
    return [c[3:] for c in subj.columns if c.startswith("主観_")]


def m0_sample(subj, obj):
    """M0：主観「全体」行・2023〜26年度・回答者数30以上・区を除く・同年度の客観あり（2,116行・678地域）"""
    a = subj[(subj["区分"] == "全体") & subj["年度"].between(2023, 2026) & (subj["回答者数"] >= 30)]
    a = a[~a["市区町村"].str.contains(r"区$")]                  # 政令市の区・東京23区を除く
    ocols = [c for c in obj.columns if c.startswith("客観_")]
    m = a.merge(obj[["年度", "地域ID"] + ocols], on=["年度", "地域ID"]).dropna(subset=ocols)
    m = m.reset_index(drop=True)
    assert len(m) == 2116 and m["地域ID"].nunique() == 678, (len(m), m["地域ID"].nunique())
    return m
