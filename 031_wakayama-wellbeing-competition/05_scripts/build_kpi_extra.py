"""個別KPIの追加分（上位5分野以外）を市区町村単位に整形する。

kpi_municipal.csv（build_kpi_panel.py）は上位5分野＋医療施設徒歩圏・完全失業率だけを持つ。
09-29版アーティファクトで使った残りの分野のKPIを、手元の一次資料から作り直す。

入力: 04_data/kpi_raw/mlit_toshi_monitoring_zentai.xlsx（都市モニタリングシート R6）
      04_data/kpi_raw/keizaicensus2021_b1_009_1a.xlsx（経済センサス2021 産業小分類別事業所数）
      SSDSE-A-2025（completed_projects/030_ssdse-statistics-competition/01_election_pledges/data/公的データ/2025/）
出力: 04_data/processed/kpi_extra.csv … 1行＝1市区町村（5桁コード）
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "04_data" / "kpi_raw"
OUT = ROOT / "04_data" / "processed"
SSDSE_A = (
    ROOT.parent
    / "completed_projects/030_ssdse-statistics-competition/01_election_pledges"
    / "data/公的データ/2025/SSDSE-A-2025.csv"
)


def code5(s):
    s = str(s).strip()
    m = re.match(r"^(\d{5})_", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d{6}", s):
        return s[:5]
    if re.fullmatch(r"\d{5}", s):
        return s
    return None


def num(x):
    return pd.to_numeric(pd.Series(x).replace({"-": 0, "…": np.nan, "x": np.nan, "X": np.nan, "***": np.nan}),
                         errors="coerce").values


out = {}

# ---------------------------------------------------------------- 都市モニタリングシート
mo = pd.read_excel(RAW / "mlit_toshi_monitoring_zentai.xlsx", sheet_name=0, header=None, dtype=str)
hdr = mo.iloc[8].tolist()
MO = {"指標274": "welfare_walk_cover",        # 福祉施設徒歩圏人口カバー率
      "指標275": "commercial_walk_cover",     # 商業施設徒歩圏人口カバー率
      "指標276": "station_bus_walk_cover",    # 駅またはバス停徒歩圏人口カバー率
      "指標296": "small_car_km_per_capita",   # 一人当たりの小型車走行台キロ
      "指標311": "nursery_walk_cover"}        # 保育所徒歩圏0〜4歳人口カバー率
ci = {k: hdr.index(k) for k in MO}
mo = mo.iloc[9:]
mo["code"] = mo[7].map(code5)
mo = mo.dropna(subset=["code"]).drop_duplicates("code").set_index("code")
for k, name in MO.items():
    out[name] = pd.Series(num(mo[ci[k]]), index=mo.index)

# ---------------------------------------------------------------- SSDSE-A 2025
a = pd.read_csv(SSDSE_A, encoding="cp932", header=None, skiprows=3)
a.columns = pd.read_csv(SSDSE_A, encoding="cp932", header=None, nrows=1).iloc[0].values
a["code"] = a.iloc[:, 0].str[1:]
a = a.set_index("code")
f = lambda c: pd.to_numeric(a[c], errors="coerce")
hab = f("B1103") / 100                                  # 可住地面積（ha → km²）
out["elem_school_per_hab"] = f("E2101") / hab           # 可住地面積あたり小学校数
out["jhs_per_hab"] = f("E3101") / hab                   # 同 中学校数
out["hs_per_hab"] = f("E4101") / hab                    # 同 高等学校数
out["kinder_per_hab"] = f("E1101") / hab                # 同 幼稚園数
out["restaurant_per_hab"] = f("H6131") / hab            # 同 飲食店数
out["edu_exp_share"] = f("D320310") / f("D3203") * 100  # 歳出総額の教育費割合
out["debt_service_ratio"] = f("D2211")                  # 実質公債費比率
pop = f("A1101")

# ---------------------------------------------------------------- 経済センサス2021（生活関連サービス業，娯楽業の事業所数）
k = pd.read_excel(RAW / "keizaicensus2021_b1_009_1a.xlsx", sheet_name=0, header=None, dtype=str)
col = k.iloc[5].tolist().index("N_生活関連サービス業，娯楽業")
k = k.iloc[9:]
k["code"] = k[1].map(code5)
k = k.dropna(subset=["code"]).drop_duplicates("code").set_index("code")
out["leisure_est_per_10k"] = pd.Series(num(k[col]), index=k.index) / pop * 1e4

df = pd.DataFrame(out)
df.index.name = "code"
df.to_csv(OUT / "kpi_extra.csv", encoding="utf-8-sig")
print(df.shape)
print(df.notna().sum().to_string())
