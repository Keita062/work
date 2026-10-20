"""個別KPI（ガイドブック記載の出典）を市区町村単位に整形して1表にまとめる。

入力: data/kpi_raw/ に置いた一次資料（取得元は data/kpi_raw/README.md）
出力: data/processed/kpi_municipal.csv   … 1行＝1市区町村（5桁の全国地方公共団体コード）
（取得したKPI・しなかったKPIと理由は data/kpi_raw/README.md）

対象はM1の上位5分野（健康状態・自己効力感・住宅環境・地域とのつながり・公共空間）。
環境変数（人口・高齢化率・財政力指数・人口密度）もここで作る。
"""
from pathlib import Path
import re
import zipfile, io

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "kpi_raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def code5(s):
    """'01100_札幌市' / '011002' / 11002 などから5桁コードを取り出す"""
    s = str(s).strip()
    m = re.match(r"^(\d{5})_", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d{6}", s):          # 検査数字つき6桁
        return s[:5]
    if re.fullmatch(r"\d{5}", s):
        return s
    return None


def code5_nozero(s):
    """先頭の0が落ちた検査数字つき6桁（例 11002 → 01100）"""
    s = str(s).strip()
    return s.zfill(6)[:5] if re.fullmatch(r"\d{5,6}", s) else None


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def num(x):
    return pd.to_numeric(pd.Series(x).replace({"-": 0, "…": np.nan, "x": np.nan, "X": np.nan, "***": np.nan}),
                         errors="coerce").values


# ---------------------------------------------------------------- 市区町村マスタ（国勢調査2020）
b01 = pd.read_excel(RAW / "census2020_b01_01_pop_households.xlsx", header=None, dtype=str).iloc[15:]
master = pd.DataFrame({
    "code": b01[5].str.strip(),
    "pref": b01[4].str.replace(r"^\d+_", "", regex=True),
    "name": b01[6].str.replace(r"^\d+_", "", regex=True).str.strip(),
    "pop2020": num(b01[7]),
    "area_km2": num(b01[14]),
})
master = master[master["code"].str.fullmatch(r"\d{5}") & (master["code"].str[2:] != "000")]
master = master.drop_duplicates("code").set_index("code")
master.loc[master["pop2020"] <= 0, "pop2020"] = np.nan      # 全域避難などで人口0の町
master["pop_density"] = master["pop2020"] / master["area_km2"]
kpi = master.copy()

# ---------------------------------------------------------------- 住民基本台帳 2024-01-01（年齢×男女）
jk = pd.read_excel(RAW / "jyuki_jinko2024_age_sex.xlsx", header=None, dtype=str)
hdr = jk.iloc[1].tolist()
jk = jk.iloc[3:]
jk["code"] = jk[0].map(code5)
jk = jk[jk["code"].notna()]
age_cols = [i for i, h in enumerate(hdr) if isinstance(h, str) and "歳" in h]
old_cols = [i for i in age_cols if re.match(r"^(6[5-9]|[7-9]\d|1\d\d)歳", hdr[i])]
tot_col = hdr.index("総数")
for sex, lab in [("計", "all"), ("男", "m"), ("女", "f")]:
    d = jk[jk[3] == sex].set_index("code")
    kpi[f"pop2024_{lab}"] = pd.Series(num(d[tot_col]), index=d.index)
    kpi[f"pop65_2024_{lab}"] = pd.Series(np.nansum(np.vstack([num(d[c]) for c in old_cols]), axis=0), index=d.index)
kpi["aging_rate"] = kpi["pop65_2024_all"] / kpi["pop2024_all"] * 100

# ---------------------------------------------------------------- 財政力指数（令和5年度 市町村別決算状況調）
fis = []
for f in ["soumu_kessan_r05_gaikyo_a.xlsx", "soumu_kessan_r05_gaikyo_b.xlsx"]:
    d = pd.read_excel(RAW / f, header=None, dtype=str).iloc[10:]
    d = pd.DataFrame({"code": d[14].map(code5), "v": num(d[34])}).dropna()
    fis.append(d)
fis = pd.concat(fis).drop_duplicates("code").set_index("code")["v"]
kpi["fiscal_index"] = fis

# ================================================================ 健康状態
# 平均寿命（市区町村別生命表 2020）と 65歳時の定常人口 T65
lt = pd.read_excel(RAW / "mhlw_seimeihyo_shikuchoson_2020.xlsx", sheet_name=None, header=None, dtype=str)
life = {}
for sh, d in lt.items():
    if not sh.startswith("生命表"):
        continue
    cur, sex = None, None
    for r in d.itertuples(index=False):
        c0 = str(r[0]).strip()
        if c0.startswith("表") and re.fullmatch(r"表\d{5}", c0):
            cur = c0[1:]
            continue
        if c0 in ("男", "女"):
            sex = "m" if c0 == "男" else "f"
            continue
        if cur is None or sex is None:
            continue
        if c0 == "0":
            life.setdefault(cur, {})[f"e0_{sex}"] = fnum(r[6])
            life[cur][f"l0_{sex}"] = fnum(r[2])
        elif re.match(r"^65", c0):
            life.setdefault(cur, {})[f"T65_{sex}"] = fnum(r[5])
life = pd.DataFrame(life).T
for c in life.columns:
    kpi[c] = life[c]
kpi = kpi.rename(columns={"e0_m": "life_exp_m", "e0_f": "life_exp_f"})

# 要介護2以上の認定率（介護保険事業状況報告 2024年6月末・保険者別・第1号被保険者）
def kaigo(f):
    d = pd.read_excel(RAW / f, header=None, dtype=str)
    d = d.iloc[7:]
    d = d[d[0].notna() & d[1].notna()]
    # 列10〜16 が（再掲）第1号被保険者の 要支援1,2・要介護1〜5
    care2 = np.nansum(np.vstack([num(d[c]) for c in range(13, 17)]), axis=0)
    return pd.DataFrame({"pref": d[0].str.strip(), "name": d[1].str.strip(), "care2": care2})

km, kf = kaigo("mhlw_kaigo_2406-h2-2.xlsx"), kaigo("mhlw_kaigo_2406-h2-3.xlsx")
name_key = master.reset_index()[["code", "pref", "name"]]
for lab, d in [("m", km), ("f", kf)]:
    d = d.merge(name_key, on=["pref", "name"], how="inner").set_index("code")
    kpi[f"care2_{lab}"] = d["care2"]
    kpi[f"care2_rate_{lab}"] = kpi[f"care2_{lab}"] / kpi[f"pop65_2024_{lab}"] * 100
    # 健康寿命（近似）＝ 平均寿命 − 65歳以上の生存年数 × 要介護2以上の割合
    # 年齢別の認定率が保険者別に無いため、65歳以上で一律の割合を当てる（サリバン法の簡略版）
    kpi[f"healthy_life_{lab}"] = kpi[f"life_exp_{lab}"] - kpi[f"T65_{lab}"] / kpi[f"l0_{lab}"] * kpi[f"care2_rate_{lab}"] / 100

# ================================================================ 自己効力感
mig = pd.read_excel(RAW / "jyuki_idou2024_11-3_net_migration.xlsx", header=None, dtype=str)
mig = mig[(mig[2] == "移動者")]
mig = pd.DataFrame({"code": mig[5].map(code5), "net": num(mig[7])}).dropna(subset=["code"])
kpi["net_migration_rate"] = mig.drop_duplicates("code").set_index("code")["net"] / kpi["pop2020"] * 100

lab_ = pd.read_excel(RAW / "census2020_c01_02_labour.xlsx", header=None, dtype=str).iloc[10:]
lab_ = lab_[(lab_[3] == "0_総数") & (lab_[4] == "00_総数")]
lab_ = pd.DataFrame({"code": lab_[2].map(code5), "labour": num(lab_[6]), "unemp": num(lab_[12])}).dropna(subset=["code"])
lab_ = lab_.drop_duplicates("code").set_index("code")
kpi["unemployed_per_pop"] = lab_["unemp"] / kpi["pop2020"] * 100
kpi["unemployment_rate"] = lab_["unemp"] / lab_["labour"] * 100

# ================================================================ 住宅環境（住宅・土地統計調査 2023）
e010 = pd.read_excel(RAW / "jyutaku2023_e010_4_floor_area.xlsx", header=None, dtype=str).iloc[7:]
e010 = e010[(e010[2] == "0_総数") & (e010[3] == "0_総数") & (e010[4] == "0_総数")]
kpi["floor_area_per_dwelling"] = pd.Series(num(e010[10]), index=e010[1].map(code5).values).groupby(level=0).first()

e008 = pd.read_excel(RAW / "jyutaku2023_e008_4_tenure.xlsx", header=None, dtype=str).iloc[9:]
e008["code"] = e008[1].map(code5)
tot = e008[e008[2] == "0_総数"].set_index("code")
det = e008[e008[2] == "1_一戸建"].set_index("code")
kpi["detached_owner_share"] = pd.Series(num(det[4]), index=det.index) / pd.Series(num(tot[3]), index=tot.index) * 100

e122 = pd.read_excel(RAW / "jyutaku2023_e122_4_rent.xlsx", header=None, dtype=str).iloc[9:]
e122 = e122[e122[2] == "0_総数"]
kpi["rent_per_m2"] = pd.Series(num(e122[16]), index=e122[1].map(code5).values).groupby(level=0).first()

# ================================================================ 地域とのつながり
# 自殺者数（地域における自殺の基礎資料 2023・住居地）
sui = pd.read_excel(RAW / "mhlw_jisatsu_r5_A7_municipal_residence.xls", sheet_name=0, header=None, dtype=str).iloc[8:]
sui = pd.DataFrame({"code": sui[1].map(code5_nozero), "n": num(sui[4])}).dropna(subset=["code"])
sui_n = sui.drop_duplicates("code").set_index("code")["n"].reindex(kpi.index)
is_ward = kpi["name"].str.contains(r"市.+区$")
sui_n[sui_n.isna() & ~is_ward] = 0                        # 自殺者0人の市町村は表に載らない
kpi["suicide_per_100k"] = sui_n / kpi["pop2020"] * 1e5

b10 = pd.read_excel(RAW / "census2020_b10_household_type.xlsx", header=None, dtype=str).iloc[10:]
b10 = b10[b10[3] == "0_総数"]
b10 = pd.DataFrame({"code": b10[2].map(code5), "gen": num(b10[4]), "ext": num(b10[11]), "old1": num(b10[27])}).dropna(subset=["code"])
b10 = b10.drop_duplicates("code").set_index("code")
kpi["extended_family_share"] = b10["ext"] / b10["gen"] * 100
kpi["elderly_single_share"] = b10["old1"] / b10["gen"] * 100

mar = pd.read_excel(RAW / "census2020_4-3_marital.xlsx", header=None, dtype=str).iloc[11:]
mar = mar[(mar[3] == "0_国籍総数") & (mar[4] == "0_総数")]
mar["code"] = mar[2].map(code5)
piv = mar.pivot_table(index="code", columns=5, values=6, aggfunc="first")
tot_ = num(piv["0_総数"]); never = num(piv["1_未婚"])
unk = num(piv["5_配偶関係「不詳」"])
kpi["married_share"] = pd.Series((1 - never / (tot_ - np.nan_to_num(unk))) * 100, index=piv.index)

# 経済センサス2021：政治団体・宗教の事業所数
kc = pd.read_excel(RAW / "keizaicensus2021_b1_009_1a.xlsx", header=None, dtype=str)
lab_row = kc.iloc[5].tolist()
col_pol = next(i for i, v in enumerate(lab_row) if isinstance(v, str) and re.match(r"^932_", v))
col_rel = next(i for i, v in enumerate(lab_row) if isinstance(v, str) and re.match(r"^94_", v))
kc = kc.iloc[9:]
kc = pd.DataFrame({"code": kc[1].map(code5), "pol": num(kc[col_pol]), "rel": num(kc[col_rel])}).dropna(subset=["code"])
kc = kc.drop_duplicates("code").set_index("code")
kpi["political_org_per_10k"] = kc["pol"] / kpi["pop2020"] * 1e4
kpi["religious_est_per_10k"] = kc["rel"] / kpi["pop2020"] * 1e4

# NPO法人（内閣府NPOポータル・主たる事務所の所在地）
z = zipfile.ZipFile(RAW / "npo_gyousei_000.zip")
npo = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if n.endswith(".csv")][0])), encoding="cp932", dtype=str)
# 住所に都道府県名が無いので所轄庁（都道府県か政令市）から補う。解散・認証取消の法人は除く
npo = npo[npo["解散情報"].isna() & ~npo["監督情報"].fillna("").str.contains("取消")]
mun = master.reset_index()
mun = mun[~mun["name"].str.contains(r"市.+区$")]          # 政令市の区は市に寄せる
by_pref = {p: sorted(g["name"].tolist(), key=len, reverse=True) for p, g in mun.groupby("pref")}
code_of = {(r.pref, r.name): r.code for r in mun.itertuples()}
city_pref = {r.name: r.pref for r in mun.itertuples()}
counts, unmatched = {}, 0
for auth, a in zip(npo["所轄庁"].fillna(""), npo["主たる事務所の所在地"].fillna("")):
    a = re.sub(r"\s", "", a)
    p = auth if auth in by_pref else city_pref.get(auth)
    if p is None or not a:
        unmatched += 1
        continue
    rest = a[len(p):] if a.startswith(p) else a
    rest = re.sub(r"^[^市区町村]{1,6}郡", "", rest)
    n = next((n for n in by_pref[p] if rest.startswith(n)), None)
    if n is None and auth not in by_pref:                 # 政令市所轄で住所が区から始まる
        n = auth
    if n is None:
        unmatched += 1
        continue
    counts[code_of[(p, n)]] = counts.get(code_of[(p, n)], 0) + 1
print("NPO matched", sum(counts.values()), "unmatched", unmatched)
kpi["npo_per_10k"] = pd.Series(counts) / kpi["pop2020"] * 1e4
kpi.loc[kpi["npo_per_10k"].isna() & kpi["pop2020"].notna() & ~kpi["name"].str.contains(r"市.+区$"), "npo_per_10k"] = 0

# ================================================================ 公共空間
pk = pd.read_excel(RAW / "soumu_koukyoushisetsu_shichoson_keinen_H18-R5.xlsx", sheet_name=0, header=None, dtype=str).iloc[9:]
pk = pk[pk[0].astype(str) == "2022"]
area = np.nansum(np.vstack([num(pk[c]) for c in (12, 14, 20, 22)]), axis=0)
kpi["park_area_per_capita"] = pd.Series(area, index=pk[2].map(code5).values).groupby(level=0).first() / kpi["pop2020"]

mo = pd.read_excel(RAW / "mlit_toshi_monitoring_zentai.xlsx", sheet_name=0, header=None, dtype=str)
hdr = mo.iloc[8].tolist()
ci = {k: hdr.index(k) for k in ["指標273", "指標317", "指標323"]}
mo = mo.iloc[9:]
mo["code"] = mo[7].map(code5)
mo = mo.dropna(subset=["code"]).drop_duplicates("code").set_index("code")
kpi["medical_walk_cover"] = pd.Series(num(mo[ci["指標273"]]), index=mo.index)
kpi["sidewalk_rate"] = pd.Series(num(mo[ci["指標317"]]), index=mo.index)
kpi["park_walk_cover"] = pd.Series(num(mo[ci["指標323"]]), index=mo.index)

# ウォーカブル推進都市（国交省 2026-08-31時点の一覧PDFから抽出したテキスト）
txt = (RAW / "mlit_walkable_cities_20260831.txt").read_text(encoding="utf-8")
body = txt.split("設定した市区町村")[-1] if "設定した市区町村" in txt else txt
tokens = set(body.split())
cnt = mun.groupby("name")["code"].nunique()
uniq = set(cnt[cnt == 1].index)
walk = {code_of[(r.pref, r.name)] for r in mun.itertuples() if r.name in tokens and r.name in uniq}
kpi["walkable_city"] = np.where(kpi.index.isin(list(walk)), 1.0, 0.0)
kpi.loc[~kpi["name"].isin(uniq) & ~kpi.index.isin(list(walk)), "walkable_city"] = np.nan  # 同名があり判定できない

kpi.index.name = "code"
kpi.to_csv(OUT / "kpi_municipal.csv", encoding="utf-8-sig")
print("kpi_municipal.csv", kpi.shape)
print(kpi.notna().sum().to_string())
