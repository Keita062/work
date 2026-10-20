"""ステップ0の類型（k-means）。類型の数はエルボー法で決める。ステップ0〜4・改善候補・ステップ6で共通に使う"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from wb_common import PROC, ROOT

SSDSE_A = ROOT.parent / "SSDSE/01_election_pledges/data/公的データ/2025/SSDSE-A-2025.csv"
VARS = ["log_pop", "log_density", "aging_rate", "fiscal_index", "primary_share", "habitable_share"]
K_RANGE = range(1, 13)
# 人口密度の中央値が高い順に付ける名前（k=3 のとき）
NAMES3 = ["都市部", "地方の中心都市", "農山村の小都市"]


def structure(codes):
    """5桁コードの一覧 → 類型づくりの6変数（kpi_municipal.csv＋SSDSE-A 2025）"""
    k = pd.read_csv(PROC / "kpi_municipal.csv", dtype={"code": str}).set_index("code")
    a = pd.read_csv(SSDSE_A, encoding="cp932", header=None, skiprows=3)
    a.columns = pd.read_csv(SSDSE_A, encoding="cp932", header=None, nrows=1).iloc[0].values
    a["code"] = a.iloc[:, 0].str[1:]
    a = a.set_index("code")
    f = lambda c: pd.to_numeric(a[c], errors="coerce")
    d = pd.DataFrame(index=pd.Index(codes, name="code"))
    d["pop2020"] = k["pop2020"]
    d["log_pop"] = np.log(k["pop2020"])
    d["pop_density"] = k["pop_density"]
    d["log_density"] = np.log(k["pop_density"])
    d["aging_rate"] = k["aging_rate"]
    d["fiscal_index"] = k["fiscal_index"]
    d["primary_share"] = f("F2201") / f("F1102") * 100
    d["habitable_share"] = f("B1103") / f("B1101") * 100
    return d


def elbow(z):
    """k=1〜12 のクラスタ内平方和と、エルボー（始点と終点を結ぶ直線から最も離れた点）の k"""
    ks = list(K_RANGE)
    inert = np.array([KMeans(k, n_init=50, random_state=0).fit(z).inertia_ for k in ks])
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (inert - inert.min()) / (inert.max() - inert.min())
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0]) / np.hypot(y[-1] - y[0], x[-1] - x[0])
    return ks[int(dist.argmax())], dict(zip(ks, inert.round(1).tolist()))


def typology(codes, names=None):
    """codes → 類型名の Series（index＝code）、標準化した6変数、6変数の表、エルボー法の結果"""
    d = structure(codes)
    assert d[VARS].notna().all().all(), d[d[VARS].isna().any(axis=1)]
    z = pd.DataFrame(StandardScaler().fit_transform(d[VARS]), index=d.index, columns=VARS)
    k, inert = elbow(z)
    assert k == 3, f"エルボー法の k={k}（類型名は k=3 を前提に付けている）"
    cl = pd.Series(KMeans(k, n_init=100, random_state=0).fit_predict(z), index=d.index)
    order = d.groupby(cl.values)["pop_density"].median().sort_values(ascending=False).index
    lab = {c: NAMES3[i] for i, c in enumerate(order)}
    return cl.map(lab), z, d, {"k": k, "inertia": inert}
