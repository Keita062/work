#!/usr/bin/env python3
"""
Step 4（部分実装）: 都道府県別 LCAクラス曝露度の算出と7×7グリッド可視化

要件定義書 Step 4 の「①曝露度指標の算出」「②都道府県別7×7グリッド可視化」のうち、
取得済みデータ（国勢調査 産業大分類×都道府県）で実施可能な部分のみを実装する。

- 国勢調査（2020年・就業者数）の産業大分類（A〜T）を Step 3 の9業種に対応付け
- 各業種の LCA クラス（Class 1=安定型 / Class 2=変動型）を付与
- 都道府県ごとに Class 1 / Class 2 への就業者曝露度（合計100%）を算出
- 47都道府県を 7×7 グリッド（全国コード順・上段=北 / 下段=南）の積み上げ棒で可視化

空間統計（Moran's I）・GISマッピング・他の地理観測変数は地理データ取得後の別タスク。
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# ---- 日本語フォント設定（Windows）----
matplotlib.rcParams['font.sans-serif'] = [
    'Yu Gothic', 'Meiryo', 'MS Gothic', 'Noto Sans CJK JP', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False

# ---- パス設定 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = DATA_PROCESSED / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CENSUS_CSV = DATA_RAW / "census_industry_prefecture.csv"
CLASS_MAP_CSV = DATA_PROCESSED / "step3_industry_class_mapping.csv"
OUT_CSV = DATA_PROCESSED / "step4_prefecture_class_exposure.csv"
OUT_PNG = FIGURES_DIR / "step4_prefecture_class_exposure_grid.png"

print("=" * 80)
print("[Step 4 / geo] 都道府県別 LCAクラス曝露度")
print("=" * 80)

# ============================================================================
# 産業大分類（コード）→ Step 3 の9業種 対応表
#   除外: 100総数 / 110,140,180(再掲) / 190 F電気ガス / 320 S公務 / 330 T分類不能
# ============================================================================
CODE_TO_INDUSTRY = {
    "120": "農・林・漁・鉱業",  # A農業，林業
    "130": "農・林・漁・鉱業",  # B漁業
    "150": "農・林・漁・鉱業",  # C鉱業，採石業，砂利採取業
    "160": "建設業",            # D建設業
    "170": "製造業",            # E製造業
    "200": "情報通信業",        # G情報通信業
    "210": "運輸業",            # H運輸業，郵便業
    "220": "卸売業，小売業",    # I卸売業，小売業
    "230": "金融・保険業",      # J金融業，保険業
    "240": "不動産業",          # K不動産業，物品賃貸業
    "250": "サービス業他",      # L学術研究，専門・技術サービス業
    "260": "サービス業他",      # M宿泊業，飲食サービス業
    "270": "サービス業他",      # N生活関連サービス業，娯楽業
    "280": "サービス業他",      # O教育，学習支援業
    "290": "サービス業他",      # P医療，福祉
    "300": "サービス業他",      # Q複合サービス事業
    "310": "サービス業他",      # Rサービス業（他に分類されないもの）
}

# 47都道府県（全国コード順）→ 7×7グリッド行優先配置（上段=北 / 下段=南）
PREF_ORDER = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
    "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
    "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
]

# ============================================================================
# Section 1: 業種→クラス対応（Step 3 の成果物）を読み込み
# ============================================================================
print("\n[Section 1] 業種→クラス対応表を読み込み...")
industry_class = {}
with open(CLASS_MAP_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        industry_class[r["industry"]] = int(r["dominant_class"])
print(f"✓ {len(industry_class)} 業種のクラス割当を読み込み")
class1_industries = sorted(i for i, c in industry_class.items() if c == 1)
class2_industries = sorted(i for i, c in industry_class.items() if c == 2)
print(f"  Class 1（安定型）: {', '.join(class1_industries)}")
print(f"  Class 2（変動型）: {', '.join(class2_industries)}")

# ============================================================================
# Section 2: 国勢調査（2020年・就業者数）を読み込み、都道府県×業種で集計
# ============================================================================
print("\n[Section 2] 国勢調査を読み込み・集計...")
with open(CENSUS_CSV, encoding="cp932") as f:
    census_rows = list(csv.reader(f))

# 列: [2]都道府県 [5]性別 [6]産業大分類コード [15]2020年就業者数_不詳補完値
pref_emp = defaultdict(lambda: defaultdict(float))  # pref -> industry -> employed
for r in census_rows:
    if len(r) <= 15:
        continue
    pref = r[2].strip()
    gender = r[5].strip()
    code = r[6].strip()
    if gender != "総数" or pref not in PREF_ORDER or code not in CODE_TO_INDUSTRY:
        continue
    raw = r[15].strip().replace(",", "")
    emp = 0.0 if raw in ("", "-") else float(raw)
    pref_emp[pref][CODE_TO_INDUSTRY[code]] += emp

missing = [p for p in PREF_ORDER if p not in pref_emp]
if missing:
    raise RuntimeError(f"就業者データが取得できない都道府県: {missing}")
print(f"✓ {len(pref_emp)} 都道府県 × 9業種の就業者数を集計")

# ============================================================================
# Section 3: 曝露度の算出
# ============================================================================
print("\n[Section 3] クラス曝露度を算出...")
results = []
for pref in PREF_ORDER:
    emp = pref_emp[pref]
    total = sum(emp.values())
    c1 = sum(v for ind, v in emp.items() if industry_class[ind] == 1)
    c2 = sum(v for ind, v in emp.items() if industry_class[ind] == 2)
    results.append({
        "prefecture": pref,
        "class1_share": c1 / total,
        "class2_share": c2 / total,
        "class1_employed": int(c1),
        "class2_employed": int(c2),
        "total_employed": int(total),
    })

# 全国集計（就業者数合計ベース）
nat_c1 = sum(r["class1_employed"] for r in results)
nat_total = sum(r["total_employed"] for r in results)
nat_c1_share = nat_c1 / nat_total
nat_c2_share = 1 - nat_c1_share
print(f"✓ 全国 Class 1曝露度={nat_c1_share:.1%} / Class 2曝露度={nat_c2_share:.1%}")

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    for r in results:
        w.writerow(r)
print(f"✓ 保存: {OUT_CSV.name}")

# 妥当性チェック用ランキング（Class 1曝露度の高い/低い県）
ranked = sorted(results, key=lambda r: r["class1_share"], reverse=True)
print("\nClass 1（安定型）曝露度 上位5県:")
for r in ranked[:5]:
    print(f"  {r['prefecture']}: Class1={r['class1_share']:.1%}  Class2={r['class2_share']:.1%}")
print("Class 1（安定型）曝露度 下位5県:")
for r in ranked[-5:]:
    print(f"  {r['prefecture']}: Class1={r['class1_share']:.1%}  Class2={r['class2_share']:.1%}")

# ============================================================================
# Section 4: 7×7グリッド積み上げ棒グラフ
# ============================================================================
print("\n[Section 4] 7×7グリッド図を生成...")
C2_COLOR = "#d9534f"  # Class 2（変動型・ターゲット集中）= 目立つ赤
C1_COLOR = "#5bc0de"  # Class 1（安定型）= 落ち着いた青
res_by_pref = {r["prefecture"]: r for r in results}

fig, axes = plt.subplots(7, 7, figsize=(15, 15))
for idx in range(49):
    ax = axes[idx // 7][idx % 7]
    if idx >= len(PREF_ORDER):
        ax.axis("off")
        continue
    pref = PREF_ORDER[idx]
    r = res_by_pref[pref]
    c1, c2 = r["class1_share"], r["class2_share"]
    ax.bar(0, c2, width=0.8, color=C2_COLOR)
    ax.bar(0, c1, width=0.8, bottom=c2, color=C1_COLOR)
    # Class 1 の厚み（＝県間差の主因）を数値表示
    ax.text(0, 1.02, f"C1 {c1:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_title(pref, fontsize=10, pad=2)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# 凡例・タイトル
handles = [
    plt.Rectangle((0, 0), 1, 1, color=C2_COLOR),
    plt.Rectangle((0, 0), 1, 1, color=C1_COLOR),
]
fig.legend(
    handles,
    ["Class 2（変動型：ターゲット3業種を含む）", "Class 1（安定型：農林漁鉱・金融保険）"],
    loc="lower center", ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.01),
)
fig.suptitle(
    "都道府県別 LCAクラス曝露度（国勢調査2020・就業者数ベース、7×7グリッド）\n"
    f"全国平均: Class 2 = {nat_c2_share:.0%} / Class 1 = {nat_c1_share:.0%}"
    "（配置は全国コード順＝上段が北・下段が南）",
    fontsize=14,
)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"✓ 保存: {OUT_PNG.name}")

print("\n" + "=" * 80)
print("✓ Step 4（部分：曝露度算出＋7×7グリッド）完了")
print("=" * 80)
