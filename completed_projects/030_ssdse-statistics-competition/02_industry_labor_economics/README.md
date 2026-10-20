# 02_industry_labor_economics（産業別労働経済分析）

## 目的
産業別の失業者数を被説明変数、勤続年数・超過実労働時間数・所定内給与額（＋年度ダミー・業界ダミー）を説明変数としたOLS回帰分析。雇用の安定性・労働需要・産業競争力が失業者数に与える影響を定量化する。

方針の詳細は [`Docs/方針書.md`](Docs/方針書.md)、複数業界プーリングモデルの構造は [`Docs/複数業界プーリングモデルの構造.md`](Docs/複数業界プーリングモデルの構造.md)、時系列の解釈は [`Docs/解釈_06_04.md`](Docs/解釈_06_04.md) を参照。

## 状況
モデルは単一業界版（`01_ols_policy1.py`）→ 固定効果版（`02_ols_policy1_fixed.py`）→ 複数業界プーリング版（`03_ols_multiindustry.py`）と発展させて進行中（`Data/model_results/` に各段階の結果）。

## フォルダ構成
- `Data/USE`, `Data/covert_use*` — 元データ→共通年抽出→区分統一、の変換段階ごとの中間データ（変換フローは [`Docs/データスケール問題の図式説明.md`](Docs/データスケール問題の図式説明.md) 参照）
- `Data/model_results/` — 各モデル段階のOLS結果（`*.txt`）と分析用テーブル（`*.csv`）
- `Data/Output/`, `Data/shitugyoushasu_時系列/`, `Data/倒産件数/` — 可視化出力（時系列グラフ等）
- `Data/変換前データ/`, `Data/学歴、年齢階級別...` — 元データ（政府統計）とその変換スクリプト
- `scripts/create_model/` — OLSモデル構築スクリプト（`01`〜`03` が本流、`99_*` は診断用）
- `scripts/preprocessing/` — 前処理ノートブック

## 注意
スクリプト内の絶対パスは `NEW_SSDSE` フォルダ名を前提にしていたため、`SSDSE/02_industry_labor_economics` へ再配置した際にあわせて更新済み。再実行時は環境（pandas, statsmodels等）が整っているか確認すること。
