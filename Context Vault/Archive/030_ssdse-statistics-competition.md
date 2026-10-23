# SSDSE（統計データ分析競技）

## 案件概要
- 案件ID：030
- 目的：統計データ分析競技（SSDSE）向けのリサーチ・分析。公的統計を用いた論文を提出する
- ステータス：完了（アーカイブ）
- 作業フォルダ：`completed_projects/030_ssdse-statistics-competition/`
- 発端：個人で参加。レビュアーも共同作業者もいない
- **移行メモ（2026-10-20）**：旧 `Data-Analysis_competition` リポジトリから本リポジトリへ移行。旧リポジトリ内の `NEW_SSDSE/`（産業別労働経済分析の旧作業領域）を本案件フォルダ配下の `NEW_SSDSE/` として統合済み

## 背景・経緯

- **スケジュール**: エントリー締切 2026-08-07 ／ 論文締切 2026-08-28 午前中
- **共通リソース**: [`common/`](../../completed_projects/030_ssdse-statistics-competition/common/) — 2025年実施会の過去問PDF（`2025_toukei/`）、[`2025年統計データ分析競技_完全ガイド.md`](../../completed_projects/030_ssdse-statistics-competition/common/2025年統計データ分析競技_完全ガイド.md)
- 分析テーマが複数回転換しているため、**テーマ単位のサブフォルダ**（`01_` `02_` `03_`）で管理していた

| テーマ | 状況 |
|---|---|
| `01_election_pledges` | 一区切り・保留 |
| `02_industry_labor_economics` | 進行中だった（03と並行） |
| `03_unemployment_lca_gis` | 最後の主戦場 |

---

# 03_unemployment_lca_gis（業種別LCA × GIS）— 最後の主戦場

## 概要

完全失業率を「業種の労働特性」から捉え直し、**都道府県×年を分析単位**として、ターゲット3業種（**製造業／卸売業，小売業／サービス業他**）ごとに労働市場の「型（潜在クラス）」を抽出する。型の地理分布・時間遷移（LTA）・完全失業率との調整済み関連を示し、「業種×地域タイプを特定した雇用対策」の検討材料にする。

## 正典ドキュメント

| ファイル | 役割 |
|---|---|
| [docs/要件定義書.md](../../completed_projects/030_ssdse-statistics-competition/03_unemployment_lca_gis/docs/要件定義書.md) | **要件定義の唯一の正典**（改訂版3）。初版→改訂版→改訂版2→改訂版3 を単一ファイルに集約済み。各版の遍歴は `git log --follow docs/要件定義書.md` で追う |
| [docs/分析結果まとめ_業種別LCA再設計版.md](../../completed_projects/030_ssdse-statistics-competition/03_unemployment_lca_gis/docs/分析結果まとめ_業種別LCA再設計版.md) | **実行結果の正典**（論文の素材版・実行日 2026-07-23） |
| [docs/修正点整理_分析結果まとめレビュー.md](../../completed_projects/030_ssdse-statistics-competition/03_unemployment_lca_gis/docs/修正点整理_分析結果まとめレビュー.md) | レビュー指摘 M1／M2 の構造整理（2026-07-26作成）。改訂版3の入力 |
| [docs/方針書_業種別LCA再設計.md](../../completed_projects/030_ssdse-statistics-competition/03_unemployment_lca_gis/docs/方針書_業種別LCA再設計.md) | 再設計の方針書 |
| [docs/実行結果報告書_改訂版要件定義書.md](../../completed_projects/030_ssdse-statistics-competition/03_unemployment_lca_gis/docs/実行結果報告書_改訂版要件定義書.md) | 実行ログの詳細 |
| [docs/データ取得手順書.md](../../completed_projects/030_ssdse-statistics-competition/03_unemployment_lca_gis/docs/データ取得手順書.md) | e-Stat からのデータ取得手順 |
| `docs/過去分/` | 旧設計ドキュメント11点を退避済み |

## 進捗（アーカイブ時点）

### 完成しているパイプライン（2026-07-23 実行）

生データ（すべて e-Stat から Web 直取得）→ `panel_annual.csv`（**都道府県×業種×年・10観測変数・987行**）→ 構成概念PCA（5主成分＝賃金処遇／雇用安定性／労働需給／労働時間負荷／労働力構成）→ **業種別LCA（K=5）** → **LTA（2019→2025）** → **distal（完全失業率との調整済み関連）**、加えて**都道府県別クラス所属確率レーダー（個別リスク評価）**まで実装済み。

- 旧設計（全国×業種・81レコード）はクラス分離が弱かった（相対エントロピー 0.08）。都道府県×年へ再設計して**分離 0.86〜0.92**に改善
- スクリプトは `scripts/collect/`（取得）→ `scripts/preprocess/`（パネル構築）→ `scripts/model/`（PCA・LCA・LTA・distal・所属確率）→ `scripts/geo/`（曝露度）の4層構成
- 主な成果物は `data/processed/`（`panel_annual.csv` `panel_pca_scores.csv` `lca_class_assignments.csv` `lta_transition.csv` `distal_outcome_summary.csv` `class_membership_by_pref.csv` `class_membership_pref_year.csv`）と `data/processed/figures/`（レーダー各種・LTA図）

### 2026-07-26〜08-05 のレビューと改訂版3

まとめレポートのレビューで **M1 / M2** が確定し、要件定義書に改訂版3として反映済み（commit `453cc9f`）。

- **M1（データ年次化）**: 時変なのは賃金系3変数（所定内給与額・勤続年数・超過実労働時間数、2019-2025）のみで、離職率・入職率（2016）、女性比率・就業シェア・高齢/若年就業比率（2020）、有効求人倍率・人口密度・高齢化率（2020/2022）、**完全失業率（2020）は単年の断面値を全年に複製**していた。そのため LTA の遷移が実質的に賃金だけで駆動され、滞留確率 0.96-0.97 という「見かけの時間的安定」が出ている。とくに完全失業率が単年のため **t→t+1 の時系列 distal が組めない**
- **M2（所属確率の層化）**: 連続主成分に対する GMM は分離が鋭く事後確率がほぼ 0/1 になり、所属確率が単一クラスに偏って「層（複数クラスに％分散）」にならない → **新課題 C8** として追加

### 2026-08-05 M2 先行プロトタイプ（commit `c6dcf5e`）

- `scripts/model/soften_lca_experiment.py` — `covariance_type ∈ {diag, tied, full}` × `reg_covar ∈ {1e-6 … 3e-1}` を掃引し、**層化（支配確率<0.9 の割合）と分離度（相対エントロピー）のトレードオフ**を比較。出力は `data/processed/soften_lca_experiment.csv`
- `scripts/model/soften_lca_apply.py` — 層状の所属確率の試作
- **位置づけはあくまで先行プロトタイプ**。本実装は M1（年次化）で変数・時間変動が増えた効果を確認した後（M1 → M2 の順）

## 決定済み事項

- **分析単位は都道府県×年**（旧設計の全国×業種から変更）。ターゲットは製造業／卸売業，小売業／サービス業他の3業種
- **K=5 で固定**。BIC 最小は K=7 だが過適合傾向のため、レーダー可視化を目的として K=5 を採用し、論文でその旨を丁寧に断る（課題C4・C7）
- **完全失業率は LCA の指標に含めない**。クラス決定後の distal（事後比較）として扱う（トートロジー回避）
- **賃金構造基本統計の3変数は比率補完している**。「都道府県×産業」のクロス集計が公表されないため、`県×産業×年 = 全国×産業×年 × (県・全産業×年 ÷ 全国・全産業×年)` で推定（`scripts/preprocess/build_panel.py`）。妥当性検証は残課題のまま終了
- **型は単一クラスへの決め打ちではなく、層状のソフトな所属確率として推定する**方針を改訂版3で確定（C8）
- 要件定義書は**単一ファイルに集約**し、版の遍歴は git 履歴で追う運用に変更済み

## 注意点

- 残課題（要件定義書 第2章の C1〜C8）は未解消のままアーカイブ：C2（完全失業率の業種別・年次取得）、C3（全10観測変数の2019-2025時系列化＝M1）が最優先のまま残った。C6・C8・C1・C4・C7 はさらにその後の段階
- 比率補完の検証・置換、所属割合の地図化（GIS県境界データでの塗り分け）は未着手のまま終了

## 次のアクション

- 特になし（アーカイブ済み）。再開する場合は上記「注意点」のC2・C3から着手する

---

# 02_industry_labor_economics（産業別労働経済分析）

## 概要

産業別の失業者数を被説明変数、勤続年数・超過実労働時間数・所定内給与額（＋年度ダミー・業界ダミー）を説明変数とした **OLS回帰分析**。雇用の安定性・労働需要・産業競争力が失業者数に与える影響を定量化する。

- **フォルダ**: [`02_industry_labor_economics/`](../../completed_projects/030_ssdse-statistics-competition/02_industry_labor_economics/README.md)
- 方針は [`Docs/方針書.md`](../../completed_projects/030_ssdse-statistics-competition/02_industry_labor_economics/Docs/方針書.md)、モデル構造は [`Docs/複数業界プーリングモデルの構造.md`](../../completed_projects/030_ssdse-statistics-competition/02_industry_labor_economics/Docs/複数業界プーリングモデルの構造.md)、時系列の解釈は [`Docs/解釈_06_04.md`](../../completed_projects/030_ssdse-statistics-competition/02_industry_labor_economics/Docs/解釈_06_04.md)

## 進捗（アーカイブ時点）

モデルを3段階で発展させ、各段階の結果を `Data/model_results/` に保存済み。

1. `scripts/create_model/01_ols_policy1.py` — 単一業界版
2. `scripts/create_model/02_ols_policy1_fixed.py` — 固定効果版
3. `scripts/create_model/03_ols_multiindustry.py` — 複数業界プーリング版（本流の最新）
- `99_*.py` は診断用

データは `Data/変換前データ/` → `Data/covert_use*` → `Data/USE` と段階的に変換している（変換フローは [`Docs/データスケール問題の図式説明.md`](../../completed_projects/030_ssdse-statistics-competition/02_industry_labor_economics/Docs/データスケール問題の図式説明.md)）。

## 決定済み事項

- 03テーマ（業種別LCA）が主軸になった後も、このOLS分析は並行して進めてよい方針だった（03の要件定義で確定）。03のLCAとは補完関係で、論文に定量的な裏づけとして使える想定
- 03テーマの観測変数データの一部は、ここで整備した産業別労働経済データを土台にしている

## 注意点

- スクリプト内の絶対パスは `NEW_SSDSE` フォルダ名を前提にしていたため、`02_industry_labor_economics` へ再配置した際に更新済み。再実行時は環境（pandas, statsmodels 等）が整っているか確認すること（→ [`ナレッジベース/03_フォルダ再配置時の絶対パス破損.md`](../../ナレッジベース/03_フォルダ再配置時の絶対パス破損.md)）
- 03テーマ優先のため、アーカイブ時点でアクティブには進んでいなかった

---

# 01_election_pledges（選挙公約テキスト分析）— 一区切り

## 概要

2025年衆議院・参議院選挙における各政党の公約テキストを分析し、「2025年の日本の有権者が直面する社会課題は何か、政党の政策優先度とどう対応しているか」を、テキスト分析（言及率）と SSDSE 公的統計の突き合わせから明らかにしようとしたテーマ。

- **フォルダ**: [`01_election_pledges/`](../../completed_projects/030_ssdse-statistics-competition/01_election_pledges/README.md)

## 結論（出典: [`事前分析の結論.md`](../../completed_projects/030_ssdse-statistics-competition/01_election_pledges/事前分析の結論.md)）

1. 第1優先課題: 地域経済格差による生活困窮（地方創生の言及率が最高、食料支出格差10.6%と対応）
2. 第2優先課題: 高齢化社会への不安（医療・介護、社会保障の言及率が上位）
3. 第3優先課題: 労働条件と生活保障の統合的課題（部分的対応にとどまる）

## 決定済み事項

- この結論を踏まえてコンペのテーマを一度決定したが、その後 **02（産業別労働経済分析）へ方針転換**し、本テーマは一区切り。第3優先課題（労働条件）の系譜が 02 → 03 につながっている
- 作業は時系列フォルダ（`04_28/` `05_11/` `05_12/`）＋ `election/` に残してあり、再開する場合は続きから着手できる

## 注意点

- `scripts/election/2025_shu/work/extract_policies.py` は元々別PCの絶対パスを参照して壊れており、リポジトリ相対に修正済み。他のスクリプトも再実行時は動作確認が必要
- スクリプトの多くはリポジトリルートを作業ディレクトリとして想定している

---

# NEW_SSDSE（02_industry_labor_economics の旧作業領域・参照専用）

## 概要

SSDSE の産業別労働経済分析を始めた頃の**旧作業領域**。実質的な後継は上記 [`02_industry_labor_economics/`](../../completed_projects/030_ssdse-statistics-competition/02_industry_labor_economics/README.md) で、そちらへ再配置済み。残っているのは出力物（グラフ・集計テキスト）のみで、生成元のスクリプトは残っていない。

- **フォルダ**: [`NEW_SSDSE/`](../../completed_projects/030_ssdse-statistics-competition/NEW_SSDSE/)

## 残っているもの

`NEW_SSDSE/Data/USE/アウトプット/` のみ。

- `サマリー/分析結果.txt` — 令和7年（2025年）・男女計・学歴計を対象に、**事業所規模別（10人／10〜99人／100〜999人／1,000人以上）**の産業別集計を文章化したもの。「超過実労働時間数」から始まる項目別の記述
- `データ/超過実労働時間数/`, `データ/所定内給与額/`, `データ/労働者数/` — 産業大分類（C鉱業〜Rサービス業）ごとの PNG グラフ、各16枚

## なぜ参照専用のまま残しているか

- 分析の本体は `02_industry_labor_economics/` へ移り、そちらでモデルが単一業界版→固定効果版→複数業界プーリング版へ発展している
- さらに実質的な主戦場は `03_unemployment_lca_gis/`（業種別LCA）で、02 の観測変数データがその土台になっている
- 02 の README に「スクリプト内の絶対パスは `NEW_SSDSE` フォルダ名を前提にしていたため、再配置時に更新済み」と記録がある。**このフォルダ名に依存したパスを新たに書かないこと**（→ [`ナレッジベース/03_フォルダ再配置時の絶対パス破損.md`](../../ナレッジベース/03_フォルダ再配置時の絶対パス破損.md)）
- 論文で使う可能性のあるグラフが含まれるため削除せず残した

## 最終更新（移行前の記録）

2026-08-06

---

## 最終更新

2026-10-20（`Data-Analysis_competition` リポジトリから本リポジトリへ移行。`NEW_SSDSE` を本ファイル内に統合）
