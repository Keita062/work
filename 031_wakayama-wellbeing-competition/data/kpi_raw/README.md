# 個別KPIの一次資料（kpi_raw）

デジタル庁 Well-Being ガイドブックが各個別KPIの出典として挙げている一次資料（一覧は [`../guidebook_sources_municipalities.xlsx`](../guidebook_sources_municipalities.xlsx)）のうち、M1上位5分野（健康状態・自己効力感・住宅環境・地域とのつながり・公共空間）の分を取得したもの。取得日は 2026-09-22。

整形は [`../../scripts/build_kpi_panel.py`](../../scripts/build_kpi_panel.py)、分析は [`../../scripts/m4_kpi_analysis.py`](../../scripts/m4_kpi_analysis.py)。出力は `../processed/`。

## ファイルと取得元

| ファイル | 内容 | 取得元 |
|---|---|---|
| census2020_b01_01_pop_households.xlsx | 国勢調査2020 第1-1表 人口・世帯数・面積（市区町村マスタ） | e-Stat statInfId=000032142402 |
| census2020_b10_household_type.xlsx | 国勢調査2020 第10表 世帯の家族類型 | e-Stat 000032142501 |
| census2020_4-3_marital.xlsx | 国勢調査2020 第4-3表 配偶関係（市区町村） | e-Stat 000032142476 |
| census2020_c01_02_labour.xlsx | 国勢調査2020 就業状態等基本集計 第1-2表 労働力状態 | e-Stat 000032201175 |
| jyuki_jinko2024_age_sex.xlsx | 住民基本台帳 年齢階級別人口（2024-01-01） | e-Stat 000040306674 |
| jyuki_idou2024_11-3_net_migration.xlsx | 住民基本台帳人口移動報告 2024 表11-3 転入超過数 | e-Stat 000040271508 |
| jyutaku2023_e010_4_floor_area.xlsx | 住宅・土地統計調査2023 延べ面積 | e-Stat 000040209867 |
| jyutaku2023_e008_4_tenure.xlsx | 同 所有の関係×建て方 | e-Stat 000040209861 |
| jyutaku2023_e122_4_rent.xlsx | 同 延べ面積1m²当たり家賃 | e-Stat 000040210062 |
| keizaicensus2021_b1_009_1a.xlsx | 経済センサス‐活動調査2021 産業小分類別事業所数 | e-Stat 000040067884 |
| mhlw_seimeihyo_shikuchoson_2020.xlsx | 令和2年市区町村別生命表 | e-Stat 000040052725 |
| mhlw_kaigo_2406-h2-2.xlsx / -h2-3.xlsx | 介護保険事業状況報告 2024年6月 保険者別 要介護認定者数（男・女） | https://www.mhlw.go.jp/topics/kaigo/osirase/jigyo/m24/2406.html |
| mhlw_jisatsu_r5_A7_municipal_residence.xls | 地域における自殺の基礎資料 2023 A7表（市町村・自殺日・住居地） | https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/hukushi_kaigo/seikatsuhogo/jisatsu/jisatsu_kiso_r5.html |
| npo_gyousei_000.zip | NPO法人ポータル 行政入力情報（全国） | https://www.npo-homepage.go.jp/npoportal/download/all |
| soumu_koukyoushisetsu_shichoson_keinen_H18-R5.xlsx | 公共施設状況調 市町村経年比較表 | https://www.soumu.go.jp/iken/shisetsu/index.html |
| soumu_kessan_r05_gaikyo_a.xlsx / _b.xlsx | 令和5年度 市町村別決算状況調 概況（都市・町村）＝財政力指数 | https://www.soumu.go.jp/iken/zaisei/r05_shichouson.html |
| mlit_toshi_monitoring_zentai.xlsx | 都市モニタリングシート 全体表（R6） | https://www.mlit.go.jp/toshi/tosiko/toshi_tosiko_tk_000036.html |
| mlit_walkable_cities_20260831.pdf / .txt | ウォーカブル推進都市一覧（.txt は pdftotext で抽出） | https://www.mlit.go.jp/toshi/walkable/walkablecity/ |

## 取得しなかったKPI

| 分野 | KPI | 理由 |
|---|---|---|
| 健康状態 | 人口あたり精神疾患患者在院数 | ReMHRAD は Web 閲覧のみで一括取得できない |
| 自己効力感 | 首長選挙投票率・市区町村議会投票率 | 出典が民間サイト（選挙ドットコム）。**公式の客観スコアの大半を占めるとみられ、次に取るべきもの** |
| 住宅環境 | 平均価格（住宅地） | 都道府県地価調査は地点別の公表で、市区町村平均の表が無い |
| 地域とのつながり | 居住期間20年以上の人口割合 | ガイドブックのリンク先（statInfId=000031611770）は全国値のみの表 |
| 地域とのつながり | 祭りの数・関係人口創出活動指数 | 出典が民間サイト（オマツリジャパン・SMOUT） |
| 地域とのつながり | 自治会・町内会加入率 | 各自治体サイトに散在 |
| 地域とのつながり | 都市再生推進法人・UDC数 | 該当が一部の都市のみ |
| 公共空間 | 滞在快適性等向上区域・地方版自転車活用推進計画 | ウォーカブル推進都市のPDFから区別できない／一覧未取得 |

## 作り方の注意

- **健康寿命は近似**。保険者別の年齢別認定率が無いため、`平均寿命 − T65/l0 × 要介護2以上の割合（65歳以上一律）` で出した。広域連合が保険者の町村は欠損
- 自殺者数：A7表に載らない市町村は自殺者0人として扱った
- NPO：住所に都道府県名が無いため所轄庁から補った。解散・認証取消の法人は除外（48,690法人を割り当て）
- 住宅・土地統計調査は人口1.5万人以上の市区町村のみ
- 公園緑地徒歩圏人口カバー率の公園データは2011年時点（人口メッシュのみ更新）
- ウォーカブル推進都市は、市区町村名が全国で一意のものだけ判定し、同名がある市区町村は欠損
