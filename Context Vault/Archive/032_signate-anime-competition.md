# SIGNATE（アニメ系テキストコンペ）

## 案件概要
- 案件ID：032
- 目的：SIGNATE主催のデータ分析コンペティションへの参加。現在参加していたのは `signate_anime`
- 課題（signate_anime）：LLM が生成した「架空作品のあらすじ」から、**元ネタとなった作品を予測する**多クラス分類。評価用データは架空作品340件で、各件について元ネタ作品のid（a・b の2つ）を予測して提出する
- ステータス：完了（アーカイブ。停止中のまま移行）
- 作業フォルダ：`completed_projects/032_signate-anime-competition/`
- 発端：個人で参加。締切等の期限情報は元リポジトリ内に記録なし
- **移行メモ（2026-10-20）**：旧 `Data-Analysis_competition` リポジトリから本リポジトリへ移行

## 背景・経緯

- [`signate_anime/`](../../completed_projects/032_signate-anime-competition/signate_anime/README.md) 配下で作業

### データ

[`signate_anime/Data/`](../../completed_projects/032_signate-anime-competition/signate_anime/Data/)

| ファイル | 内容 |
|---|---|
| `base_stories.tsv` | 元ネタ作品名とあらすじ（`id`, `category`, `title`, `story`）。ラベルの定義元 |
| `fiction_stories_practice.tsv` | 練習用（`id_a`, `id_b`, `title_a`, `title_b`, `story`）。提出不要 |
| `fiction_stories_test.tsv` | 評価用 架空作品340件（`id`, `story`） |
| `sample_submit.csv` | 提出フォーマットのサンプル |

## 進捗（アーカイブ時点）

提出ごとに分析イテレーションのフォルダを分け、回を追うごとに手法を発展させていた。

| フォルダ | 中身 |
|---|---|
| `tutorial/` | チュートリアル用ノートブック |
| `Analysis_file_1st/` | `Analysis.ipynb`, `Analysis_second.ipynb` |
| `Analysis_file_2nd/` | `2nd_README.md`（課題・データ仕様のメモのみ） |
| `Analysis_file_3rd/` | `3rd.ipynb`, `charenge.ipynb` ＋ `out_put/`（`submission.csv`、正答率ランキング統計、残差プロット群） |
| `Analysis_file_4th/` | `analysis.ipynb`（最新） |

提出用CSVの出力先は `signate_anime/output/`（`output.csv`, `output_1.csv`）。

## 決定済み事項

- 旧フォルダ名のタイポ `sigante_anime` は `signate_anime` に修正済み
- SIGNATE の別コンペに参加する場合は `SIGNATE/`（現 `032_signate-anime-competition/`）直下に新しいフォルダを追加する方針だった（1コンペ=1フォルダ）

## 注意点

- **各イテレーションで何を試して、スコアがどう動いたのかの記録が残っていない。** README はどれも課題・データ仕様の写しで、手法や結果の記述がない。再開するならまず `Analysis_file_4th/analysis.ipynb` を読んで現状の手法を把握する必要がある
- コンペの締切・リーダーボード順位は不明なまま終了

## 次のアクション

- 特になし（アーカイブ済み）。再開する場合は、まず 4th の手法とスコアを確認し、このファイルに記録してから次の施策を決める

## 最終更新

2026-10-20（`Data-Analysis_competition` リポジトリから本リポジトリへ移行。元の最終更新は 2026-08-06）
