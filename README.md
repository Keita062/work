# work — 案件・ナレッジ運用リポジトリ

鈴木啓太の個人・インターン業務を、**Claude Code を前提に回すためのリポジトリ**。

案件の文脈・知見・作業実体を1か所に集約し、
「セッションが変わっても、時間が空いても、同じ前提で作業を再開できる」状態を保つことを目的とする。

- 最終更新：2026-10-20（`Data-Analysis_competition` リポジトリの案件（SSDSE・和歌山・SIGNATE）を統合）
- 運用ルールの正本：[CLAUDE.md](CLAUDE.md)（本ファイルは**全体像と手順**を示す入口）

---

## 1. 設計思想：3層に分けて置く

このリポジトリは、情報を性質ごとに3層へ分離している。**迷ったらこの表に戻る。**

| 層 | 場所 | 置くもの | 寿命 | 置かないもの |
|---|---|---|---|---|
| **文脈（Context）** | `Context Vault/` | 案件の目的・経緯・決定事項・次のアクション | 案件が続く限り更新 | 実体ファイル（スクリプト・成果物等） |
| **知見（Knowledge）** | `ナレッジベース/` | 案件をまたいで再利用できる知見・手順・ハマりどころ | 永続（案件終了後も残る） | 特定案件だけの事情 |
| **実体（Work）** | `案件フォルダ/` | スクリプト・データ・成果物・受領資料 | 案件完了で `completed_projects/` へ退避 | 進行状況の記録（Vaultへ） |

「進行状況を知りたい → Context Vault」「どうやるか知りたい → ナレッジベース」「現物を触りたい → 案件フォルダ」で固定する。

---

## 2. ディレクトリ構成

```
work/
├── CLAUDE.md                  # 運用ルールの正本（Claude が毎セッション自動で読む）
├── README.md                  # 本ファイル。全体像と手順
├── .claude/
│   ├── commands/wrapup.md     # セッション終了処理（/wrapup）
│   └── skills/new-project/    # 新規案件作成スキル（/new-project）
├── .venv/                     # Python仮想環境（gitignore）
│
├── 027_data-scientist-roadmap/  # 【実体】進行中案件の作業場（NNN_案件名）
├── 029_graduation-thesis/
├── 031_wakayama-wellbeing-competition/
│
├── Context Vault/             # 【文脈】案件の進行状況・決定事項
│   ├── Projects/              #   進行中の案件（NNN_案件名.md）
│   └── Archive/               #   完了案件
│
├── completed_projects/        # 完了案件の作業フォルダ退避先（IDは付けたまま）
│
├── ナレッジベース/              # 【知見】再利用可能な知見
│   ├── README.md               #   索引（新規追加時は必ず1行追記）
│   ├── slides/                 #   スライド作成の原則・テンプレート
│   ├── analysis/                #   データ分析・統計処理のお作法
│   └── snippets/                #   再利用するコード片・プロンプト
```

### 案件ID

- 案件フォルダと Context Vault ファイルには、**着手順の3桁の通し番号（案件ID）**を先頭に付ける
  （例：`026_persol-2030-strategy/` ↔ `Context Vault/Projects/026_persol-2030-strategy.md`）
- IDは一度付けたら変えない。`completed_projects/` へ移動しても同じIDのまま
- 最大IDがこれまでに着手した案件数になる
- 新規案件は `/new-project 案件名` で作成する（次のIDの採番・標準フォルダ・Vaultファイル・READMEを一括作成）
- `001`〜`024` は案件ID導入前（2026-10-19以前）に完了していた大学の課題・自主学習で、導入のタイミングで
  まとめてIDを付与し `completed_projects/` に格納した。中のフォルダ構成は旧来のまま（README.md＋成果物）
- `030`〜`032` は別リポジトリ `Data-Analysis_competition` で進めていたコンペ案件を2026-10-20に本リポジトリへ移行したもの。
  着手順（元リポジトリのコミット履歴）でID付与し、中のフォルダ構成は移行元のまま（README.md＋成果物）。
  `030`（SSDSE）には元リポジトリの `NEW_SSDSE/`（旧作業領域）も `NEW_SSDSE/` サブフォルダとして統合済み

| ID | 案件 | 状態 |
|---|---|---|
| 001 | cup-noodle-pos-analysis | 完了 |
| 002 | eeg-attention-analysis | 完了 |
| 003 | football-league-stats | 完了 |
| 004 | mermaid-diagram-exercises | 完了 |
| 005 | anime-collaborative-filtering | 完了 |
| 006 | csharp-exercises | 完了 |
| 007 | eye-tracking-analysis | 完了 |
| 008 | google-trends-analysis | 完了 |
| 009 | sql-exercises | 完了 |
| 010 | ssdse-economic-index | 完了 |
| 011 | statistics-exercises | 完了 |
| 012 | stock-price-analysis | 完了 |
| 013 | text-mining-demos | 完了 |
| 014 | llm-and-graph-demos | 完了 |
| 015 | machine-learning-basics | 完了 |
| 016 | paper-text-mining | 完了 |
| 017 | autonomous-bus-survey | 完了 |
| 018 | premium-teachers-survey | 完了 |
| 019 | ssdse-prefecture-lifestyle | 完了 |
| 020 | basketball-ticket-kpi | 完了 |
| 021 | employee-attrition-analysis | 完了 |
| 022 | childcare-waitlist-analysis | 完了 |
| 023 | one-off-experiments | 完了 |
| 024 | sidewalk-priority-survey | 完了 |
| 025 | onboarding-worksheet（内定者研修 宿題シート） | 完了 |
| 026 | persol-2030-strategy（パーソルキャリア内定者研修） | 完了 |
| 027 | data-scientist-roadmap（個人キャリア10ヶ年計画） | 進行中 |
| 028 | doda-hp-redesign | 完了 |
| 029 | graduation-thesis（卒論） | 着手前 |
| 030 | ssdse-statistics-competition（統計データ分析競技） | 完了 |
| 031 | wakayama-wellbeing-competition（和歌山県ウェルビーイングコンペ） | 進行中（提出済み・発表会待ち） |
| 032 | signate-anime-competition | 完了 |
| 033 | GDGoC | 着手 |

### 案件フォルダ ↔ Context Vault の対応

| 案件フォルダ | Context Vault |
|---|---|
| `NNN_案件名/` | `Context Vault/Projects/NNN_案件名.md` |
| `completed_projects/NNN_案件名/` | `Context Vault/Archive/NNN_案件名.md` |

---

## 3. 案件のライフサイクル

### ① 案件開始

`/new-project 案件名` を実行すると、以下を一括で行う。

1. 既存の最大ID+1 を採番し、リポジトリ直下に作業フォルダ `NNN_案件名/` と標準フォルダ（§4）を作る
2. `Context Vault/Projects/NNN_案件名.md` を作る（テンプレートは §5）
3. 作業フォルダに `README.md` を置き、**Vaultファイルへのリンクを必ず張る**
4. 本ファイル §2 の案件ID一覧に1行追記する

### ② 作業中

- 着手前に `ナレッジベース/README.md` の索引を確認し、該当ファイルがあれば読んでから始める
- 実体ファイルは案件フォルダへ、進行状況は Vault へ。**混ぜない**

### ③ 作業の区切り（実装完了・方針決定・問題発生）

- `Context Vault/Projects/NNN_案件名.md` に **進捗・決定事項・次のアクション・注意点** を追記する
- 技術的な問題を解決したら、ナレッジベースへの追記・新規作成を検討する（§6）

### ④ 案件完了

`git mv` で履歴を保持したまま移動する。

```bash
git mv NNN_案件名 completed_projects/NNN_案件名
git mv "Context Vault/Projects/NNN_案件名.md" "Context Vault/Archive/NNN_案件名.md"
```

IDは付けたまま移動する（振り直さない）。移動後、本ファイル §2 の案件ID一覧の状態を更新する。

---

## 4. 案件フォルダの標準構成

```
026_persol-2030-strategy/
├── README.md
├── 00_context/    # 受領した原本（先方資料・議事録原文）。編集しない
├── 01_docs/       # 自分が作った成果物・設計・要件・提出ドキュメント
├── 02_memo/       # 調査メモ・検討メモ・個人タスク
└── 03_mtg/        # MTG関連
    ├── agenda/    #   実際に使用したアジェンダ・共有資料
    └── minutes/   #   議事録
```

必要な案件だけ、`04_` 以降を追加した順に付ける（例：`04_scripts/`・`05_output/`）。

- **原則：`00_context/` は編集しない。**加工物は `01_docs/` に別ファイルで作る
- `001`〜`024`（ID導入前に完了した案件）は、中のフォルダ名を旧来のまま（README.md＋成果物フラット）にしている

---

## 5. Context Vault 案件ファイルのテンプレート

`Context Vault/Projects/NNN_案件名.md` は、以下のセクションで揃える。

```markdown
# 案件名

## 案件概要
- 案件ID：NNN
- 目的：
- ステータス：（着手前／着手／レビュー待ち／完了）
- 期限：
- 作業フォルダ：`NNN_案件名/`
- 発端：（誰から・いつ・どういう依頼か）

## 背景・経緯
- 日付つきで時系列に記録。方針変更は「【YYYY-MM-DD 方針変更】」見出しで残す

## 進め方
- ステップ／フェーズ

## 現在の進捗
- 【完了】YYYY-MM-DD：何をしたか

## 決定済み事項
- 蒸し返さないための決着ログ

## 注意点
- 次に触るときに踏み抜きやすい罠

## 未確認事項
- 誰かに聞かないと決まらないこと

## 次のアクション
1. 具体的な一手（ファイルパス付きで書く）

## 最終更新
YYYY-MM-DD（何を更新したか）
```

---

## 6. ナレッジベースへの蓄積

### いつ書くか

| トリガー | やること |
|---|---|
| 類似タスクに着手する | `ナレッジベース/README.md` の索引を確認し、該当ファイルを読んでから着手 |
| 技術的な問題を解決した／手順が変わった | 既存ファイルへ追記、なければ新規作成 |

### どう書くか

1. `ナレッジベース/` のファイル一覧を確認し、既存ファイルに入るなら追記する
2. 入らなければ新規作成し、**更新履歴セクション**を設ける
3. 新規作成したら `ナレッジベース/README.md` の索引に**1行追記する**（これを忘れると検索されず死蔵する）

索引の行は「ファイル名｜**対象タスク**」で書く。内容の要約ではなく、
**どういうタスクのときに読むべきか**を書くのが検索性の肝。

---

## 7. Claude Code の使い方

| 場面 | 言い方の例 | Claude 側の動き |
|---|---|---|
| 開始時 | 「persol-2030-strategy の続きをやりたい」 | IDを除いた名前で照合し `Context Vault/Projects/026_persol-2030-strategy.md` を読んで文脈復元 |
| 新規案件 | 「/new-project 案件名」 | 次のIDで作業フォルダ・標準フォルダ・Vaultファイル・READMEを作成 |
| 区切り | 「/wrapup」 | 進捗・決定事項・次のアクションを追記（承認後に書き込み） |
| 知見発生 | 「今のハマりどころをナレッジ化して」 | 追記先の案を提示 → 承認で書き込み＋索引更新 |
| 案件完了 | 「この案件を完了にして」 | `git mv` でフォルダ・Vaultファイルを移動し、参照パスを更新 |

前提として入っているルール（`CLAUDE.md`）：
- 日本語・敬語・結論から先
- Python は**必ず同階層の `.venv`** で実行
- 情報源の優先順位（正本宣言）：共通規約→`CLAUDE.md` ／ 案件進行→`Context Vault` ／ 知見→`ナレッジベース`

---

## 8. 環境セットアップ

```bash
source .venv/bin/activate
python script.py
deactivate
```

- `.venv` が存在しない案件フォルダでは、作業前に作成するか確認する

---

## 9. 運用ルール早見表

| する | しない |
|---|---|
| 進行状況は Context Vault に書く | 案件フォルダのREADMEに進捗を書き散らす |
| 受領物は `00_context/` に原本のまま置く | 受領ファイルを直接編集する |
| 完了案件は `git mv` で移動する | 削除する・コピーして残す |
| 知見はナレッジベース＋索引に1行 | ファイルだけ作って索引を更新しない |
| 同じ情報は1か所（正本）に置く | 同じ内容を複数ファイルにコピーする |

---

## 10. 本ドキュメントの位置づけ

| ファイル | 読み手 | 内容 |
|---|---|---|
| `CLAUDE.md` | Claude（毎セッション自動読み込み） | **守らせるルール**の正本。簡潔に保つ |
| `README.md`（本ファイル） | 人間（自分・引き継ぎ先） | **全体像と手順**。テンプレートと具体例を持つ |

ルールを変えるときは `CLAUDE.md` を先に直し、本ファイルの該当箇所を追従させる。
両者が矛盾した場合は `CLAUDE.md` が正。
