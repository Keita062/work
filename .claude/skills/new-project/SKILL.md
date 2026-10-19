---
name: new-project
description: 新規案件を立ち上げる。次の案件ID（3桁の通し番号）を採番し、作業フォルダ・標準サブフォルダ・Context Vaultファイル・READMEを一括作成する。「新しい案件を作りたい」「案件フォルダを作って」「/new-project 案件名」のときに使う。
argument-hint: 案件名（例：bach）
---

# 新規案件の作成

引数：`$ARGUMENTS`（案件名）。空なら案件名をユーザーに確認してから進める。

## 1. 案件IDの採番

以下4か所に存在する `NNN_` で始まる名前から最大値を取り、+1 したものを3桁ゼロ埋めで新IDとする。

- リポジトリ直下のフォルダ
- `completed_projects/` 直下のフォルダ
- `Context Vault/Projects/` 直下
- `Context Vault/Archive/` 直下

```bash
ls -1 . completed_projects "Context Vault/Projects" "Context Vault/Archive" | grep -oE '^[0-9]{3}_' | sort -u | tail -1
```

同名（IDを除いた名前）の案件が既にある場合は作成せず、ユーザーに確認する。

## 2. 作業フォルダの作成

```
NNN_案件名/
├── README.md
├── 00_context/.gitkeep   # 受領した原本。編集しない
├── 01_docs/.gitkeep      # 自分が作った成果物・設計・要件・提出ドキュメント
├── 02_memo/.gitkeep      # 調査メモ・検討メモ・個人タスク
└── 03_mtg/
    ├── agenda/.gitkeep   # 実際に使用したアジェンダ・共有資料
    └── minutes/.gitkeep  # 議事録
```

- フォルダにファイルを置いたら、そのフォルダの `.gitkeep` は削除してよい
- 標準以外のフォルダ（scripts・raw・work 等）は必要になった時点で `04_` 以降を追加順に付ける

`README.md` の雛形：

```markdown
# 案件名

（案件の一言説明）

- 案件ID：NNN
- ステータス：着手前
- 案件の詳細・経緯・次のアクションは [Context Vault/Projects/NNN_案件名.md](../Context%20Vault/Projects/NNN_案件名.md) を参照

## フォルダ構成

| フォルダ | 内容 |
|---|---|
| `00_context/` | 受領した原本。編集しない |
| `01_docs/` | 自分が作った成果物・設計・提出ドキュメント |
| `02_memo/` | 調査メモ・検討メモ・個人タスク |
| `03_mtg/agenda/` | MTGで使用したアジェンダ・共有資料 |
| `03_mtg/minutes/` | 議事録 |
```

## 3. Context Vault ファイルの作成

`Context Vault/Projects/NNN_案件名.md` を、リポジトリ直下 `README.md` §5 のテンプレートで作る。
「案件ID」「作業フォルダ：`NNN_案件名/`」「発端」は会話から分かる範囲で埋め、
分からない項目は空欄のままにする（推測で埋めない）。

## 4. 索引への追記

リポジトリ直下 `README.md` §2 の「案件ID」一覧の表の末尾に1行追記する。

```markdown
| NNN | 案件名 | 進行中 |
```

## 5. 完了報告

作成したパスの一覧と採番したIDを報告する。コミットはユーザーの指示があるまで行わない。
