# Google Drive へのファイルアップロード手順

> 出典：`mbk-digital/set_up`リポジトリのナレッジベースより、案件に依存しない技術知見として移植（2026-10-20）。

ローカルで作成したファイル（pptx・xlsx・画像など）を、Claude が Google Drive の指定フォルダへアップロードするときの手順と注意点。

Driveコネクタ（`mcp__claude_ai_Google_Drive__*`）が使えない・数十KB超のファイルを送る必要がある場合のフォールバック手順として読む。

## 結論

- 数十KB以上のファイルは、Drive コネクタ（`create_file`）ではアップロードしない。gcloud のトークンで Drive API を直接呼ぶ
- Drive API でのアップロードは「ファイル本体の送信」→「ファイル名と保存先フォルダの設定」の2段階で行う
- 日本語のファイル名は、Drive コネクタの `update_file` で付ける

## 手順

### 1. gcloud のトークンに Drive の権限を付ける（初回のみ）

`gcloud auth application-default login` で取得したトークンには Drive の権限がない。鈴木がターミナルで次のコマンドを実行し、ブラウザでログインする（ブラウザでの操作が必要なため、Claude は実行できない）。

```bash
gcloud auth login --enable-gdrive-access
```

トークンに Drive の権限が付いたかは、次のコマンドで確認する。`scope` に `https://www.googleapis.com/auth/drive` が含まれていればよい。

```bash
T=$(gcloud auth print-access-token)
curl -s "https://oauth2.googleapis.com/tokeninfo?access_token=$T" | grep scope
```

### 2. ファイル本体を送信する

```bash
T=$(gcloud auth print-access-token)
curl -s -X POST "https://www.googleapis.com/upload/drive/v3/files?uploadType=media&supportsAllDrives=true" \
  -H "Authorization: Bearer $T" \
  -H "Content-Type: application/vnd.openxmlformats-officedocument.presentationml.presentation" \
  --data-binary @ファイル.pptx
```

- 応答の `id` が、Drive 上のファイルIDになる
- `Content-Type` はファイルの種類に合わせる（xlsx は `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`、png は `image/png`）
- `uploadType=media` で送ると、Google 形式（Google スライドなど）には変換されず、元の形式のまま保存される

### 3. 保存先フォルダを設定する

```bash
curl -s -X PATCH "https://www.googleapis.com/drive/v3/files/<ファイルID>?addParents=<フォルダID>&removeParents=root&supportsAllDrives=true&fields=id,parents,md5Checksum,size" \
  -H "Authorization: Bearer $T"
```

- フォルダIDは、フォルダのURL（`https://drive.google.com/drive/folders/<フォルダID>`）の末尾の文字列
- 応答の `md5Checksum` とローカルファイルの `md5sum` の結果が一致すれば、ファイルは壊れずに届いている

### 4. ファイル名を付ける

Drive コネクタの `update_file` に、ファイルIDとファイル名（`title`）を渡す。

## Googleドキュメントとして作成する場合（Markdown・HTMLの文書）

Drive コネクタがない環境で、ローカルの文書を Googleドキュメントとして作成するときは、ファイル名・保存先・変換先の形式の指定（JSON）と本文（HTML）を1つのファイルにまとめ、`uploadType=multipart` で1回で送る。

- Markdown は見出し・箇条書きに変換されないため、先に HTML（`<h1>`〜`<h3>`・`<ul><li>`・`<ol><li>`・`<table>`）に書き換えておく（`06_MTGアジェンダドキュメント作成とGoogleドキュメント反映のノウハウ.md` の2-1）
- ファイル名をコマンドの引数ではなく送信するファイルの中に書くため、日本語のファイル名も文字化けしない。Drive コネクタの `update_file` で名前を付け直す必要はない

```bash
T=$(gcloud auth print-access-token)
B=bnd123456
{
  printf -- "--$B\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
  printf '{"name":"ファイル名","mimeType":"application/vnd.google-apps.document","parents":["<フォルダID>"]}'
  printf "\r\n--$B\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n"
  cat 本文.html
  printf "\r\n--$B--\r\n"
} > body.bin
curl -s -X POST "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&supportsAllDrives=true&fields=id,name,parents,webViewLink" \
  -H "Authorization: Bearer $T" \
  -H "Content-Type: multipart/related; boundary=$B" \
  --data-binary @body.bin
```

- 応答の `webViewLink` が Googleドキュメントのリンクになる
- 作成後は、次のコマンドで本文をテキストとして読み戻し、見出し・箇条書き・表が崩れていないかを確認する

```bash
curl -s "https://www.googleapis.com/drive/v3/files/<ファイルID>/export?mimeType=text/plain" -H "Authorization: Bearer $T"
```

## 注意点

| 起きたこと | 原因 | 対処 |
|---|---|---|
| Drive コネクタの `create_file` でアップロードできない | ファイルを base64 の文字列にしてツールの引数に書く必要があり、60KB の pptx でも約8万文字になる。Claude が一度に書き出せる量を超え、1文字でも誤るとファイルが壊れる | gcloud のトークンで Drive API を呼ぶ（上記の手順） |
| `curl -F` で送ると `Malformed multipart body.` になる | `curl -F` は `multipart/form-data` で送るが、Drive API は `multipart/related` しか受け付けない | 本体の送信（`uploadType=media`）と、ファイル名・保存先の設定（PATCH）を分ける |
| curl で日本語のファイル名を付けると文字化けする | Windows の Git Bash から curl に渡した日本語が、UTF-8 として送られない | ファイル名は Drive コネクタの `update_file` で付ける。Drive コネクタがない場合は、ファイル名を送信するファイルの中に書いて `uploadType=multipart` で送る |
| Drive API が `insufficientPermissions`（`ACCESS_TOKEN_SCOPE_INSUFFICIENT`）を返す | gcloud のトークンに Drive の権限がない | 鈴木が `gcloud auth login --enable-gdrive-access` を実行してログインし直す（手順1） |

## 更新履歴

- 2026-10-11: 初版作成（hinata 法人営業個別対処の部長向け報告スライド（pptx）を、Drive「#2 法人営業部向けデータ活用」へアップロードした際に、Drive コネクタでは送れないこと・`curl -F` の形式エラー・日本語ファイル名の文字化けを解決したため）
- 2026-10-16: 「Googleドキュメントとして作成する場合」の節と、注意点の2行を追記（hinata 法人営業個別対処のスライド骨子v4を、Drive コネクタがない環境で Googleドキュメントとして出力した際に、`uploadType=multipart` で日本語のファイル名ごと1回で作成できることを確認したため）
