# ナレッジベース 索引

再利用可能なノウハウを蓄積する場所。特定案件の詳細ではなく、
抽象化・一般化した知見のみを置く。特定案件固有の経緯・決定事項は
`Context Vault/Projects/NNN_案件名.md` に書く。

- slides/     : スライド作成の原則・テンプレート・過去の良い例
- analysis/   : データ分析・統計処理のお作法
- ai/         : AI API（Gemini等）の技術仕様・コスト最適化の知見
- docs/       : MTGアジェンダ・Googleドキュメント／Google Driveとの連携ノウハウ
- excel/      : Excel（openpyxl等）でのファイル生成・更新の落とし穴
- snippets/   : 再利用するコード片・プロンプト

### 索引

| ファイル | 対象タスク |
|---|---|
| `slides/principles.md` | スライド作成全般（構成・レビュー観点） |
| `slides/Officeファイルをスクリプトで作る.md` | python-docx/python-pptxでdocx・pptxをスクリプト生成するとき（表レイアウト崩れ・XML順序・PDF化での目視確認） |
| `analysis/フォルダ再配置時の絶対パス破損.md` | 分析スクリプトがあるフォルダを移動・リネームするとき（絶対パス直書きの検出・対処） |
| `analysis/AsIsToBeGap分析フレームワーク.md` | 課題整理・改善提案・目標設定などAsIs/ToBe/Gapで構成するドキュメント作成全般 |
| `analysis/効果検証と要素分解の切り分け.md` | 施策の効果を測る分析設計（効果検証と要素分解の切り分け、対照群の有無の判別、限界の書き方） |
| `ai/Geminiプロンプトキャッシュ仕様メモ.md` | Gemini APIのコスト削減検討（明示的／暗黙的プロンプトキャッシュの使い分け） |
| `docs/MTGアジェンダドキュメント作成とGoogleドキュメント反映のノウハウ.md` | キックオフMTG等のアジェンダドキュメントをGoogleドキュメントとして作成・共有するタスク |
| `docs/GoogleDriveへのファイルアップロード手順.md` | ローカルで作成したファイル（pptx・xlsx・画像等）をGoogle Driveへアップロードするタスク（Driveコネクタが使えない場合のgcloud/Drive API直接呼び出し） |
| `excel/openpyxlで作成したExcelの数式・書式の落とし穴.md` | openpyxlでExcelを新規作成・更新するタスク（数式・書式が壊れる典型パターンと検出スクリプト） |

新規作成したら、この索引に「ファイル名｜対象タスク」の形式で1行追記する。
