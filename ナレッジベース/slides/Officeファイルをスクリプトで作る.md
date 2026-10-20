# Officeファイルをスクリプトで作る

## 症状

`python-docx` / `python-pptx` で配布用の docx・pptx を生成したとき、**スクリプトは正常終了するのに、開くとレイアウトが崩れている**。実際に最初の生成物で起きたのは次の3つ。

1. **表の列幅が中身に合わない。** ヘッダの `STEP` が `STE` / `P` に折り返し、`判定` 列は `説明` / `でき` / `る` と3行に割れた
2. **表がページ境界で行の途中から割れる。** 次ページの先頭にヘッダ行が無いまま、セルの2行目だけが残った
3. **罫線・背景色を付ける XML が、OOXML の決めている子要素順序を無視した位置に入る。** Word は寛容なので今回は開けたが、仕様上は不正で、いつ表示が壊れてもおかしくない

いずれも**例外が出ないので、実行結果だけ見ていると気づけない**。

## 原因

- `python-docx` / `python-pptx` は低レベルAPIで、罫線・セル背景・行の分割制御に高レベルのメソッドが無い。XML要素を自分で作って足すことになるが、素朴に `append()` すると**スキーマが要求する順序を守れない**。`w:tblPr` なら `tblBorders` は `tblLayout` より前、`w:tcPr` なら `shd` は `tcW` より後、と決まっている
- 列幅を `len()` で按分すると、**全角も半角も1として数えてしまう**。日本語の表では実際の描画幅と2倍ずれる。さらに、長い列と短い列の文字数比をそのまま幅比にすると、短い列が潰れる
- 表の行はデフォルトでページ境界で分割される。ヘッダ行の繰り返しも既定では無効

## 対処

### 1. 子要素は順序を見て差し込む

```python
TBLPR_ORDER = [
    "tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
    "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd",
    "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
]
TCPR_ORDER = [
    "cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
    "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark",
]

def _insert_ordered(parent, el, order):
    idx = order.index(el.tag.split("}")[1])
    for child in parent:
        cname = child.tag.split("}")[1]
        if cname in order and order.index(cname) > idx:
            child.addprevious(el)   # lxml。後ろの要素の前に割り込ませる
            return
    parent.append(el)
```

書いたあとに**順序を機械チェックする**。目視では絶対に見つからない。

```python
idx = [order.index(c.tag.split("}")[1]) for c in pr
       if c.tag.split("}")[1] in order]
assert idx == sorted(idx)
```

### 2. 列幅は「表示幅」で按分し、指数で圧縮する

```python
def _dw(text):
    """全角を2、半角を1として数えた表示幅。改行があれば最長行で測る。"""
    return max(sum(2 if ord(c) > 0x2000 else 1 for c in seg)
               for seg in (text.replace("<br>", "\n").split("\n") or [""]))

weights = []
for ci in range(ncol):
    content = max([_dw(r[ci]) for r in rows[1:]] or [0])
    head = _dw(rows[0][ci])
    # 長い列が全部を飲み込まないよう content は強めに圧縮し、
    # 見出しは折り返させたくないので圧縮を弱くかける
    weights.append(max(content ** 0.62, head ** 0.85, 2.0))
```

`** 0.62` の圧縮が肝。生の比率で割ると、長文の列が幅を独占して短い列が1〜2文字幅になる。

### 3. 行を割らせない・ヘッダを繰り返す

```python
tr_pr = row._tr.get_or_add_trPr()
tr_pr.append(OxmlElement("w:cantSplit"))
if is_header:
    tr_pr.append(OxmlElement("w:tblHeader"))
```

### 4. 生成物は必ず PDF 化して目で見る

**これが一番効いた。** Word / PowerPoint を COM で叩いて PDF に変換すれば、`Read` ツールでページを画像として読める。上の崩れ3件は全部この方法で見つけた。

```powershell
$word = New-Object -ComObject Word.Application
$word.Visible = $false; $word.DisplayAlerts = 0
$doc = $word.Documents.Open($src, $false, $true)   # ReadOnly
"pages: $($doc.ComputeStatistics(2))"
$doc.SaveAs([ref]$out, [ref]17)                     # 17 = wdFormatPDF
$doc.Close($false); $word.Quit()
```

```powershell
$pp = New-Object -ComObject PowerPoint.Application
$pres = $pp.Presentations.Open($src, $true, $false, $false)
$pres.SaveAs($out, 32)                              # 32 = ppSaveAsPDF
$pres.Close(); $pp.Quit()
```

**Word で開くと `~$ファイル名.docx`（オーナーファイル）が残る。** 変換後に消さないとリポジトリに紛れ込む。

```powershell
Get-ChildItem "<dir>\~`$*" -Force -ErrorAction SilentlyContinue | Remove-Item -Force
```

PowerPoint 側は `SaveAs` の第2引数が PDF の場合、拡張子ではなく**フォーマット番号で決まる**ので、番号を間違えると pptx が上書き保存される。`32` を必ず指定する。

### 補足：日本語フォントの指定

`run.font.name` だけでは日本語グリフに当たらない（Latin にしか効かない）。`w:rFonts` の `eastAsia` も明示する。

```python
rpr = run._r.get_or_add_rPr()
rfonts = rpr.find(qn("w:rFonts"))
if rfonts is None:
    rfonts = OxmlElement("w:rFonts"); rpr.insert(0, rfonts)
for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
    rfonts.set(qn(attr), "Yu Gothic")
```

pptx 側は `a:latin` / `a:ea` / `a:cs` の3つに `typeface` を設定する。

等幅が要る箇所（罫線で描いたフロー図など）は **`MS Gothic`** を使う。全角も等幅になるので、日本語混じりのアスキーアートが崩れない。

## 再発防止

- **Officeファイルを生成したら、必ず PDF 化して全ページに目を通してからコミットする。** スクリプトが正常終了したことは、レイアウトが正しいことを何も保証しない
- **md を単一の正本にして、docx・pptx はそこからの再生成物として扱う。** バイナリを手編集すると次の再生成で消える。スクリプトの docstring に「手で編集せず、md を直して再実行する」と書いておく（`make_gantt.py` から続く運用）
- **XMLを自前で足したら、順序チェックのアサーションも一緒に書く。** Word が寛容なせいで、不正なまま気づかず運用してしまう
- 日本語を含む表を作るときは、**`len()` ではなく全角=2 の表示幅で計算する**
- COM を使ったあとは `~$` オーナーファイルの後始末をする。`git status` に見慣れないファイルが出たら疑う

## 出典

- [`Wakayama/scripts/make_review_docx.py`](../../031_wakayama-wellbeing-competition/scripts/make_review_docx.py)（md → A4縦8ページの送付用 Word）
- [`Wakayama/scripts/make_review_slides.py`](../../031_wakayama-wellbeing-competition/scripts/make_review_slides.py)（16:9・20枚のスライド）
- 和歌山コンペのパーソルキャリア様向けレビュー資料（09-04版）を作った作業
- `python-docx` はこの時点で未導入だったので `python -m pip install python-docx` から入れている
- 記録日: 2026-09-03
