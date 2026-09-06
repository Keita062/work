# -*- coding: utf-8 -*-
"""
内定者研修 チームA 中間フィードバック依頼 スライド生成

05_中間FB_スライド構成.md の構成に従って .pptx を生成する。
構成ドキュメントを修正 → このスクリプトを修正 → 再実行、の順で更新する。

    .venv\\Scripts\\python.exe projects\\persol-2030-strategy\\build_slides.py
"""
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).parent / "output" / "内定者研修チームA_中間FB_20260907.pptx"

# ---------------------------------------------------------------- 見た目の定義

FONT = "Yu Gothic"

INK = RGBColor(0x14, 0x1B, 0x2D)  # 見出し・本文
BODY = RGBColor(0x33, 0x3A, 0x4B)  # 本文
MUTED = RGBColor(0x77, 0x7F, 0x92)  # 補足・脚注
ACCENT = RGBColor(0x0B, 0x4F, 0x9E)  # メインアクセント（青）
ACCENT_L = RGBColor(0xE3, 0xEC, 0xF7)  # 青の薄いフィル
WARN = RGBColor(0xD1, 0x4D, 0x1F)  # GAP・強調（オレンジ）
WARN_L = RGBColor(0xFB, 0xEC, 0xE5)
LINE = RGBColor(0xD8, 0xDD, 0xE6)
PANEL = RGBColor(0xF4, 0xF6, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SW, SH = Inches(13.333), Inches(7.5)
ML = Inches(0.72)  # 左マージン
CW = Inches(11.89)  # コンテンツ幅
TOP = Inches(1.42)  # 本文開始
BOT = Inches(6.82)  # フッター位置

prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]

_page = {"n": 0}


# ------------------------------------------------------------------ ヘルパー


def _set_font(run, size, bold=False, color=BODY, font=FONT):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    f.name = font
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", font)


def tb(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, first=False, space_before=0, space_after=4, align=PP_ALIGN.LEFT,
         line=1.28, level=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    p.alignment = align
    p.line_spacing = line
    p.level = level
    return p


def text(tf, chunks, first=False, **kw):
    """chunks: str | (str, dict) のリスト。dict は _set_font の引数。"""
    p = para(tf, first=first, **kw)
    if isinstance(chunks, str):
        chunks = [(chunks, {})]
    for c in chunks:
        s, opt = (c, {}) if isinstance(c, str) else c
        r = p.add_run()
        r.text = s
        _set_font(r, opt.pop("size", 14), **opt)
    return p


def rect(slide, x, y, w, h, fill=None, line_color=None, line_w=0.75,
         shape=MSO_SHAPE.RECTANGLE, dash=False):
    sh = slide.shapes.add_shape(shape, x, y, w, h)
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line_color is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line_color
        sh.line.width = Pt(line_w)
        if dash:
            ln = sh.line._get_or_add_ln()
            d = etree.SubElement(ln, qn("a:prstDash"))
            d.set("val", "dash")
    sh.text_frame.word_wrap = True
    sh.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    sh.text_frame.margin_left = sh.text_frame.margin_right = Inches(0.16)
    sh.text_frame.margin_top = sh.text_frame.margin_bottom = Inches(0.11)
    return sh


def slide_new(title, kicker=None, note=None):
    """見出し付きの本文スライドを作る。"""
    s = prs.slides.add_slide(BLANK)
    _page["n"] += 1
    y = Inches(0.46)
    if kicker:
        t = tb(s, ML, y, CW, Inches(0.26))
        text(t, [(kicker, dict(size=11.5, bold=True, color=ACCENT))], first=True,
             space_after=0)
        y = Inches(0.74)
    t = tb(s, ML, y, CW, Inches(0.5))
    text(t, [(title, dict(size=23, bold=True, color=INK))], first=True,
         space_after=0, line=1.15)
    rect(s, ML, Inches(1.24), Inches(0.62), Inches(0.045), fill=ACCENT)
    if note:
        n = tb(s, ML, BOT, CW, Inches(0.3))
        text(n, [(note, dict(size=9.5, color=MUTED))], first=True, space_after=0,
             line=1.2)
    _footer(s)
    return s


def _footer(s):
    f = tb(s, Inches(12.2), BOT, Inches(0.55), Inches(0.3))
    text(f, [(str(_page["n"]), dict(size=10, color=MUTED))], first=True,
         space_after=0, align=PP_ALIGN.RIGHT)


def bullets(tf, items, size=13.5, gap=6, first_para=True):
    """items: (level, chunks) のリスト。level 0=中黒, 1=ダッシュ, 2=平文"""
    marks = {0: "● ", 1: "－ ", 2: "  "}
    for i, (lv, chunks) in enumerate(items):
        if isinstance(chunks, str):
            chunks = [(chunks, {})]
        chunks = [(c, {}) if isinstance(c, str) else c for c in chunks]
        p = para(tf, first=(i == 0 and first_para), space_after=gap, line=1.3)
        p.level = 0
        r = p.add_run()
        r.text = ("    " * lv) + marks[lv]
        _set_font(r, size, color=(ACCENT if lv == 0 else MUTED))
        for s_, opt in chunks:
            opt = dict(opt)
            rr = p.add_run()
            rr.text = s_
            _set_font(rr, opt.pop("size", size), **opt)


def card(slide, x, y, w, h, title, lines, accent=ACCENT, fill=WHITE,
         tsize=14.5, bsize=11.8, dash=False, badge=None):
    sh = rect(slide, x, y, w, h, fill=fill, line_color=LINE, dash=dash)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    rect(slide, x, y, Inches(0.05), h, fill=accent)
    tf = sh.text_frame
    tf.margin_left = Inches(0.22)
    if badge:
        text(tf, [(badge, dict(size=9.5, bold=True, color=accent))], first=True,
             space_after=3, line=1.1)
        text(tf, [(title, dict(size=tsize, bold=True, color=INK))], space_after=6,
             line=1.2)
    else:
        text(tf, [(title, dict(size=tsize, bold=True, color=INK))], first=True,
             space_after=6, line=1.2)
    for ln in lines:
        if isinstance(ln, str):
            ln = [(ln, {})]
        ln = [(c, {}) if isinstance(c, str) else c for c in ln]
        p = para(tf, space_after=3.5, line=1.28)
        for s_, opt in ln:
            opt = dict(opt)
            r = p.add_run()
            r.text = s_
            _set_font(r, opt.pop("size", bsize), **opt)
    return sh


def label(slide, x, y, w, s, size=12, bold=True, color=INK,
          align=PP_ALIGN.LEFT, h=Inches(0.3)):
    tf = tb(slide, x, y, w, h)
    text(tf, [(s, dict(size=size, bold=bold, color=color))], first=True,
         space_after=0, align=align, line=1.15)
    return tf


# --- 本日いただきたいFB（スライド3・14で同一のものを再掲する） -----------------

FB_ITEMS = [
    (ACCENT, "1", "「選ばれる」の定量定義は、これでよいか"),
    (WARN, "2", "5つの課題のうち、どれを本命に絞るべきか"),
]


def fb_cards(slide):
    """本日いただきたいFB。縦に2項目の箇条書き（スライド3・12で共通）。"""
    H, GAP = Inches(1.38), Inches(0.42)
    for i, (ac, no, q) in enumerate(FB_ITEMS):
        y = TOP + Inches(0.5) + i * (H + GAP)
        rect(slide, ML, y, CW, H, fill=PANEL)
        rect(slide, ML, y, Inches(0.06), H, fill=ac)

        rect(slide, ML + Inches(0.46), y + Inches(0.38), Inches(0.62),
             Inches(0.62), fill=ac, shape=MSO_SHAPE.OVAL)
        label(slide, ML + Inches(0.46), y + Inches(0.54), Inches(0.62), no,
              size=17, color=WHITE, align=PP_ALIGN.CENTER)

        label(slide, ML + Inches(1.42), y + Inches(0.45), CW - Inches(1.9), q,
              size=24, color=INK, h=Inches(0.52))

    label(slide, ML, TOP + Inches(4.34), CW,
          "④解決策・⑤1か月の行動は、次回の中間FBでご報告します。", size=12,
          bold=False, color=MUTED)


# ================================================================ スライド 1
s = prs.slides.add_slide(BLANK)
_page["n"] += 1
rect(s, Inches(0), Inches(0), SW, Inches(0.1), fill=ACCENT)

CX, CXW = Inches(1.2), Inches(10.93)  # 中央寄せの領域

label(s, CX, Inches(2.32), CXW, "内定者研修 課題", size=12, color=MUTED,
      align=PP_ALIGN.CENTER)

tf = tb(s, CX, Inches(2.78), CXW, Inches(0.85))
text(tf, [("内定者研修 チームA 中間フィードバック依頼",
           dict(size=36, bold=True, color=INK))], first=True, space_after=0,
     align=PP_ALIGN.CENTER, line=1.15)

label(s, CX, Inches(3.72), CXW, "中間報告（1 / 2）── ここまでの発表", size=16,
      bold=False, color=BODY, align=PP_ALIGN.CENTER)

rect(s, Inches(6.29), Inches(4.42), Inches(0.75), Inches(0.035), fill=ACCENT)

label(s, CX, Inches(4.78), CXW, "チームA", size=13, color=ACCENT,
      align=PP_ALIGN.CENTER)
label(s, CX, Inches(5.13), CXW,
      "にし ゆうま　／　かな　／　みお　／　まい　／　けーた", size=13.5,
      bold=False, color=INK, align=PP_ALIGN.CENTER)
label(s, CX, Inches(5.62), CXW, "2026年9月7日", size=11.5, bold=False,
      color=MUTED, align=PP_ALIGN.CENTER)

# ================================================================ スライド 2
s = slide_new("本日お話しすること", kicker="AGENDA")

agenda = [
    ("01", "前提", "「選ばれる」をどう定義したか（あるべき姿／現状）"),
    ("02", "リサーチ内容", "定義に基づいて集めたファクト"),
    ("03", "課題", "いま我々が課題と捉えているもの（発散段階）"),
    ("04", "いただきたいフィードバック", "本日ご指摘いただきたい2点"),
]
for i, (no, ttl, sub) in enumerate(agenda):
    yy = TOP + Inches(i * 0.75)
    rect(s, ML, yy, Inches(0.62), Inches(0.62), fill=ACCENT_L)
    label(s, ML, yy + Inches(0.16), Inches(0.62), no, size=14, color=ACCENT,
          align=PP_ALIGN.CENTER)
    label(s, ML + Inches(0.85), yy + Inches(0.05), Inches(4), ttl, size=16)
    label(s, ML + Inches(0.85), yy + Inches(0.33), Inches(8), sub, size=12,
          bold=False, color=MUTED)

# 研修課題5要素のうち、どこまでを本日扱うか（ステップライン）
rect(s, ML, Inches(4.62), CW, Inches(0.015), fill=LINE)
label(s, ML, Inches(4.86), Inches(8), "研修課題の5要素のうち、本日は ①〜③ まで",
      size=12, color=INK)

steps = [("① 前提", True), ("② リサーチ", True), ("③ 課題", True),
         ("④ 解決策", False), ("⑤ 1か月の行動", False)]
slot = Emu(int(CW / 5))
cx = [ML + Emu(int(slot * (i + 0.5))) for i in range(5)]
LY = Inches(5.62)          # ライン中心
DOT = Inches(0.24)

rect(s, cx[0], LY - Inches(0.008), cx[4] - cx[0], Inches(0.016), fill=LINE)
rect(s, cx[0], LY - Inches(0.012), cx[2] - cx[0], Inches(0.024), fill=ACCENT)

for i, (name, on) in enumerate(steps):
    rect(s, cx[i] - DOT / 2, LY - DOT / 2, DOT, DOT,
         fill=(ACCENT if on else WHITE),
         line_color=(None if on else LINE), line_w=1.25,
         shape=MSO_SHAPE.OVAL)
    label(s, cx[i] - slot / 2, LY + Inches(0.24), slot, name, size=12.5,
          color=(ACCENT if on else MUTED), align=PP_ALIGN.CENTER)

label(s, cx[0] - slot / 2, LY + Inches(0.66), cx[2] - cx[0] + slot,
      "本日はここまで", size=11.5, color=ACCENT, align=PP_ALIGN.CENTER)
label(s, cx[3] - slot / 2, LY + Inches(0.66), cx[4] - cx[3] + slot,
      "次回の中間FBでご報告します", size=11.5, bold=False, color=MUTED,
      align=PP_ALIGN.CENTER)

# ================================================================ スライド 3
s = slide_new("本日いただきたいフィードバック", kicker="本日のゴール",
              note="定義の詳細はスライド4、課題の詳細はスライド8〜10でご説明します。")

fb_cards(s)

# ================================================================ スライド 4
s = slide_new("「選ばれる」の定義 ── 法人／個人それぞれで置いた",
              kicker="① 前提 ─ 定義",
              note="出典：9/4 チーム共有ドキュメント「①前提（「選ばれる」の定義）」"
                   "および 9/4 対面MTGのホワイトボード。")

# 上段：スコープを1本の帯に
sh = rect(s, ML, TOP, CW, Inches(0.6), fill=PANEL)
sh.text_frame.margin_top = Inches(0.17)
text(sh.text_frame,
     [("データが取れている範囲 ", dict(size=11, bold=True, color=ACCENT)),
      ("リクルートエージェント ／ パーソル（doda）", dict(size=13, bold=True, color=INK)),
      ("　　｜　　", dict(size=13, color=LINE)),
      ("時間軸 ", dict(size=11, bold=True, color=ACCENT)),
      ("2030年度", dict(size=13, bold=True, color=INK))],
     first=True, space_after=0, align=PP_ALIGN.CENTER)

# 中段：法人／個人
LX, LW = ML, Inches(5.87)
RX, RW = ML + Inches(6.02), Inches(5.87)
views = [
    (LX, "法人視点",
     ["doda等人材サービスにおける就職率・定着率、",
      "日本企業全体におけるパーソルサービスのシェア率"],
     ["単なる採用にとどまらず、育成・人員配置・組織開発まで",
      "一括して任される状態、および長期的・継続的な契約関係",
      "（LTV・NPS向上）"]),
    (RX, "個人視点",
     ["登録者数、転職成功率、キャリア選択の幅、",
      "プラットフォーム全体の生涯価値（LTV）"],
     ["キャリアのあらゆるフェーズにおいて選ばれ、",
      "中長期的に利用され続ける状態"]),
]
for x, head, quant, qual in views:
    sh = rect(s, x, Inches(2.3), LW, Inches(2.9), fill=WHITE, line_color=LINE)
    rect(s, x, Inches(2.3), LW, Inches(0.05), fill=ACCENT)
    sh.text_frame.margin_left = Inches(0.26)
    sh.text_frame.margin_top = Inches(0.24)
    text(sh.text_frame, [(head, dict(size=18, bold=True, color=INK))],
         first=True, space_after=13)
    for ttl, lines in (("定量指標", quant), ("定性・状態", qual)):
        text(sh.text_frame,
             [(ttl, dict(size=10.5, bold=True, color=ACCENT))],
             space_before=(0 if ttl == "定量指標" else 12), space_after=5)
        for j, ln in enumerate(lines):
            text(sh.text_frame, [(ln, dict(size=12.2, color=BODY))],
                 space_after=(0 if j < len(lines) - 1 else 0), line=1.2)

# 下段：9/4に収束させた結論
rect(s, ML, Inches(5.42), CW, Inches(0.7), fill=ACCENT_L)
label(s, ML, Inches(5.63), CW,
      "9/4に収束させた結論　「選ばれる」＝ 決定率が高い状態（リクルートに数値で上回る状態）",
      size=14.5, color=ACCENT, align=PP_ALIGN.CENTER)

# ================================================================ スライド 5
s = slide_new("あるべき姿（2030）と現状 ── GAPは 個人26.4万人／法人2.7万人",
              kicker="① 前提 ─ To Be / As Is",
              note="出典は Appendix 1。年度がそろっていない点（個人＝FY23、法人＝令和7年度）に"
                   "ご留意ください。")

hdr = ["", "あるべき姿 To Be 2030", "現状 As Is（パーソル）", "GAP"]
colx = [ML, ML + Inches(3.5), ML + Inches(6.55), ML + Inches(9.5)]
colw = [Inches(3.5), Inches(3.05), Inches(2.95), Inches(2.39)]
for i, h in enumerate(hdr):
    label(s, colx[i], TOP, colw[i], h, size=11.5,
          color=(ACCENT if i == 1 else MUTED))
rect(s, ML, TOP + Inches(0.3), CW, Inches(0.02), fill=INK)

data = [
    ("個人KPI：年間登録者数", "202万4,000人", "176万人", "（FY23）", "26万4,000人"),
    ("法人KPI：就職者数", "87,754人", "60,307人", "（令和7年度）", "27,447人"),
]
for i, (k, b, a, note_, g) in enumerate(data):
    yy = TOP + Inches(0.46 + i * 0.92)
    label(s, colx[0], yy + Inches(0.12), colw[0], k, size=13.5)
    label(s, colx[1], yy + Inches(0.05), colw[1], b, size=19, color=INK)
    tfa = tb(s, colx[2], yy + Inches(0.05), colw[2], Inches(0.5))
    text(tfa, [(a, dict(size=19, bold=True, color=INK)),
               (" " + note_, dict(size=10.5, bold=False, color=MUTED))],
         first=True, space_after=0)
    label(s, colx[3], yy + Inches(0.05), colw[3], g, size=19, color=WARN)
    rect(s, ML, yy + Inches(0.74), CW, Inches(0.012), fill=LINE)

# 横棒バー
bar_y = Inches(3.62)
label(s, ML, bar_y, Inches(11.8), "あるべき姿に対する現在地", size=12)
BX, BW = ML, Inches(9.4)
for i, (name, cur, tot) in enumerate([("個人KPI 年間登録者数", 176.0, 202.4),
                                      ("法人KPI 就職者数", 60307, 87754)]):
    yy = bar_y + Inches(0.42 + i * 1.02)
    label(s, BX, yy - Inches(0.02), Inches(3), name, size=11, bold=False,
          color=MUTED)
    w_cur = Emu(int(BW * (cur / tot)))
    rect(s, BX, yy + Inches(0.24), BW, Inches(0.42), fill=WARN_L,
         line_color=WARN, line_w=0.75, dash=True)
    rect(s, BX, yy + Inches(0.24), w_cur, Inches(0.42), fill=ACCENT)
    label(s, BX + Inches(0.14), yy + Inches(0.35), Inches(3),
          f"現状 {'176万人' if i == 0 else '60,307人'}", size=11, color=WHITE)
    label(s, BX + w_cur + Inches(0.12), yy + Inches(0.35), Inches(2.6),
          f"GAP {'26.4万人' if i == 0 else '27,447人'}", size=11, color=WARN)

sh = rect(s, ML + Inches(9.72), bar_y + Inches(0.36), Inches(2.17), Inches(2.05),
          fill=PANEL)
tf = sh.text_frame
text(tf, [("To Be の置き方", dict(size=10.5, bold=True, color=ACCENT))],
     first=True, space_after=4)
text(tf, [("リクルートの現在値に並ぶことを2030年の到達点とする\n（リクルート横ばい前提）",
           dict(size=10.5, color=BODY))], space_after=6, line=1.25)
text(tf, [("Indeed統合で伸びれば目標は上振れする",
           dict(size=10, color=MUTED))], space_after=0, line=1.25)

# ================================================================ スライド 6
s = slide_new("GAPを、登録者数と決定率に分けて見る",
              kicker="① 前提 ─ GAPの分解",
              note="決定率は両社が公表している指標ではなく、実数2つからの逆算値です。")

steps = [
    ("① 式を置く", "就職者数 ＝ 年間登録者数 × 決定率"),
    ("② 変形する", "決定率 ＝ 就職者数 ÷ 年間登録者数"),
]
for i, (k, v) in enumerate(steps):
    yy = TOP + Inches(i * 0.66)
    label(s, ML, yy + Inches(0.08), Inches(1.6), k, size=12, color=ACCENT)
    label(s, ML + Inches(1.6), yy, Inches(8), v, size=18)
label(s, ML + Inches(1.6), TOP + Inches(1.26), Inches(8),
      "※ 公表指標ではなく、実数2つからの逆算値", size=11, bold=False, color=MUTED)

# ③ 代入する（全幅の表）
label(s, ML, TOP + Inches(1.78), Inches(1.6), "③ 代入する", size=12, color=ACCENT)
TX, TW = ML + Inches(1.6), Inches(10.29)
cw = [Inches(2.4), Inches(2.63), Inches(2.63), Inches(2.63)]
cxs = [TX, TX + cw[0], TX + cw[0] + cw[1], TX + cw[0] + cw[1] + cw[2]]
for i, h in enumerate(["", "就職者数", "年間登録者数", "決定率"]):
    label(s, cxs[i], TOP + Inches(1.78), cw[i], h, size=11.5, color=MUTED,
          align=(PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT))
rect(s, TX, TOP + Inches(2.08), TW, Inches(0.018), fill=INK)

calc = [("パーソル", "60,307 人", "1,760,000 人", "3.43%"),
        ("リクルート", "87,754 人", "2,024,000 人", "4.34%")]
for i, row in enumerate(calc):
    yy = TOP + Inches(2.24 + i * 0.66)
    label(s, cxs[0], yy + Inches(0.07), cw[0], row[0], size=14)
    for j in (1, 2, 3):
        label(s, cxs[j], yy, cw[j], row[j], size=17, color=INK,
              align=PP_ALIGN.RIGHT)
    rect(s, TX, yy + Inches(0.5), TW, Inches(0.012), fill=LINE)

# ④ 差を出す
label(s, ML, TOP + Inches(3.72), Inches(1.6), "④ 差を出す", size=12, color=ACCENT)
tfd = tb(s, TX, TOP + Inches(3.62), Inches(4.7), Inches(0.5))
text(tfd, [("0.91", dict(size=26, bold=True, color=INK)),
           (" ポイント", dict(size=14, bold=True, color=INK))], first=True,
     space_after=0)

# ================================================================ スライド 7
s = slide_new("リサーチで分かったこと", kicker="② リサーチ内容 ─ ファクト",
              note="出典：パーソル市場調査（2022年）／厚生労働省 令和4年版 労働経済の分析／"
                   "パーソル総合研究所 人事部大研究／各社IR・公式サイト。")

facts = [
    ("個人", [
        [("年間登録者数：リクルート 202万4,000人 vs パーソル 176万人",
          dict(bold=True, color=INK)), ("（差 26万4,000人）", {})],
        [("転職希望者の 67%（約358万人）が doda と接点なし",
          dict(bold=True, color=INK))],
        [("転職希望者のうち、実際に転職を実現するのは 1〜2割程度", {})],
    ]),
    ("法人", [
        [("就職者数：リクルート 87,754人 vs パーソル 60,307人",
          dict(bold=True, color=INK)), ("（差 27,447人）", {})],
        [("求人掲載数：リクルート 約80万件 vs パーソル 約30万件", {})],
        [("入社後7か月以上の定着率 94.5%", dict(bold=True, color=INK))],
    ]),
    ("市場", [
        [("2030年に全国で 644万人 の人手不足", dict(bold=True, color=INK))],
        [("企業の 約55% が「最適な人員配置」に課題", {})],
    ]),
]
y = TOP
for head, lines in facts:
    label(s, ML, y, Inches(1.4), head, size=15, color=ACCENT)
    rect(s, ML + Inches(1.4), y + Inches(0.14), CW - Inches(1.4), Inches(0.012),
         fill=LINE)
    y += Inches(0.42)
    for ln in lines:
        rect(s, ML + Inches(0.16), y + Inches(0.12), Inches(0.08), Inches(0.08),
             fill=ACCENT, shape=MSO_SHAPE.OVAL)
        tfl = tb(s, ML + Inches(0.48), y, CW - Inches(0.48), Inches(0.32))
        p = para(tfl, first=True, space_after=0, line=1.2)
        for s_, opt in ln:
            opt = dict(opt)
            r = p.add_run()
            r.text = s_
            _set_font(r, opt.pop("size", 14.5), **opt)
        y += Inches(0.4)
    y += Inches(0.2)

label(s, ML, y + Inches(0.02), CW,
      "原因ではないと確認できたもの：手数料率（両社に差なし）／6か月以内離職率"
      "（むしろパーソルが優位）", size=12, bold=False, color=MUTED)

# ================================================================ スライド 9
s = slide_new("いま挙がっている課題は5つ ── まだ発散段階です",
              kicker="③ 課題（発散）",
              note="課題3・4・5は、私たちが置いた定量定義と接続していません。"
                   "また課題5だけ粒度が違い（HiProという1サービスの話）、他と並べる階層ではない可能性があります。")

issues = [
    ("課題1", "登録者数・プラットフォーム規模の格差と未リーチ層への接触不足", True),
    ("課題2", "登録者から成果（就職・決定）への転換力の低さ", True),
    ("課題3", "点の支援（単発関係）によるLTVの低さと、一貫したトータルソリューションの不備",
     False),
    ("課題4", "求人「量」の劣勢と、定着・活躍など「質」を証明・評価する手段の不足", False),
    ("課題5", "HiPro（副業・フリーランス）の規模感と、dodaからの連携ストーリーの脆弱さ",
     False),
]
yy = TOP + Inches(0.06)
for no, name, linked in issues:
    hh = Inches(0.86)
    rect(s, ML, yy, CW, hh, fill=(WHITE if linked else PANEL),
         line_color=(ACCENT if linked else LINE), dash=(not linked))
    rect(s, ML, yy, Inches(0.05), hh, fill=(ACCENT if linked else LINE))
    label(s, ML + Inches(0.28), yy + Inches(0.14), Inches(1.1), no, size=11,
          color=(ACCENT if linked else MUTED))
    label(s, ML + Inches(0.28), yy + Inches(0.42), Inches(8.6), name, size=15,
          color=(INK if linked else BODY))
    tag = "定量定義と接続する" if linked else "定量定義の外側"
    label(s, ML + CW - Inches(2.25), yy + Inches(0.32), Inches(2.0), tag,
          size=10.5, bold=False, color=(ACCENT if linked else MUTED),
          align=PP_ALIGN.RIGHT)
    yy += hh + Inches(0.16)

# ================================================================ スライド 10
s = slide_new("課題1・2 ── 定量定義と直接つながる2つ", kicker="③ 課題（発散）",
              note="記載は 9/4 チーム共有ドキュメント「③課題の整理」の内容そのままです。"
                   "深掘りの論点はかな、仮説はホワイトボードとみおの発言によります。")

card(s, ML, TOP, Inches(5.82), Inches(3.9),
     "登録者数・プラットフォーム規模の格差と\n未リーチ層への接触不足",
     [
         [("リクルートに対して個人登録者数（26.4万人差）や法人シェア率で劣っており、"
           "転職希望者の67%（358万人）と接点が持てていない", dict(size=12.2))],
         [("", dict(size=6))],
         [("深掘りの論点（かな）", dict(size=11, bold=True, color=ACCENT))],
         [("認知・第一想起／集客チャネル・顧客接点／求人数・企業数による登録魅力度／"
           "獲得できている年代・職種・地域", dict(size=11.8))],
         [("", dict(size=6))],
         [("ホワイトボードで出た仮説", dict(size=11, bold=True, color=WARN))],
         [("登録時の入力コスト・UI（年収レンジ）・Web上の企業名表示",
           dict(size=11.8, bold=True, color=INK))],
     ], badge="課題1 ｜ 発案：リーダー・まい・けーた", tsize=15.5)

card(s, ML + Inches(6.07), TOP, Inches(5.82), Inches(3.9),
     "登録者から成果（就職・決定）への\n転換力の低さ",
     [
         [("母数の差だけでなく、登録者から就職に至る決定率（パーソル3.43% vs "
           "リクルート4.34%）そのものに0.91ポイントの構造的な弱さがある",
           dict(size=12.2))],
         [("", dict(size=6))],
         [("深掘りの論点（かな）", dict(size=11, bold=True, color=ACCENT))],
         [("登録者に紹介できる求人数／求人紹介→応募率／応募→書類→面接→内定の通過率／"
           "内定→承諾・入社率／CA・法人営業の支援体制・生産性", dict(size=11.8))],
         [("", dict(size=6))],
         [("ホワイトボード・みおの仮説", dict(size=11, bold=True, color=WARN))],
         [("CAの質（専門性／雑務量）／最終意思決定までが遅い",
           dict(size=11.8, bold=True, color=INK))],
     ], accent=WARN, badge="課題2 ｜ 発案：まい・けーた", tsize=15.5)

# ================================================================ スライド 11
s = slide_new("課題3・4・5 ── 定量定義の外側にある3つ", kicker="③ 課題（発散）",
              note="3つとも市場環境とは接続しますが、GAPの数字とは接続していません。"
                   "課題5は他と階層が違う可能性があります。")

c3 = [
    [("人材紹介事業単体では転職時のみの「点の支援」で終わり、転職しない期間の接点が"
      "少なく、顧客生涯価値（LTV）を高められていない", dict(size=11.8))],
]
c4 = [
    [("求人掲載数（30万件 vs 80万件）では競合に勝てず、高い定着率（94.5%）や入社後の"
      "活躍度を法人側に客観的に示すデータ・手段が不足している", dict(size=11.8))],
]
c5 = [
    [("HiProは急成長しているが全社LTVを支えるには規模（10万人）が小さく、dodaからの"
      "実際の流入率（クロスユース率）が不明。また利用者の7割が現職会社員であり、"
      "「転職後の受け皿」というストーリーとズレがある", dict(size=11.8))],
]
for i, (badge, ttl, lines) in enumerate([
        ("課題3 ｜ 発案：リーダー・まい・みお・かな",
         "点の支援によるLTVの低さと、\nトータルソリューションの不備", c3),
        ("課題4 ｜ 発案：リーダー・まい・みお",
         "求人「量」の劣勢と、「質」を\n証明・評価する手段の不足", c4),
        ("課題5 ｜ 発案：まい",
         "HiProの規模感と、dodaからの\n連携ストーリーの脆弱さ", c5)]):
    x = ML + i * (Inches(3.83) + Inches(0.2))
    card(s, x, TOP, Inches(3.83), Inches(2.5), ttl, lines, accent=MUTED,
         fill=PANEL, badge=badge, tsize=14, dash=True)

sh = rect(s, ML, Inches(4.32), CW, Inches(1.08), fill=WARN_L, line_color=WARN)
tf = sh.text_frame
text(tf, [("自分たちで気づいている弱点", dict(size=11, bold=True, color=WARN))],
     first=True, space_after=4)
text(tf, [("課題3・4・5は、私たちが置いた定量定義（登録者数・就職者数）と接続していません。"
           "定性の議論から出てきたもので、GAPのどこを埋めるのかを説明できていません。"
           "また課題5だけ粒度が違い、他と並べる階層ではない可能性があります。",
           dict(size=12.5, color=INK))], space_after=0, line=1.3)

# ================================================================ スライド 13
s = slide_new("この先の進め方", kicker="④ クロージング")

steps = [
    ("STEP 1", "本日 〜 次回の中間FBまで",
     ["現場社員の方へのヒアリングと、メンターからのフィードバックをもとに議論する",
      "課題の捉え方を詰めたうえで、解決策の方向性を導き出す"], ACCENT),
    ("STEP 2", "次回の中間FB",
     ["解決策を含めた全体をレビューいただく"], MUTED),
    ("STEP 3", "最終発表",
     ["いただいたご指摘をチームでまとめ、①前提／②リサーチ／③課題／④解決策／"
      "⑤1か月の行動として発表する"], MUTED),
]
yy = TOP + Inches(0.14)
for badge, ttl, lines, c in steps:
    hh = Inches(1.52) if len(lines) > 1 else Inches(1.34)
    rect(s, ML, yy, CW, hh, fill=(ACCENT_L if c == ACCENT else PANEL))
    rect(s, ML, yy, Inches(0.06), hh, fill=c)
    label(s, ML + Inches(0.34), yy + Inches(0.2), Inches(2), badge, size=10.5,
          color=c)
    label(s, ML + Inches(0.34), yy + Inches(0.48), Inches(6), ttl, size=18,
          color=INK, h=Inches(0.4))
    for j, ln in enumerate(lines):
        yl = yy + Inches(0.94 + j * 0.34)
        rect(s, ML + Inches(0.38), yl + Inches(0.12), Inches(0.08), Inches(0.08),
             fill=c, shape=MSO_SHAPE.OVAL)
        label(s, ML + Inches(0.66), yl, CW - Inches(1.0), ln, size=13,
              bold=False, color=BODY, h=Inches(0.32))
    yy += hh + Inches(0.24)

# ================================================================ スライド 14
s = slide_new("本日いただきたいフィードバック（再掲）", kicker="④ クロージング",
              note="この2点について、ご指摘をいただけますと幸いです。")

fb_cards(s)

# ============================================================ Appendix 扉なし
# --- Appendix 1
s = slide_new("出典一覧", kicker="APPENDIX 1")
src = [
    ("個人KPI｜リクルート 202万4,000人", "r-agent.com 注記※2（申込者数・2025/4/1〜2026/3/31）"),
    ("個人KPI｜パーソル 176万人",
     "PERSOL IR-DAY 2024 p.10「転職希望者数」脚注「FY23累計」"),
    ("法人KPI｜リクルート 87,754人",
     "厚生労働省 人材サービス総合サイト／許可番号 13-ユ-317880\n"
     "（4か月以上の有期および無期の就職者数）"),
    ("法人KPI｜パーソル 60,307人",
     "厚生労働省 人材サービス総合サイト／許可番号 13-ユ-304785（同上）"),
    ("転職希望者の67%（358万人）がdodaと接点なし", "パーソル市場調査（2022年）"),
    ("転職実現に至るのは1〜2割程度",
     "厚生労働省 令和4年版 労働経済の分析 第Ⅱ部第3章"),
    ("企業の約55%が「最適な人員配置」に課題",
     "パーソル総合研究所「人事部大研究」"),
    ("HiPro 登録者10万名突破（+29.1%）", "パーソルキャリア プレスリリース（2025年10月）"),
]
yy = TOP
for k, v in src:
    label(s, ML, yy, Inches(4.6), k, size=11.5, color=INK)
    label(s, ML + Inches(4.75), yy, Inches(7.14), v, size=11, bold=False,
          color=MUTED)
    rect(s, ML, yy + Inches(0.42), CW, Inches(0.012), fill=LINE)
    yy += Inches(0.6)
label(s, ML, yy + Inches(0.12), CW,
      "※ 人材サービス総合サイトの数値は2026年8月24日取得。発表前に再取得して更新確認する。",
      size=10.5, bold=False, color=WARN)

# --- Appendix 2
s = slide_new("1か月間の活動記録", kicker="APPENDIX 2")
acts = [
    ("8/19", "キックオフ（45分）", "役割分担・アウトプットの型・前提の論点出し"),
    ("8/24", "MTG（1時間57分）", "各自の宿題共有／「選ばれる」の定義案を持ち寄り"),
    ("8/29", "MTG（1時間）", "定量定義を確定（就職者数・年間登録者数）／リサーチ深掘り"),
    ("9/4", "対面MTG・新宿（約5時間）", "ファクトの統合／課題の発散と整理／中間FBの方針決め"),
    ("9/7", "中間フィードバック（本日）", "定量定義と課題設定についてご指摘をいただく"),
]
yy = TOP
for d, t, v in acts:
    rect(s, ML, yy, CW, Inches(0.86), fill=WHITE, line_color=LINE)
    rect(s, ML, yy, Inches(0.05), Inches(0.86), fill=ACCENT)
    label(s, ML + Inches(0.24), yy + Inches(0.26), Inches(0.9), d, size=14,
          color=ACCENT)
    label(s, ML + Inches(1.3), yy + Inches(0.14), Inches(3.6), t, size=13.5)
    label(s, ML + Inches(1.3), yy + Inches(0.44), Inches(10.3), v, size=11,
          bold=False, color=MUTED)
    yy += Inches(0.98)

# --- Appendix 3
s = slide_new("その他のファクト", kicker="APPENDIX 3")
card(s, ML, TOP, Inches(5.82), Inches(2.35), "パーソルの強み",
     [
         [("スポットワーク「シェアフル」は業界2位（ユーザー数5.5万人）", dict(size=12))],
         [("地方拠点539か所／地方銀行との強力な連携基盤", dict(size=12))],
         [("プロ人材・副業「HiPro」登録者10万人突破（前年比+29.1%）", dict(size=12))],
         [("派遣の専門性が高く、幅広い事業基盤（派遣・正社員・BPO・RPO）", dict(size=12))],
     ], tsize=15)
card(s, ML + Inches(6.07), TOP, Inches(5.82), Inches(2.35), "パーソルの弱み",
     [
         [("労働集約型ビジネス／原価率が高く営業利益率 約4%", dict(size=12))],
         [("地方シェア率の低さ（約0.1〜0.5%）", dict(size=12))],
         [("LTVの低さ・点の支援になっている", dict(size=12))],
     ], accent=WARN, tsize=15)

label(s, ML, Inches(4.28), CW, "外部環境（PEST）", size=14)
pest = [
    ("P 政治", "働き方改革・副業推進・リスキリングによる労働移動の後押し。人的資本の情報開示義務化。"),
    ("E 経済", "少子高齢化による構造的な人手不足（2030年に644万人）。採用コスト高騰で定着への投資意欲が高まる。"),
    ("S 社会", "終身雇用の崩壊とキャリア観の多様化。個人は中長期的に相談できる拠り所を求めている。"),
    ("T 技術", "条件マッチングのAIコモディティ化。一方でAIにできない人的介入の相対価値が高まる。"),
]
for i, (k, v) in enumerate(pest):
    x = ML + (i % 2) * (Inches(5.82) + Inches(0.25))
    yy = Inches(4.66) + (i // 2) * Inches(1.02)
    sh = rect(s, x, yy, Inches(5.82), Inches(0.88), fill=PANEL)
    tf = sh.text_frame
    text(tf, [(k, dict(size=11, bold=True, color=ACCENT))], first=True,
         space_after=3)
    text(tf, [(v, dict(size=11.2, color=BODY))], space_after=0, line=1.25)

# ================================================================ 保存
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"saved: {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
