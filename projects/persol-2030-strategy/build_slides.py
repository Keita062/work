# -*- coding: utf-8 -*-
"""
内定者研修 チームA 中間フィードバック依頼 スライド生成

05_中間FB_スライド構成.md の構成に従って .pptx を生成する。
構成ドキュメントを修正 → このスクリプトを修正 → 再実行、の順で更新する。

    .venv\\Scripts\\python.exe projects\\persol-2030-strategy\\build_slides.py

2026-09-07 改訂（第3版）：
  - 「本日のゴール」の位置を上げ、「次回ご報告します」の1行はゴール側にだけ残す
  - 「いただいた課題」からグレーの補足と出典を削除
  - 「選ばれる」の定義を 法人視点／個人視点／結論 の3枚に分割
  - 現状①（年間登録者数）を横棒バーから数値表示に変更
  - 差分スライドに、リーチ率の算出式と出典を明記
  - 差分の要因を「就職者数が足りない → 登録者数を増やす → 決定率を上げる」に整理
  - 課題1・2／課題3・4・5 の2枚を削除（課題は発散マップの1枚のみ）
  - Appendix の前に Appendix の目次を追加
  - 2030年のあるべき決定率を 5.0% に確定（なりゆき2.4%・リクルート4.34%・
    自社計画の延長5.3% の3水準から置いた。→ 冒頭の定数コメント）
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

# ------------------------------------------------------------ 2030年の目標値
#
# 決定率 ＝ 就職者数 ÷ 年間登録者数（実数2つからの逆算値）
#
#   As Is       パーソル 3.43% ／ リクルート 4.34%
#   なりゆき    2.4%  … 就職者数の実績CAGR +3.4%/年（56,434→60,307）で分子を、
#                       doda会員数の伸び +11.3%/年 で分母を5年延長した値
#   会社計画    5.3%  … FY2028「登録決定率1.3倍」（＝4.45%）を同ペースで2030年度まで延長
#   → 目標      5.0%  … リクルート4.34%を上回り、会社計画の延長5.3%の内側に収まる水準
#
# 算出スクリプト: scratchpad/calc_rate.py（このファイルのコメントと同じ計算）

TO_BE_RATE = "5.0%"          # 2030年のあるべき決定率
TO_BE_PLACED = "101,200 人"  # 2,024,000 × 5.0%
GAP_PLACED = "40,893 人"     # 101,200 − 60,307
GAP_PT = "1.57 pt"           # 5.00% − 3.43%

RATE_NATURAL = "2.4%"        # なりゆき（このまま）
RATE_PLAN = "5.3%"           # 会社計画（FY2028 1.3倍）の2030年延長

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


def band(slide, y, h, lines, fill=PANEL, bar=None, x=ML, w=CW):
    """帯。lines は text() に渡す chunks のリスト。"""
    sh = rect(slide, x, y, w, h, fill=fill)
    if bar:
        rect(slide, x, y, Inches(0.06), h, fill=bar)
    tf = sh.text_frame
    tf.margin_left = Inches(0.30)
    tf.margin_top = Inches(0.16)
    for i, ln in enumerate(lines):
        text(tf, ln, first=(i == 0), space_after=4, line=1.25)
    return sh


def scope_band(slide, y=TOP):
    """データが取れている範囲／時間軸の帯。"""
    sh = rect(slide, ML, y, CW, Inches(0.54), fill=PANEL)
    sh.text_frame.margin_top = Inches(0.14)
    text(sh.text_frame,
         [("データが取れている範囲 ", dict(size=11, bold=True, color=ACCENT)),
          ("リクルートエージェント ／ パーソル（doda）",
           dict(size=13, bold=True, color=INK)),
          ("　　｜　　", dict(size=13, color=LINE)),
          ("時間軸 ", dict(size=11, bold=True, color=ACCENT)),
          ("2030年度", dict(size=13, bold=True, color=INK))],
         first=True, space_after=0, align=PP_ALIGN.CENTER)
    return sh


# --- セクション扉（大きなセクションの前に1枚はさむ） --------------------------

SECTIONS = [("01", "前提"), ("02", "差分の要因"), ("03", "課題")]


def section_slide(no, title, sub):
    """中央に大きく1つだけ置く中扉。"""
    sl = prs.slides.add_slide(BLANK)
    _page["n"] += 1
    rect(sl, Inches(0), Inches(0), SW, Inches(0.1), fill=ACCENT)

    label(sl, ML, Inches(2.62), CW, no, size=15, color=ACCENT,
          align=PP_ALIGN.CENTER)
    tfs = tb(sl, ML, Inches(3.00), CW, Inches(0.9))
    text(tfs, [(title, dict(size=44, bold=True, color=INK))], first=True,
         space_after=0, align=PP_ALIGN.CENTER, line=1.15)
    rect(sl, Inches(6.29), Inches(4.06), Inches(0.75), Inches(0.035), fill=ACCENT)
    label(sl, ML, Inches(4.32), CW, sub, size=15, bold=False, color=BODY,
          align=PP_ALIGN.CENTER)

    # 下端に進行位置を示す
    slot_ = Emu(int(CW / 3))
    for i, (n_, t_) in enumerate(SECTIONS):
        on = (n_ == no)
        x = ML + Emu(int(slot_ * i))
        label(sl, x, Inches(6.10), slot_, f"{n_}  {t_}", size=11.5,
              bold=on, color=(ACCENT if on else LINE), align=PP_ALIGN.CENTER)
        rect(sl, x + Emu(int(slot_ * 0.30)), Inches(6.40),
             Emu(int(slot_ * 0.40)), Inches(0.03),
             fill=(ACCENT if on else LINE))
    _footer(sl)
    return sl


# --- 本日いただきたいFB（スライド4・17で同一のものを再掲する） -----------------

FB_ITEMS = [
    (ACCENT, "1", "「選ばれる」の定量定義は、これでよいか",
     "法人視点の定量指標は、まだ定まっていません。"),
    (WARN, "2", "5つの課題のうち、どれを本命に絞るべきか", None),
]


def fb_body(slide):
    """いただきたいFB 2点。スライド4・17で共通。"""
    y = TOP + Inches(0.62)
    for ac, no, q, sub in FB_ITEMS:
        h = Inches(1.62) if sub else Inches(1.32)
        rect(slide, ML, y, CW, h, fill=PANEL)
        rect(slide, ML, y, Inches(0.06), h, fill=ac)

        rect(slide, ML + Inches(0.46), y + Inches(0.36), Inches(0.62),
             Inches(0.62), fill=ac, shape=MSO_SHAPE.OVAL)
        label(slide, ML + Inches(0.46), y + Inches(0.52), Inches(0.62), no,
              size=17, color=WHITE, align=PP_ALIGN.CENTER)

        label(slide, ML + Inches(1.42), y + Inches(0.38 if sub else 0.44),
              CW - Inches(1.9), q, size=24, color=INK, h=Inches(0.52))
        if sub:
            label(slide, ML + Inches(1.42), y + Inches(1.00), CW - Inches(1.9),
                  sub, size=12.5, bold=False, color=BODY, h=Inches(0.36))
        y += h + Inches(0.40)


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
    ("01", "前提", "「選ばれる」をどう定義したか（定義／あるべき姿／現状／差分）"),
    ("02", "差分の要因", "なぜ差がついているのか（仮説）"),
    ("03", "課題", "いま我々が課題と捉えているもの（発散段階）"),
    ("04", "いただきたいフィードバック", "本日ご指摘いただきたい2点"),
]
for i, (no, ttl, sub) in enumerate(agenda):
    yy = TOP + Inches(i * 0.74)
    rect(s, ML, yy, Inches(0.6), Inches(0.6), fill=ACCENT_L)
    label(s, ML, yy + Inches(0.16), Inches(0.6), no, size=13, color=ACCENT,
          align=PP_ALIGN.CENTER)
    label(s, ML + Inches(0.86), yy + Inches(0.02), Inches(4.4), ttl, size=16)
    label(s, ML + Inches(0.86), yy + Inches(0.32), Inches(9), sub, size=11.5,
          bold=False, color=MUTED)

# 研修課題5要素のうち、どこまでを本日扱うか（ステップライン）
rect(s, ML, Inches(4.72), CW, Inches(0.015), fill=LINE)
label(s, ML, Inches(4.92), Inches(8), "研修課題の5要素のうち、本日は ①〜③ まで",
      size=12, color=INK)
label(s, ML, Inches(5.22), CW,
      "②リサーチしたファクトは、前提でお話しする数字と Appendix に載せています。",
      size=11.5, bold=False, color=MUTED)

steps = [("① 前提", True), ("② リサーチ", True), ("③ 課題", True),
         ("④ 解決策", False), ("⑤ 1か月の行動", False)]
slot = Emu(int(CW / 5))
cx = [ML + Emu(int(slot * (i + 0.5))) for i in range(5)]
LY = Inches(5.90)          # ライン中心
DOT = Inches(0.24)

rect(s, cx[0], LY - Inches(0.008), cx[4] - cx[0], Inches(0.016), fill=LINE)
rect(s, cx[0], LY - Inches(0.012), cx[2] - cx[0], Inches(0.024), fill=ACCENT)

for i, (name, on) in enumerate(steps):
    rect(s, cx[i] - DOT / 2, LY - DOT / 2, DOT, DOT,
         fill=(ACCENT if on else WHITE),
         line_color=(None if on else LINE), line_w=1.25,
         shape=MSO_SHAPE.OVAL)
    label(s, cx[i] - slot / 2, LY + Inches(0.20), slot, name, size=12.5,
          color=(ACCENT if on else MUTED), align=PP_ALIGN.CENTER)

label(s, cx[0] - slot / 2, LY + Inches(0.54), cx[2] - cx[0] + slot,
      "本日はここまで", size=11.5, color=ACCENT, align=PP_ALIGN.CENTER)
label(s, cx[3] - slot / 2, LY + Inches(0.54), cx[4] - cx[3] + slot,
      "次回の中間FBでご報告します", size=11.5, bold=False, color=MUTED,
      align=PP_ALIGN.CENTER)

# ================================================================ スライド 3
s = slide_new("本日のゴール", kicker="GOAL")

rect(s, ML, Inches(1.66), CW, Inches(2.10), fill=ACCENT_L)
rect(s, ML, Inches(1.66), CW, Inches(0.05), fill=ACCENT)
tfg = tb(s, ML + Inches(0.9), Inches(2.18), CW - Inches(1.8), Inches(1.3))
text(tfg, [("前提（定義・あるべき姿・現状・差分）と、\nいま挙がっている課題について"
            "ご指摘をいただき、\n次回の中間FBまでに何を詰めるかを決める",
            dict(size=23, bold=True, color=INK))],
     first=True, space_after=0, align=PP_ALIGN.CENTER, line=1.45)

band(s, Inches(4.10), Inches(0.72),
     [[("④解決策・⑤1か月の行動は、次回の中間FBでご報告します。",
        dict(size=14, bold=True, color=ACCENT))]],
     fill=PANEL)

# ================================================================ スライド 4
s = slide_new("本日いただきたいフィードバック", kicker="GOAL",
              note="定義の詳細は P.7〜9、課題の詳細は P.19 でご説明します。")
fb_body(s)

# ================================================== スライド 5（セクション扉①）
section_slide("01", "前提",
              "「選ばれる」をどう定義したか ── 定義／あるべき姿／現状／差分")

# ================================================================ スライド 6
s = slide_new("いただいた課題", kicker="① 前提 ─ 課題内容")

rect(s, ML, Inches(2.30), CW, Inches(2.66), fill=PANEL)
rect(s, ML, Inches(2.30), CW, Inches(0.05), fill=ACCENT)

tfq = tb(s, ML + Inches(0.9), Inches(2.86), CW - Inches(1.8), Inches(1.6))
text(tfq, [("「パーソルキャリアが2030年までに\nリクルート様の人材サービスと比較し、\n"
            "法人個人双方から", dict(size=25, bold=True, color=INK)),
           ("選ばれる", dict(size=25, bold=True, color=ACCENT)),
           ("ための戦略を考えよ」", dict(size=25, bold=True, color=INK))],
     first=True, space_after=0, align=PP_ALIGN.CENTER, line=1.42)

# ======================================================== スライド 7・8（定義）


def definition_slide(no, view, items, foot_text, foot_fill, foot_bar):
    sl = slide_new(f"「選ばれる」の定義 {no} {view}",
                   kicker=f"① 前提 ─ 定義（{view}）")
    y = TOP + Inches(0.34)
    for k, v, nl in items:
        rect(sl, ML, y, Inches(1.7), Inches(0.36), fill=ACCENT_L)
        label(sl, ML, y + Inches(0.08), Inches(1.7), k, size=12, color=ACCENT,
              align=PP_ALIGN.CENTER)
        tfd = tb(sl, ML + Inches(2.0), y - Inches(0.06),
                 CW - Inches(2.0), Inches(0.42 * nl))
        text(tfd, [(v, dict(size=17, color=INK))], first=True, space_after=0,
             line=1.34)
        y += Inches(0.40 * nl + 0.44)
    band(sl, Inches(4.60), Inches(0.86), [foot_text], fill=foot_fill,
         bar=foot_bar)
    return sl


definition_slide(
    "①", "法人視点",
    [("定量指標",
      "doda等人材サービスにおける就職率・定着率、"
      "日本企業全体におけるパーソルサービスのシェア率", 2),
     ("定性・状態",
      "単なる採用にとどまらず、育成・人員配置・組織開発まで一括して任される状態、"
      "および長期的・継続的な契約関係（LTV・NPS向上）", 2)],
    [("※ どの指標を主指標に置くかは、まだ定まっていません",
      dict(size=16, bold=True, color=WARN))],
    WARN_L, WARN)

definition_slide(
    "②", "個人視点",
    [("定量指標",
      "登録者数、転職成功率、キャリア選択の幅、"
      "プラットフォーム全体の生涯価値（LTV）", 1),
     ("定性・状態",
      "キャリアのあらゆるフェーズにおいて選ばれ、中長期的に利用され続ける状態", 1)],
    [("本日お話しする数字は、すべてこの個人視点の定義に沿って出しています",
      dict(size=16, bold=True, color=ACCENT))],
    ACCENT_L, ACCENT)

# ================================================================ スライド 9
s = slide_new("「選ばれる」の定義 ── 収束させた結論",
              kicker="① 前提 ─ 定義（結論）",
              note="出典：9/4 チーム共有ドキュメント memo「選ばれる定義 ＝ 決定率が高い状態／"
                   "リクルートに数値で上回る状態」および 9/4 対面MTGのホワイトボード。")

scope_band(s)

rect(s, ML, Inches(2.86), CW, Inches(1.86), fill=ACCENT_L)
rect(s, ML, Inches(2.86), CW, Inches(0.05), fill=ACCENT)
tfc = tb(s, ML, Inches(3.38), CW, Inches(0.7))
text(tfc, [("「選ばれる」 ＝ ", dict(size=27, bold=True, color=INK)),
           ("決定率でリクルートを上回っている状態",
            dict(size=27, bold=True, color=ACCENT))],
     first=True, space_after=0, align=PP_ALIGN.CENTER)
label(s, ML + Inches(1.0), Inches(4.16), CW - Inches(2.0),
      "決定率が何%であるべきかを絶対値で定めるのは難しいため、"
      "競合であるリクルートを上回っている状態を基準に置いています。",
      size=14, bold=False, color=BODY, align=PP_ALIGN.CENTER)

# ================================================================ スライド 10
s = slide_new(f"あるべき姿（2030年）── 決定率 {TO_BE_RATE}",
              kicker="① 前提 ─ あるべき姿 To Be 2030",
              note="※ 決定率は両社の公表指標ではなく、実数2つからの逆算値です。"
                   "なりゆきの分母は、年間登録者数の時系列が非開示のため doda会員数の伸び"
                   "（+11.3%/年）で代用しています。リクルートは横ばい前提です。")

label(s, ML, TOP, Inches(1.9), "決定率とは", size=12, color=ACCENT)
tfv = tb(s, ML + Inches(1.9), TOP - Inches(0.06), Inches(9.9), Inches(0.4))
text(tfv, [("決定率 ＝ 就職者数 ÷ 年間登録者数",
            dict(size=17, bold=True, color=INK)),
           ("　※ 実数2つからの逆算値", dict(size=11, color=MUTED))],
     first=True, space_after=0)

label(s, ML, Inches(2.06), Inches(4.0), "2030年に向けた3つの水準", size=12,
      color=ACCENT)

LVW = Inches(3.83)
levels = [
    ("なりゆき（このまま）", RATE_NATURAL,
     "就職者数は実績CAGR +3.4%/年、\n登録者数は +11.3%/年。分母が先に伸びて下がる",
     MUTED, PANEL),
    ("リクルート（横ばい前提）", "4.34%",
     "「上回っている状態」の下限。\n並ぶだけなら 202.4万人 × 4.34% ＝ 87,754人",
     ACCENT, WHITE),
    ("パーソルの計画を延長", RATE_PLAN,
     "FY2028「登録決定率 1.3倍」（4.45%）を\n2030年度まで同ペースで延長した値",
     ACCENT, WHITE),
]
for i, (head, val, sub, ac, fl) in enumerate(levels):
    x = ML + i * (LVW + Inches(0.2))
    rect(s, x, Inches(2.42), LVW, Inches(1.52), fill=fl, line_color=LINE)
    rect(s, x, Inches(2.42), LVW, Inches(0.05), fill=ac)
    label(s, x, Inches(2.62), LVW, head, size=11, color=ac,
          align=PP_ALIGN.CENTER)
    label(s, x, Inches(2.90), LVW, val, size=30, color=INK,
          align=PP_ALIGN.CENTER, h=Inches(0.55))
    tfl = tb(s, x + Inches(0.2), Inches(3.46), LVW - Inches(0.4), Inches(0.5))
    text(tfl, [(sub, dict(size=10, color=MUTED))], first=True, space_after=0,
         align=PP_ALIGN.CENTER, line=1.25)

rect(s, ML, Inches(4.24), CW, Inches(1.42), fill=WARN_L, line_color=WARN,
     line_w=1.5, dash=True)
tfr = tb(s, ML, Inches(4.44), CW, Inches(0.66))
text(tfr, [("あるべき姿　決定率 ＝ ", dict(size=22, bold=True, color=INK)),
           (TO_BE_RATE, dict(size=40, bold=True, color=WARN))],
     first=True, space_after=0, align=PP_ALIGN.CENTER)
label(s, ML, Inches(5.22), CW,
      "リクルートの現在値 4.34% を上回り、自社計画の延長線 5.3% の内側に収まる水準として置く",
      size=13, bold=False, color=BODY, align=PP_ALIGN.CENTER)

# ================================================================ スライド 11
s = slide_new("現状① 年間登録者数 ── 26万4,000人の差",
              kicker="① 前提 ─ 現状 As Is ①",
              note="※ 定義がずれています：リクルート＝エージェント単体・FY2025 ／ "
                   "パーソル＝Career SBU全体・FY2023。出典は Appendix 1。")

band(s, TOP, Inches(0.58),
     [[("比較対象　", dict(size=11, bold=True, color=ACCENT)),
       ("リクルート側 ＝ リクルートエージェント　　／　　パーソル側 ＝ パーソル（doda）",
        dict(size=13.5, bold=True, color=INK))]])

for i, (name, val) in enumerate([("リクルートエージェント", "202万4,000人"),
                                 ("パーソル（doda）", "176万人")]):
    x = ML + i * (Inches(5.82) + Inches(0.25))
    sh = rect(s, x, Inches(2.32), Inches(5.82), Inches(1.62), fill=WHITE,
              line_color=LINE)
    rect(s, x, Inches(2.32), Inches(5.82), Inches(0.05), fill=ACCENT)
    label(s, x, Inches(2.60), Inches(5.82), name, size=13, color=ACCENT,
          align=PP_ALIGN.CENTER)
    label(s, x, Inches(3.02), Inches(5.82), val, size=36, color=INK,
          align=PP_ALIGN.CENTER, h=Inches(0.7))

rect(s, ML, Inches(4.26), CW, Inches(1.46), fill=WARN_L, line_color=WARN)
tfg = tb(s, ML, Inches(4.46), CW, Inches(0.7))
text(tfg, [("差　", dict(size=15, bold=True, color=WARN)),
           ("26万4,000人", dict(size=36, bold=True, color=WARN))],
     first=True, space_after=0, align=PP_ALIGN.CENTER)
label(s, ML, Inches(5.26), CW,
      "同じ市場に対して、パーソルは年間で26万4,000人ぶん少ない", size=14,
      bold=False, color=INK, align=PP_ALIGN.CENTER)

# ================================================================ スライド 12
s = slide_new("現状② 決定率 ── 0.91ポイントの差",
              kicker="① 前提 ─ 現状 As Is ②",
              note="決定率は両社が公表している指標ではなく、実数2つからの逆算値です"
                   "（P.10 で置いた式）。")

label(s, ML, TOP, Inches(1.9), "代入する", size=12, color=ACCENT)
TX, TW = ML + Inches(1.9), Inches(9.99)
cw = [Inches(2.1), Inches(2.63), Inches(2.63), Inches(2.63)]
cxs = [TX, TX + cw[0], TX + cw[0] + cw[1], TX + cw[0] + cw[1] + cw[2]]
for i, h in enumerate(["", "就職者数", "年間登録者数", "決定率"]):
    label(s, cxs[i], TOP, cw[i], h, size=11.5, color=MUTED,
          align=(PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT))
rect(s, TX, TOP + Inches(0.30), TW, Inches(0.018), fill=INK)

for i, row in enumerate([("パーソル", "60,307 人", "1,760,000 人", "3.43%"),
                         ("リクルート", "87,754 人", "2,024,000 人", "4.34%")]):
    yy = TOP + Inches(0.48 + i * 0.78)
    label(s, cxs[0], yy + Inches(0.10), cw[0], row[0], size=15)
    for j in (1, 2, 3):
        label(s, cxs[j], yy, cw[j], row[j], size=20, color=INK,
              align=PP_ALIGN.RIGHT)
    rect(s, TX, yy + Inches(0.58), TW, Inches(0.012), fill=LINE)

label(s, ML, TOP + Inches(2.34), Inches(1.9), "差を出す", size=12, color=ACCENT)
tfd = tb(s, TX, TOP + Inches(2.22), Inches(6.0), Inches(0.5))
text(tfd, [("0.91", dict(size=30, bold=True, color=WARN)),
           (" ポイント", dict(size=15, bold=True, color=WARN))],
     first=True, space_after=0)

band(s, Inches(4.86), Inches(0.78),
     [[("同じ人数を集めても、就職に至る割合そのものが 0.91ポイント 低い",
        dict(size=15.5, bold=True, color=ACCENT))]],
     fill=ACCENT_L, bar=ACCENT)

# ================================================================ スライド 13
s = slide_new(f"差分① 就職者数 ── {GAP_PLACED} 増やす必要がある",
              kicker="① 前提 ─ 差分 GAP ①",
              note="To Be 就職者数 ＝ 年間登録者数 202万4,000人（リクルートの現在値に並ぶ）"
                   f"× 決定率 {TO_BE_RATE} ＝ {TO_BE_PLACED}。")

gcx = [ML, ML + Inches(3.0), ML + Inches(6.0), ML + Inches(9.0)]
gcw = [Inches(3.0), Inches(3.0), Inches(3.0), Inches(2.89)]
for i, h in enumerate(["", "あるべき姿 To Be 2030", "現状 As Is", "GAP"]):
    label(s, gcx[i], TOP + Inches(0.10), gcw[i], h, size=11.5,
          color=(ACCENT if i == 1 else MUTED))
rect(s, ML, TOP + Inches(0.40), CW, Inches(0.018), fill=INK)

for i, (k, b, a, g) in enumerate([("決定率", TO_BE_RATE, "3.43%", GAP_PT),
                                  ("就職者数", TO_BE_PLACED, "60,307 人",
                                   GAP_PLACED)]):
    yy = TOP + Inches(0.62 + i * 0.90)
    label(s, gcx[0], yy + Inches(0.14), gcw[0], k, size=15)
    label(s, gcx[1], yy, gcw[1], b, size=26, color=WARN)
    label(s, gcx[2], yy, gcw[2], a, size=26, color=INK)
    label(s, gcx[3], yy, gcw[3], g, size=26, color=WARN)
    rect(s, ML, yy + Inches(0.68), CW, Inches(0.012), fill=LINE)

band(s, Inches(4.30), Inches(0.90),
     [[("決定率を 3.43% から ", dict(size=16, bold=True, color=INK)),
       (TO_BE_RATE, dict(size=16, bold=True, color=WARN)),
       (" に上げるには、就職者数を ", dict(size=16, bold=True, color=INK)),
       (GAP_PLACED, dict(size=16, bold=True, color=WARN)),
       (" 増やす必要があります", dict(size=16, bold=True, color=INK))]],
     fill=ACCENT_L, bar=ACCENT)

# ================================================================ スライド 14
s = slide_new("差分② 年間登録者数 ── 伸びしろは「未リーチ層」にある",
              kicker="① 前提 ─ 差分 GAP ②",
              note="出典：転職希望者534万人・未接点67%（約358万人）＝ パーソルHD "
                   "IR-DAY 2024 説明資料（Career SBU）p.14／パーソルキャリア 2022年市場調査。")

band(s, TOP, Inches(0.66),
     [[("転職希望者 534万人 のうち、", dict(size=13.5, color=BODY)),
       ("67%（約358万人）が doda と接点を持っていない",
        dict(size=15, bold=True, color=INK))]],
     fill=PANEL)

# 左：未リーチ層の規模
sh = rect(s, ML, Inches(2.36), Inches(4.6), Inches(1.90), fill=WARN_L,
          line_color=WARN, line_w=1.5, dash=True)
label(s, ML, Inches(2.62), Inches(4.6), "doda と接点を持っていない転職希望者",
      size=12, color=WARN, align=PP_ALIGN.CENTER)
label(s, ML, Inches(3.06), Inches(4.6), "358万人", size=44, color=WARN,
      align=PP_ALIGN.CENTER, h=Inches(0.9))

# 右：リーチ率
RX2 = ML + Inches(4.85)
RW2 = Inches(7.04)
sh = rect(s, RX2, Inches(2.36), RW2, Inches(1.90), fill=WHITE, line_color=LINE)
tfr2 = sh.text_frame
tfr2.margin_left = Inches(0.28)
tfr2.margin_top = Inches(0.22)
text(tfr2, [("リーチ率 ＝ 年間登録者数 ÷ 転職希望者534万人",
             dict(size=14, bold=True, color=INK))], first=True, space_after=3)
text(tfr2, [("※ 公表値ではなく、この2つからの逆算値", dict(size=10.5, color=MUTED))],
     space_after=10)
text(tfr2, [("パーソル　　176万 ÷ 534万 ＝ ", dict(size=13.5, color=BODY)),
            ("33.0%", dict(size=18, bold=True, color=INK))], space_after=5)
text(tfr2, [("リクルート　202.4万 ÷ 534万 ＝ ", dict(size=13.5, color=BODY)),
            ("37.9%", dict(size=18, bold=True, color=INK)),
            ("　→ 4.9ポイント差", dict(size=13.5, bold=True, color=WARN))],
     space_after=0)

band(s, Inches(4.62), Inches(0.90),
     [[("登録者数を増やす余地は残っている。取りに行く先は、この358万人",
        dict(size=16, bold=True, color=ACCENT))]],
     fill=ACCENT_L, bar=ACCENT)

# ================================================= スライド 15（セクション扉②）
section_slide("02", "差分の要因",
              "なぜ差がついているのか ── 構造と、いま挙がっている仮説")

# ================================================================ スライド 16
s = slide_new("差分の要因 ── 足りないのは「就職者数」",
              kicker="② 差分の要因 ─ 構造",
              note="「登録者数を伸ばす」「登録者を就職に転換する」それぞれの課題を、"
                   "次のページでご説明します。")

band(s, TOP, Inches(0.66),
     [[("差分の要因　", dict(size=11, bold=True, color=WARN)),
       ("就職者数が足りていないこと", dict(size=17, bold=True, color=INK))]],
     fill=WARN_L, bar=WARN)

tff = tb(s, ML, Inches(2.36), CW, Inches(0.5))
text(tff, [("就職者数 ＝ 年間登録者数 × 決定率",
            dict(size=24, bold=True, color=INK))],
     first=True, space_after=0, align=PP_ALIGN.CENTER)

# 3ステップ（①登録者数を増やす → ②就職者数を増やす → ③決定率を上げる）
STEPW, ARW = Inches(3.63), Inches(0.5)
flow = [
    ("STEP 1", "年間登録者数を\n増やす", "176万人", f"未リーチ 358万人へ", ACCENT),
    ("STEP 2", "そのうえで、\n就職者数を増やす", "60,307人",
     f"{TO_BE_PLACED} へ", ACCENT),
    ("STEP 3", "それによって、\n決定率を上げる", "3.43%", f"{TO_BE_RATE} へ", WARN),
]
for i, (badge, ttl, now, goal, ac) in enumerate(flow):
    x = ML + i * (STEPW + ARW)
    sh = rect(s, x, Inches(3.20), STEPW, Inches(1.96), fill=WHITE,
              line_color=LINE)
    rect(s, x, Inches(3.20), STEPW, Inches(0.05), fill=ac)
    tfc = sh.text_frame
    tfc.margin_left = Inches(0.24)
    tfc.margin_top = Inches(0.20)
    text(tfc, [(badge, dict(size=10, bold=True, color=ac))], first=True,
         space_after=4)
    text(tfc, [(ttl, dict(size=16.5, bold=True, color=INK))], space_after=10,
         line=1.25)
    text(tfc, [(now, dict(size=13, color=MUTED)),
               ("　→　", dict(size=13, color=MUTED)),
               (goal, dict(size=14, bold=True, color=ac))], space_after=0)
    if i < 2:
        label(s, x + STEPW, Inches(4.02), ARW, "▶", size=15, color=LINE,
              align=PP_ALIGN.CENTER)

band(s, Inches(5.38), Inches(0.86),
     [[("年間登録者数を増やし、そのうえで就職者数を増やす。"
        "それによって決定率を向上させる",
        dict(size=15.5, bold=True, color=ACCENT))]],
     fill=ACCENT_L, bar=ACCENT)

# ================================================================ スライド 17
s = slide_new("それぞれの差分に対する課題",
              kicker="② 差分の要因 ─ 課題",
              note="記載は 9/4 対面MTGでの議論を整理したものです。")

ISSUES2 = [
    (ML, ACCENT, "登録者数を伸ばすときの課題", [
        ("転職希望者の67%（約358万人）が doda と接点を持っていない", 2,
         "＝ 転職しない期間の接点を持てていない"),
        ("総合型でバランスがいい ＝ ポジショニング認知が足りない", 2, None),
    ]),
    (ML + Inches(6.07), WARN, "登録者を就職に転換するときの課題", [
        ("応募 → 書類 → 面接 → 内定の通過率に差があるのか", 2, None),
        ("内定承諾までの意思決定を支える支援の差", 1, None),
        ("CAの専門性の差", 1, None),
    ]),
]
COLW = Inches(5.82)
for x, ac, head, rows in ISSUES2:
    rect(s, x, TOP, COLW, Inches(0.48), fill=ac)
    label(s, x + Inches(0.24), TOP + Inches(0.13), COLW - Inches(0.44), head,
          size=14, color=WHITE)
    yy = TOP + Inches(0.78)
    for main, nl, sub in rows:
        rect(s, x + Inches(0.08), yy + Inches(0.10), Inches(0.09), Inches(0.09),
             fill=ac, shape=MSO_SHAPE.OVAL)
        tfi = tb(s, x + Inches(0.42), yy - Inches(0.02), COLW - Inches(0.54),
                 Inches(0.34 * nl))
        text(tfi, [(main, dict(size=15, bold=True, color=INK))], first=True,
             space_after=0, line=1.30)
        yy += Inches(0.34 * nl + 0.06)
        if sub:
            label(s, x + Inches(0.42), yy, COLW - Inches(0.54), sub, size=12,
                  bold=False, color=ac, h=Inches(0.3))
            yy += Inches(0.34)
        yy += Inches(0.26)

band(s, Inches(4.96), Inches(0.78),
     [[("検証状況　", dict(size=11, bold=True, color=WARN)),
       ("工程別の歩留まりは両社とも非開示のため、公開情報では特定できません。",
        dict(size=13.5, bold=True, color=INK))]],
     fill=WARN_L, bar=WARN)

# ================================================= スライド 18（セクション扉③）
section_slide("03", "課題",
              "いま我々が課題と捉えているもの ── まだ発散段階です")

# ================================================================ スライド 19
s = slide_new("いま挙がっている課題は5つ ── まだ発散段階です",
              kicker="③ 課題（発散）",
              note="課題3・4・5は、私たちが置いた定量定義と接続していません。"
                   "また課題5だけ粒度が違い（HiProという1サービスの話）、"
                   "他と並べる階層ではない可能性があります。")

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
    label(s, ML + CW - Inches(2.55), yy + Inches(0.32), Inches(2.3),
          "定量定義と接続する" if linked else "定量定義の外側", size=10.5,
          bold=False, color=(ACCENT if linked else MUTED), align=PP_ALIGN.RIGHT)
    yy += hh + Inches(0.16)

# ================================================================ スライド 20
s = slide_new("この先の進め方", kicker="④ クロージング")

steps = [
    ("STEP 1", "本日 〜 次回の中間FBまで",
     ["現場社員の方へのヒアリングと、メンターからのフィードバックをもとに議論する",
      "P.17 の課題を検証し、捉え方を詰めたうえで、解決策の方向性を導き出す",
      "2030年のあるべき決定率（本日は未算出）を確定させる"], ACCENT),
    ("STEP 2", "次回の中間FB",
     ["解決策を含めた全体をレビューいただく"], MUTED),
    ("STEP 3", "最終発表",
     ["いただいたご指摘をまとめ、①前提／②リサーチ／③課題／④解決策／"
      "⑤1か月の行動として発表する"], MUTED),
]
yy = TOP + Inches(0.10)
for badge, ttl, lines, c in steps:
    hh = Inches(0.86 + 0.32 * len(lines) + 0.16)
    rect(s, ML, yy, CW, hh, fill=(ACCENT_L if c == ACCENT else PANEL))
    rect(s, ML, yy, Inches(0.06), hh, fill=c)
    label(s, ML + Inches(0.34), yy + Inches(0.16), Inches(2), badge, size=10.5,
          color=c)
    label(s, ML + Inches(0.34), yy + Inches(0.42), Inches(6), ttl, size=16,
          color=INK, h=Inches(0.36))
    for j, ln in enumerate(lines):
        yl = yy + Inches(0.86 + j * 0.32)
        rect(s, ML + Inches(0.38), yl + Inches(0.11), Inches(0.08), Inches(0.08),
             fill=c, shape=MSO_SHAPE.OVAL)
        label(s, ML + Inches(0.66), yl, CW - Inches(1.0), ln, size=13,
              bold=False, color=BODY, h=Inches(0.3))
    yy += hh + Inches(0.22)

# ================================================================ スライド 21
s = slide_new("本日いただきたいフィードバック（再掲）", kicker="④ クロージング",
              note="この2点について、ご指摘をいただけますと幸いです。")
fb_body(s)

# ============================================================ Appendix アジェンダ
s = slide_new("Appendix ─ アジェンダ", kicker="APPENDIX")
label(s, ML, TOP, CW, "以降は、ご質問をいただいた際に参照する資料です。", size=13,
      bold=False, color=MUTED)

apx = [
    ("1", "出典一覧", "本編で使った主要数値の一次ソース・許可番号・取得日"),
    ("2", "1か月間の活動記録", "8/19・8/24・8/29・9/4・9/7 と、各回で決めたこと"),
    ("3", "リサーチしたファクト", "前提で使っていない数値／要因ではないと確認できたもの"),
    ("4", "パーソルの強み・弱みと外部環境", "SWOTの強み／弱み、PEST"),
]
yy = Inches(2.14)
for no, ttl, sub in apx:
    rect(s, ML, yy, CW, Inches(0.94), fill=WHITE, line_color=LINE)
    rect(s, ML, yy, Inches(0.05), Inches(0.94), fill=ACCENT)
    rect(s, ML + Inches(0.32), yy + Inches(0.22), Inches(0.5), Inches(0.5),
         fill=ACCENT_L)
    label(s, ML + Inches(0.32), yy + Inches(0.34), Inches(0.5), no, size=13,
          color=ACCENT, align=PP_ALIGN.CENTER)
    label(s, ML + Inches(1.06), yy + Inches(0.20), Inches(6.0), ttl, size=16)
    label(s, ML + Inches(1.06), yy + Inches(0.52), Inches(10.0), sub, size=11.5,
          bold=False, color=MUTED)
    yy += Inches(1.06)

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
    ("転職希望者534万人・67%（358万人）がdodaと接点なし",
     "PERSOL IR-DAY 2024（Career SBU）p.14／パーソルキャリア 2022年市場調査"),
    ("転職実現に至るのは1〜2割程度",
     "厚生労働省 令和4年版 労働経済の分析 第Ⅱ部第3章"),
    ("企業の約55%が「最適な人員配置」に課題",
     "パーソル総合研究所「人事部大研究」"),
    ("HiPro 登録者10万名突破（+29.1%）", "パーソルキャリア プレスリリース（2025年10月）"),
]
yy = TOP
for k, v in src:
    label(s, ML, yy, Inches(4.9), k, size=11.5, color=INK)
    label(s, ML + Inches(5.05), yy, Inches(6.84), v, size=11, bold=False,
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

# --- Appendix 3（リサーチしたファクトのうち、前提で使っていないもの）
s = slide_new("リサーチしたファクト（前提で使っていないもの）", kicker="APPENDIX 3",
              note="出典：厚生労働省 令和4年版 労働経済の分析／パーソル総合研究所 "
                   "人事部大研究／各社IR・公式サイト。")

facts = [
    ("個人", [
        [("転職希望者のうち、実際に転職を実現するのは 1〜2割程度",
          dict(bold=True, color=INK)),
         ("（厚生労働省 令和4年版 労働経済の分析）", dict(size=11, color=MUTED))],
    ]),
    ("法人", [
        [("求人掲載数：リクルート 約80万件 vs パーソル 約30万件",
          dict(bold=True, color=INK)), ("　量では勝てない", dict(size=11.5, color=MUTED))],
        [("入社後7か月以上の 定着率 94.5%", dict(bold=True, color=INK)),
         ("　質では戦える可能性がある", dict(size=11.5, color=MUTED))],
    ]),
    ("市場", [
        [("2030年に全国で 644万人 の人手不足", dict(bold=True, color=INK))],
        [("企業の 約55% が「最適な人員配置」に課題", dict(bold=True, color=INK)),
         ("（パーソル総合研究所）", dict(size=11, color=MUTED))],
    ]),
    ("要因ではないと確認できたもの", [
        [("手数料率：パーソル31.7〜33.1% vs リクルート31.4〜33.4%",
          dict(bold=True, color=INK)), ("　→ 価格ではない", dict(size=11.5, color=MUTED))],
        [("6か月以内離職率：パーソル5.49% vs リクルート5.97%",
          dict(bold=True, color=INK)),
         ("　→ 紹介の質でもない", dict(size=11.5, color=MUTED))],
    ]),
]
y = TOP
for head, lines in facts:
    label(s, ML, y, Inches(3.2), head, size=13.5, color=ACCENT)
    rect(s, ML + Inches(3.3), y + Inches(0.12), CW - Inches(3.3), Inches(0.012),
         fill=LINE)
    y += Inches(0.38)
    for ln in lines:
        rect(s, ML + Inches(0.16), y + Inches(0.11), Inches(0.07), Inches(0.07),
             fill=ACCENT, shape=MSO_SHAPE.OVAL)
        tfl = tb(s, ML + Inches(0.44), y, CW - Inches(0.44), Inches(0.3))
        p = para(tfl, first=True, space_after=0, line=1.2)
        for s_, opt in ln:
            opt = dict(opt)
            r = p.add_run()
            r.text = s_
            _set_font(r, opt.pop("size", 13.5), **opt)
        y += Inches(0.38)
    y += Inches(0.18)

# --- Appendix 4
s = slide_new("パーソルの強み・弱みと外部環境", kicker="APPENDIX 4")
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
