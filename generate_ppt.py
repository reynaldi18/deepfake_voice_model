"""
Generate PPT Presentasi: Deteksi Deepfake Audio Bahasa Indonesia
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ─── Color Palette ────────────────────────────────────────────────────────────
C_DARK_BLUE   = RGBColor(0x0D, 0x2B, 0x4A)   # slide bg / header
C_ACCENT_BLUE = RGBColor(0x1A, 0x6F, 0xB5)   # accent / title bar
C_LIGHT_BLUE  = RGBColor(0xD9, 0xEA, 0xF7)   # table header fill
C_GREEN       = RGBColor(0x1E, 0x8A, 0x44)   # positive highlights
C_ORANGE      = RGBColor(0xE8, 0x73, 0x1A)   # warning / emphasis
C_WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
C_BLACK       = RGBColor(0x1A, 0x1A, 0x1A)
C_GRAY        = RGBColor(0x4A, 0x4A, 0x4A)
C_LIGHT_GRAY  = RGBColor(0xF0, 0xF4, 0xF8)
C_YELLOW      = RGBColor(0xFF, 0xD7, 0x00)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ─── Utility helpers ──────────────────────────────────────────────────────────

def set_bg(slide, color):
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color=None, line_color=None, line_width=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height
    )
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        if line_width:
            shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape


def add_text_box(slide, text, left, top, width, height,
                 font_size=18, bold=False, color=C_BLACK,
                 align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_paragraph(tf, text, font_size=14, bold=False, color=C_BLACK,
                  align=PP_ALIGN.LEFT, space_before=Pt(4), indent_level=0, italic=False):
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = space_before
    p.level = indent_level
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return p


def header_bar(slide, title, subtitle=None):
    """Top colored header bar with title."""
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.2), fill_color=C_DARK_BLUE)
    add_text_box(slide, title,
                 Inches(0.4), Inches(0.12), Inches(12.5), Inches(0.65),
                 font_size=28, bold=True, color=C_WHITE)
    if subtitle:
        add_text_box(slide, subtitle,
                     Inches(0.4), Inches(0.72), Inches(12.5), Inches(0.4),
                     font_size=15, color=RGBColor(0xB0, 0xC8, 0xE8), italic=True)
    # bottom accent line
    add_rect(slide, 0, Inches(1.2), SLIDE_W, Inches(0.06), fill_color=C_ACCENT_BLUE)


def footer(slide, page_num, total=19):
    add_text_box(slide,
                 f"Reynaldi Pamungkas | ITB 2026  •  {page_num}/{total}",
                 Inches(0.3), Inches(7.15), Inches(12.7), Inches(0.3),
                 font_size=9, color=RGBColor(0x88, 0x99, 0xAA), align=PP_ALIGN.RIGHT)


def add_table(slide, headers, rows, left, top, width, height,
              header_fill=C_LIGHT_BLUE, alt_fill=C_LIGHT_GRAY,
              font_size=11, header_font_size=12):
    cols = len(headers)
    tbl = slide.shapes.add_table(len(rows) + 1, cols, left, top, width, height).table

    col_w = width // cols
    for i in range(cols):
        tbl.columns[i].width = col_w

    # header row
    for ci, h in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_ACCENT_BLUE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = h
        run.font.size = Pt(header_font_size)
        run.font.bold = True
        run.font.color.rgb = C_WHITE

    for ri, row in enumerate(rows):
        fill = alt_fill if ri % 2 == 1 else C_WHITE
        for ci, val in enumerate(row):
            cell = tbl.cell(ri + 1, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(font_size)
            run.font.color.rgb = C_BLACK
    return tbl


def bullet_box(slide, items, left, top, width, height,
               font_size=14, title=None, title_color=C_ACCENT_BLUE,
               bullet_color=C_BLACK, bg=None, indent=False):
    if bg:
        add_rect(slide, left, top, width, height, fill_color=bg)

    txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.05),
                                     width - Inches(0.2), height - Inches(0.1))
    tf = txBox.text_frame
    tf.word_wrap = True

    first = True
    for item in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if isinstance(item, dict):
            # dict with 'text', optional 'bold', 'color', 'size', 'indent'
            p.level = item.get('indent', 0)
            run = p.add_run()
            run.text = item['text']
            run.font.size = Pt(item.get('size', font_size))
            run.font.bold = item.get('bold', False)
            run.font.color.rgb = item.get('color', bullet_color)
        else:
            run = p.add_run()
            run.text = str(item)
            run.font.size = Pt(font_size)
            run.font.color.rgb = bullet_color
        p.space_before = Pt(3)


# ─── Slides ───────────────────────────────────────────────────────────────────

def slide_01_title(prs):
    """Title slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_bg(slide, C_DARK_BLUE)

    # decorative top stripe
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.18), fill_color=C_ACCENT_BLUE)
    add_rect(slide, 0, Inches(0.18), SLIDE_W, Inches(0.06), fill_color=C_YELLOW)

    # central content area
    add_rect(slide, Inches(0.5), Inches(1.0), Inches(12.33), Inches(4.8),
             fill_color=RGBColor(0x11, 0x35, 0x5C))

    add_text_box(slide,
                 "Deteksi Deepfake Audio",
                 Inches(0.8), Inches(1.3), Inches(11.7), Inches(1.1),
                 font_size=40, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    add_text_box(slide,
                 "Bahasa Indonesia",
                 Inches(0.8), Inches(2.3), Inches(11.7), Inches(0.8),
                 font_size=32, bold=True, color=C_YELLOW, align=PP_ALIGN.CENTER)

    add_text_box(slide,
                 "End-to-End Pipeline: Pembuatan Dataset, Pelatihan & Evaluasi Model",
                 Inches(1.0), Inches(3.05), Inches(11.3), Inches(0.6),
                 font_size=17, color=RGBColor(0xB0, 0xC8, 0xE8), align=PP_ALIGN.CENTER)

    # divider
    add_rect(slide, Inches(4.5), Inches(3.7), Inches(4.3), Inches(0.04),
             fill_color=C_ACCENT_BLUE)

    add_text_box(slide,
                 "Reynaldi Pamungkas",
                 Inches(0.8), Inches(3.85), Inches(11.7), Inches(0.45),
                 font_size=16, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide,
                 "Institut Teknologi Bandung  |  Semester 2  |  April 2026",
                 Inches(0.8), Inches(4.25), Inches(11.7), Inches(0.4),
                 font_size=14, color=RGBColor(0x88, 0xAA, 0xCC), align=PP_ALIGN.CENTER)

    # bottom bar
    add_rect(slide, 0, Inches(7.2), SLIDE_W, Inches(0.3), fill_color=C_ACCENT_BLUE)
    add_text_box(slide,
                 "AASIST  •  MoLEx (LCNN)  •  XGBoost  |  ROC-AUC = 1.0000",
                 Inches(0.5), Inches(7.22), Inches(12.3), Inches(0.26),
                 font_size=11, color=C_WHITE, align=PP_ALIGN.CENTER)


def slide_02_agenda(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Agenda")
    footer(slide, 2)

    items_left = [
        ("1", "Latar Belakang & Motivasi"),
        ("2", "Overview Pipeline"),
        ("3", "Dataset Bona Fide"),
        ("4", "Generasi Audio Spoof"),
        ("5", "Pembagian Dataset"),
        ("6", "Arsitektur Model: AASIST"),
        ("7", "Arsitektur Model: MoLEx"),
        ("8", "Arsitektur Model: XGBoost"),
        ("9", "Hasil Training & Evaluasi"),
        ("10","Perbandingan Tiga Model"),
    ]
    items_right = [
        ("11", "Inferensi Real-World"),
        ("12", "Temuan Penting"),
        ("13", "Rekomendasi Lanjutan"),
        ("14", "Kesimpulan"),
        ("15", "Referensi"),
    ]

    def draw_items(items, left_x):
        y = Inches(1.45)
        for num, text in items:
            add_rect(slide, left_x, y, Inches(0.42), Inches(0.38), fill_color=C_ACCENT_BLUE)
            add_text_box(slide, num, left_x, y + Inches(0.03), Inches(0.42), Inches(0.35),
                         font_size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
            add_text_box(slide, text,
                         left_x + Inches(0.5), y + Inches(0.05), Inches(5.5), Inches(0.35),
                         font_size=14, color=C_BLACK)
            y += Inches(0.53)

    draw_items(items_left, Inches(0.6))
    draw_items(items_right, Inches(7.0))


def slide_03_background(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Latar Belakang & Motivasi",
               "Mengapa deteksi deepfake audio penting?")
    footer(slide, 3)

    # left box: problem
    add_rect(slide, Inches(0.4), Inches(1.45), Inches(5.9), Inches(5.5),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "Masalah",
                 Inches(0.5), Inches(1.5), Inches(5.7), Inches(0.45),
                 font_size=17, bold=True, color=C_ACCENT_BLUE)

    probs = [
        "• Teknologi TTS & Voice Cloning kian realistis",
        "• Deepfake audio digunakan untuk:\n  – Penipuan finansial (vishing)\n  – Disinformasi & hoaks\n  – Pelanggaran privasi",
        "• Belum ada sistem deteksi khusus\n  Bahasa Indonesia",
        "• Benchmark internasional (ASVspoof)\n  fokus Bahasa Inggris",
    ]
    y = Inches(2.0)
    for pr in probs:
        add_text_box(slide, pr, Inches(0.55), y, Inches(5.6), Inches(0.7),
                     font_size=12, color=C_BLACK)
        y += Inches(0.75)

    # right box: solution
    add_rect(slide, Inches(6.7), Inches(1.45), Inches(6.2), Inches(5.5),
             fill_color=RGBColor(0xE8, 0xF4, 0xE8))
    add_text_box(slide, "Solusi yang Dibangun",
                 Inches(6.8), Inches(1.5), Inches(6.0), Inches(0.45),
                 font_size=17, bold=True, color=C_GREEN)

    sols = [
        "• Dataset bona-fide Bahasa Indonesia\n  dari Mozilla Common Voice + LibriVox",
        "• 75.512 sampel spoof dari 4 TTS engine\n  Indonesia (Edge-TTS, MMS, gTTS, Kokoro)",
        "• 3 model komplementer:\n  – AASIST (raw waveform, deep learning)\n  – MoLEx/LCNN (LFCC features)\n  – XGBoost (handcrafted features)",
        "• Ensemble prediksi real-world\n  dengan confidence thresholding",
        "• Semua model: ROC-AUC = 1.0000",
    ]
    y = Inches(2.0)
    for sol in sols:
        add_text_box(slide, sol, Inches(6.85), y, Inches(5.9), Inches(0.75),
                     font_size=12, color=C_BLACK)
        y += Inches(0.78)

    # reference note
    add_text_box(slide,
                 "Ref: Wang et al. (CSL 2020) | Yi et al. (ICASSP 2022) | Bird & Lotfi (2023) | Wu et al., SEA-SPOOF (2025) | Mawalim et al., InaSAS (2025)",
                 Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.3),
                 font_size=9, italic=True, color=C_GRAY)


def slide_04_pipeline_overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Overview Pipeline", "Alur kerja end-to-end dari data mentah ke inferensi")
    footer(slide, 4)

    stages = [
        ("1\nData\nCollection", C_ACCENT_BLUE),
        ("2\nAudio\nPreprocessing", RGBColor(0x17, 0x7A, 0x9C)),
        ("3\nSpoof\nGeneration", RGBColor(0x7B, 0x2D, 0x8B)),
        ("4\nDataset\nSplit", C_GREEN),
        ("5\nModel\nTraining", C_ORANGE),
        ("6\nEvaluation\n& Inference", RGBColor(0xC0, 0x39, 0x2B)),
    ]

    box_w = Inches(1.85)
    box_h = Inches(1.4)
    start_x = Inches(0.3)
    y = Inches(2.0)
    gap = Inches(0.25)

    for i, (label, color) in enumerate(stages):
        x = start_x + i * (box_w + gap)
        add_rect(slide, x, y, box_w, box_h, fill_color=color)
        add_text_box(slide, label, x + Inches(0.05), y + Inches(0.15),
                     box_w - Inches(0.1), box_h - Inches(0.3),
                     font_size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            arrow_x = x + box_w + Inches(0.02)
            add_text_box(slide, "→", arrow_x, y + Inches(0.5), Inches(0.22), Inches(0.4),
                         font_size=20, bold=True, color=C_GRAY, align=PP_ALIGN.CENTER)

    # sub-details for each stage
    details = [
        "Common Voice\nLibriVox\n(35.891 files)",
        "16 kHz, FLAC\nDedup\nNormalisasi",
        "Edge-TTS\nMMS-VITS\ngTTS, Kokoro\n(75.512 files)",
        "Balancing 1:1\n80/10/10 split\n(71.782 total)",
        "AASIST\nMoLEx\nXGBoost",
        "ROC-AUC\nEER, MCC\nReal-World",
    ]
    y2 = Inches(3.55)
    for i, detail in enumerate(details):
        x = start_x + i * (box_w + gap)
        add_rect(slide, x, y2, box_w, Inches(1.6), fill_color=C_LIGHT_GRAY)
        add_text_box(slide, detail, x + Inches(0.05), y2 + Inches(0.08),
                     box_w - Inches(0.1), Inches(1.45),
                     font_size=11, color=C_GRAY, align=PP_ALIGN.CENTER)

    # key stats bar
    add_rect(slide, Inches(0.4), Inches(5.35), Inches(12.4), Inches(0.7),
             fill_color=C_DARK_BLUE)
    stats = "Dataset: 35.891 bona-fide + 75.512 spoof   |   Setelah balancing: 71.782 sampel   |   Perangkat: Apple MPS (PyTorch)"
    add_text_box(slide, stats, Inches(0.6), Inches(5.45), Inches(12.1), Inches(0.5),
                 font_size=13, color=C_WHITE, align=PP_ALIGN.CENTER)


def slide_05_bonafide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Dataset Bona Fide", "Mozilla Common Voice v24.0 + LibriVox")
    footer(slide, 5)

    # left: sources
    add_rect(slide, Inches(0.4), Inches(1.45), Inches(6.0), Inches(4.0),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "Sumber Data",
                 Inches(0.5), Inches(1.5), Inches(5.8), Inches(0.4),
                 font_size=16, bold=True, color=C_ACCENT_BLUE)

    src_items = [
        "Mozilla Common Voice v24.0 (Indonesian)",
        "  • 30.256 sampel validated.tsv",
        "  • Rekaman komunitas sukarela",
        "  • Berbagai aksen & umur",
        "",
        "LibriVox Bahasa Indonesia",
        "  • 5.635 rekaman audiobook publik",
        "  • Menambah variasi speaker",
        "  • Dikecualikan dari TTS spoof",
    ]
    y = Inches(2.0)
    for item in src_items:
        bold = not item.startswith(" ") and item != ""
        color = C_ACCENT_BLUE if bold and item else C_BLACK
        add_text_box(slide, item, Inches(0.55), y, Inches(5.7), Inches(0.38),
                     font_size=12, bold=bold, color=color)
        y += Inches(0.33)

    # right: preprocessing steps
    add_rect(slide, Inches(6.8), Inches(1.45), Inches(6.1), Inches(4.0),
             fill_color=RGBColor(0xE8, 0xF4, 0xE8))
    add_text_box(slide, "Proses Preprocessing",
                 Inches(6.9), Inches(1.5), Inches(5.9), Inches(0.4),
                 font_size=16, bold=True, color=C_GREEN)

    steps = [
        "1. Baca TSV: validated, train, test",
        "2. Deduplikasi (tuple file_id + filename)",
        "3. Gabung rekaman LibriVox",
        "4. Load via librosa (sr=16 kHz, mono)",
        "5. Simpan ulang → FLAC (PCM_16)",
        "6. Tulis metadata TSV",
    ]
    y = Inches(2.0)
    for step in steps:
        add_text_box(slide, step, Inches(6.95), y, Inches(5.8), Inches(0.38),
                     font_size=12, color=C_BLACK)
        y += Inches(0.42)

    # table
    headers = ["Sumber", "Records", "Duplikat"]
    rows = [
        ["Common Voice (validated)", "30.256", "0"],
        ["train.tsv", "4.973", "4.973 (semua duplikat)"],
        ["test.tsv", "3.691", "3.691 (semua duplikat)"],
        ["LibriVox", "5.635", "0"],
        ["TOTAL UNIK", "35.891", "—"],
    ]
    add_table(slide, headers, rows,
              Inches(0.4), Inches(5.6), Inches(12.5), Inches(1.55))

    add_text_box(slide,
                 "Ref: Ardila et al., Common Voice: A Massively-Multilingual Speech Corpus, LREC 2020",
                 Inches(0.4), Inches(7.1), Inches(12.5), Inches(0.28),
                 font_size=9, italic=True, color=C_GRAY)


def slide_06_spoof(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Generasi Audio Spoof", "4 TTS Engine → 75.512 sampel sintetis")
    footer(slide, 6)

    engines = [
        {
            "name": "Edge-TTS (Microsoft)",
            "color": RGBColor(0x00, 0x78, 0xD4),
            "details": [
                "Voices: id-ID-GadisNeural (♀)",
                "        id-ID-ArdiNeural (♂)",
                "2 suara × 30.256 kalimat",
                "Async concurrency: 5",
                "Output: 60.512 files ✓",
            ],
            "count": "60.512",
        },
        {
            "name": "MMS-VITS (Meta)",
            "color": RGBColor(0x10, 0x68, 0x99),
            "details": [
                "Model: facebook/mms-tts-ind",
                "Target: 5.000 sampel",
                "Resume-friendly queue",
                "Output: 5.000/5.000 ✓",
                "",
            ],
            "count": "5.000",
        },
        {
            "name": "gTTS (Google)",
            "color": RGBColor(0x0F, 0x9D, 0x58),
            "details": [
                "TLD rotasi: com / co.id / com.au",
                "Request delay: 0.3 detik",
                "Target: 5.000 sampel",
                "Output: 5.000/5.000 ✓",
                "",
            ],
            "count": "5.000",
        },
        {
            "name": "Kokoro-82M",
            "color": RGBColor(0x8E, 0x44, 0xAD),
            "details": [
                "Phoneme: espeak-ng (en)",
                "Voices: af_heart, af_bella,",
                "        am_adam, am_michael",
                "Target: 5.000 sampel",
                "Output: 5.000/5.000 ✓",
            ],
            "count": "5.000",
        },
    ]

    box_w = Inches(3.1)
    box_h = Inches(3.5)
    gap = Inches(0.18)
    start_x = Inches(0.3)
    y = Inches(1.45)

    for i, eng in enumerate(engines):
        x = start_x + i * (box_w + gap)
        add_rect(slide, x, y, box_w, Inches(0.5), fill_color=eng["color"])
        add_text_box(slide, eng["name"], x + Inches(0.05), y + Inches(0.08),
                     box_w - Inches(0.1), Inches(0.38),
                     font_size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_rect(slide, x, y + Inches(0.5), box_w, box_h - Inches(0.5),
                 fill_color=C_LIGHT_GRAY)
        dy = y + Inches(0.6)
        for detail in eng["details"]:
            add_text_box(slide, detail, x + Inches(0.12), dy,
                         box_w - Inches(0.18), Inches(0.4),
                         font_size=11, color=C_BLACK)
            dy += Inches(0.38)

        # count badge
        add_rect(slide, x + box_w - Inches(1.1), y + box_h - Inches(0.45),
                 Inches(1.0), Inches(0.38), fill_color=eng["color"])
        add_text_box(slide, eng["count"],
                     x + box_w - Inches(1.1), y + box_h - Inches(0.44),
                     Inches(1.0), Inches(0.36),
                     font_size=12, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    # summary table
    headers = ["Engine", "Sampel", "Bahasa ID Native", "Keterangan"]
    rows = [
        ["Edge-TTS", "60.512", "Ya", "Neural TTS Microsoft"],
        ["MMS-VITS", "5.000", "Ya", "Meta Massively Multilingual"],
        ["gTTS", "5.000", "Ya", "Google TTS API"],
        ["Kokoro-82M", "5.000", "Parsial", "Phoneme English (espeak-ng)"],
        ["TOTAL", "75.512", "—", "—"],
    ]
    add_table(slide, headers, rows,
              Inches(0.3), Inches(5.15), Inches(12.7), Inches(1.85),
              font_size=11)

    add_text_box(slide,
                 "Ref: Pratap et al., Scaling Speech Technology to 1,000+ Languages (MMS), Meta AI 2023",
                 Inches(0.3), Inches(7.1), Inches(12.7), Inches(0.28),
                 font_size=9, italic=True, color=C_GRAY)


def slide_07_dataset_split(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Pembagian Dataset", "Balancing 1:1 → Split 80/10/10")
    footer(slide, 7)

    # flow diagram
    steps = [
        ("Bona Fide\n35.891", C_GREEN),
        ("Spoof\n75.512", RGBColor(0xC0, 0x39, 0x2B)),
        ("Balancing\n1:1\n→ 35.891 tiap kelas", C_ACCENT_BLUE),
        ("Shuffle\nseed=42", RGBColor(0x7B, 0x2D, 0x8B)),
        ("Split\n80/10/10", C_ORANGE),
    ]
    box_w = Inches(2.1)
    box_h = Inches(1.3)
    gap = Inches(0.27)
    sx = Inches(0.3)
    y = Inches(1.5)
    for i, (label, color) in enumerate(steps):
        x = sx + i * (box_w + gap)
        add_rect(slide, x, y, box_w, box_h, fill_color=color)
        add_text_box(slide, label, x + Inches(0.05), y + Inches(0.1),
                     box_w - Inches(0.1), box_h - Inches(0.2),
                     font_size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            ax = x + box_w + Inches(0.03)
            add_text_box(slide, "→", ax, y + Inches(0.4), Inches(0.22), Inches(0.4),
                         font_size=18, bold=True, color=C_GRAY, align=PP_ALIGN.CENTER)

    # split result table
    headers = ["Split", "Total Sampel", "Bona Fide", "Spoof", "Proporsi"]
    rows = [
        ["Train", "57.425", "28.687", "28.738", "80%"],
        ["Val", "7.178", "~3.590", "~3.588", "10%"],
        ["Test", "7.179", "3.621", "3.558", "10%"],
        ["TOTAL", "71.782", "35.891", "35.891", "100%"],
    ]
    add_table(slide, headers, rows,
              Inches(0.4), Inches(3.0), Inches(12.5), Inches(1.8))

    # key points
    add_rect(slide, Inches(0.4), Inches(5.0), Inches(12.5), Inches(1.8),
             fill_color=RGBColor(0xFF, 0xF3, 0xCD))
    points = [
        "Catatan Penting:",
        "• Balancing dilakukan dengan subsample spoof → mencegah bias ke kelas mayoritas",
        "• 5.635 rekaman LibriVox dikecualikan dari generasi TTS (hanya Common Voice yang di-TTS)",
        "• Random seed=42 untuk reproducibility",
        "• Distribusi kelas hampir sempurna 50/50 di semua split",
    ]
    y2 = Inches(5.1)
    for i, pt in enumerate(points):
        bold = i == 0
        color = C_ORANGE if bold else C_BLACK
        add_text_box(slide, pt, Inches(0.6), y2, Inches(12.1), Inches(0.35),
                     font_size=12 if not bold else 13, bold=bold, color=color)
        y2 += Inches(0.32)


def slide_08_aasist_arch(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Arsitektur Model: AASIST",
               "Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks")
    footer(slide, 8)

    # architecture flow
    blocks = [
        ("Input\nRaw Waveform\n64.000 sampel\n4 detik @ 16 kHz", C_DARK_BLUE),
        ("SincConv\nLearnable sinc\nfilterbank\n(front-end)", C_ACCENT_BLUE),
        ("ResBlock\nEncoder\nResiduual conv\nblocsk", RGBColor(0x17, 0x7A, 0x9C)),
        ("Adaptive\nAvgPool\nTemporal\naggregation", RGBColor(0x7B, 0x2D, 0x8B)),
        ("Classifier\nFC + Dropout\n→ 2 kelas\n(spoof/bonafide)", C_GREEN),
    ]
    bw = Inches(2.2)
    bh = Inches(2.0)
    gap = Inches(0.2)
    sx = Inches(0.3)
    y = Inches(1.5)
    for i, (label, color) in enumerate(blocks):
        x = sx + i * (bw + gap)
        add_rect(slide, x, y, bw, bh, fill_color=color)
        add_text_box(slide, label, x + Inches(0.05), y + Inches(0.15),
                     bw - Inches(0.1), bh - Inches(0.3),
                     font_size=12, bold=False, color=C_WHITE, align=PP_ALIGN.CENTER)
        if i < len(blocks) - 1:
            ax = x + bw + Inches(0.01)
            add_text_box(slide, "→", ax, y + Inches(0.75), Inches(0.18), Inches(0.4),
                         font_size=16, bold=True, color=C_GRAY)

    # specs
    specs_left = [
        ("Parameter Total", "2.508.174"),
        ("Batch Size", "24"),
        ("Optimizer", "AdamW"),
        ("Learning Rate", "1 × 10⁻⁴"),
        ("Weight Decay", "1 × 10⁻⁴"),
    ]
    specs_right = [
        ("Epochs (max)", "20"),
        ("Scheduler", "CosineAnnealingLR"),
        ("Early Stopping", "patience = 5"),
        ("Best Epoch", "6 (val acc 99.86%)"),
        ("Device", "Apple MPS"),
    ]

    y_spec = Inches(3.75)
    add_rect(slide, Inches(0.4), y_spec, Inches(5.9), Inches(2.8),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "Spesifikasi Training",
                 Inches(0.5), y_spec + Inches(0.05), Inches(5.7), Inches(0.38),
                 font_size=14, bold=True, color=C_ACCENT_BLUE)
    y2 = y_spec + Inches(0.48)
    for k, v in specs_left:
        add_text_box(slide, f"• {k}: {v}", Inches(0.55), y2, Inches(5.6), Inches(0.38),
                     font_size=12, color=C_BLACK)
        y2 += Inches(0.42)

    add_rect(slide, Inches(6.7), y_spec, Inches(6.2), Inches(2.8),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "Hyperparameter",
                 Inches(6.8), y_spec + Inches(0.05), Inches(6.0), Inches(0.38),
                 font_size=14, bold=True, color=C_ACCENT_BLUE)
    y2 = y_spec + Inches(0.48)
    for k, v in specs_right:
        add_text_box(slide, f"• {k}: {v}", Inches(6.85), y2, Inches(5.9), Inches(0.38),
                     font_size=12, color=C_BLACK)
        y2 += Inches(0.42)

    add_text_box(slide,
                 "Ref: Jung et al., AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks, ICASSP 2022",
                 Inches(0.3), Inches(7.1), Inches(12.7), Inches(0.28),
                 font_size=9, italic=True, color=C_GRAY)


def slide_09_molex_arch(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Arsitektur Model: MoLEx (LCNN)",
               "Light CNN + Attention Pooling dengan fitur LFCC")
    footer(slide, 9)

    # architecture flow
    blocks = [
        ("Input Audio\n16 kHz FLAC\n≤ 4 detik", C_DARK_BLUE),
        ("LFCC\nExtraction\n60 koef.\n401 frame", C_ACCENT_BLUE),
        ("LCNN\nBlocks\nLight CNN\n(max-feature-map)", RGBColor(0x17, 0x7A, 0x9C)),
        ("Attention\nPooling\nWeighted\ntemporal agg.", RGBColor(0x7B, 0x2D, 0x8B)),
        ("Classifier\nFC Layer\n→ 2 kelas", C_GREEN),
    ]
    bw = Inches(2.2)
    bh = Inches(2.0)
    gap = Inches(0.2)
    sx = Inches(0.3)
    y = Inches(1.5)
    for i, (label, color) in enumerate(blocks):
        x = sx + i * (bw + gap)
        add_rect(slide, x, y, bw, bh, fill_color=color)
        add_text_box(slide, label, x + Inches(0.05), y + Inches(0.2),
                     bw - Inches(0.1), bh - Inches(0.3),
                     font_size=12, color=C_WHITE, align=PP_ALIGN.CENTER)
        if i < len(blocks) - 1:
            ax = x + bw + Inches(0.01)
            add_text_box(slide, "→", ax, y + Inches(0.75), Inches(0.18), Inches(0.4),
                         font_size=16, bold=True, color=C_GRAY)

    # LFCC params + model specs
    lfcc_params = [
        ("n_lfcc", "60"),
        ("n_filter", "70"),
        ("n_fft", "512"),
        ("win_length", "320 sampel (20 ms)"),
        ("hop_length", "160 sampel (10 ms)"),
    ]
    model_specs = [
        ("Parameter Total", "179.491"),
        ("vs AASIST", "14× lebih ringan"),
        ("Batch Size", "32"),
        ("Best Epoch", "8 (val acc 100.00%)"),
        ("Early Stopping", "epoch 13"),
    ]

    y_spec = Inches(3.75)
    add_rect(slide, Inches(0.4), y_spec, Inches(5.9), Inches(2.8),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "Parameter LFCC",
                 Inches(0.5), y_spec + Inches(0.05), Inches(5.7), Inches(0.38),
                 font_size=14, bold=True, color=C_ACCENT_BLUE)
    y2 = y_spec + Inches(0.48)
    for k, v in lfcc_params:
        add_text_box(slide, f"• {k} = {v}", Inches(0.55), y2, Inches(5.6), Inches(0.38),
                     font_size=12, color=C_BLACK)
        y2 += Inches(0.42)

    add_rect(slide, Inches(6.7), y_spec, Inches(6.2), Inches(2.8),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "Spesifikasi Model",
                 Inches(6.8), y_spec + Inches(0.05), Inches(6.0), Inches(0.38),
                 font_size=14, bold=True, color=C_ACCENT_BLUE)
    y2 = y_spec + Inches(0.48)
    for k, v in model_specs:
        add_text_box(slide, f"• {k}: {v}", Inches(6.85), y2, Inches(5.9), Inches(0.38),
                     font_size=12, color=C_BLACK)
        y2 += Inches(0.42)

    add_text_box(slide,
                 "Ref: Sahidullah et al., A Comparison of Features for Synthetic Speech Detection, Interspeech 2015 | "
                 "Wu et al., Light CNN for Deep Face Representation with Noisy Labels, IEEE T-IFS 2015",
                 Inches(0.3), Inches(7.05), Inches(12.7), Inches(0.35),
                 font_size=9, italic=True, color=C_GRAY)


def slide_10_xgboost(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Model XGBoost: Fitur Akustik",
               "Handcrafted features — cepat & interpretable")
    footer(slide, 10)

    # feature groups
    add_rect(slide, Inches(0.4), Inches(1.45), Inches(6.0), Inches(3.2),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "26 Fitur Input",
                 Inches(0.5), Inches(1.5), Inches(5.8), Inches(0.4),
                 font_size=16, bold=True, color=C_ACCENT_BLUE)

    groups = [
        ("Fitur Akustik (6)", [
            "Chroma, RMS, Spectral Centroid",
            "Spectral Bandwidth, Spectral Rolloff",
            "Zero Crossing Rate (ZCR)",
        ]),
        ("MFCC (20)", [
            "MFCC_1 s/d MFCC_20",
            "Rata-rata per segmen 1 detik",
            "Standardisasi: StandardScaler",
        ]),
    ]
    y2 = Inches(2.0)
    for group_name, items in groups:
        add_text_box(slide, group_name, Inches(0.55), y2, Inches(5.7), Inches(0.38),
                     font_size=13, bold=True, color=C_ORANGE)
        y2 += Inches(0.35)
        for item in items:
            add_text_box(slide, f"  • {item}", Inches(0.55), y2, Inches(5.7), Inches(0.35),
                         font_size=12, color=C_BLACK)
            y2 += Inches(0.3)
        y2 += Inches(0.1)

    # top 10 features table
    headers = ["Rank", "Fitur", "Gain (XGBoost)"]
    rows = [
        ["1", "MFCC_7", "0.2520"],
        ["2", "RMS", "0.0984"],
        ["3", "MFCC_16", "0.0733"],
        ["4", "MFCC_5", "0.0724"],
        ["5", "MFCC_10", "0.0692"],
        ["6", "Rolloff", "0.0566"],
        ["7", "MFCC_13", "0.0461"],
        ["8", "MFCC_2", "0.0400"],
        ["9", "MFCC_17", "0.0372"],
        ["10", "MFCC_15", "0.0347"],
    ]
    add_table(slide, headers, rows,
              Inches(6.7), Inches(1.45), Inches(6.2), Inches(4.1),
              font_size=11)

    # xgboost hyperparams
    add_rect(slide, Inches(0.4), Inches(4.75), Inches(6.0), Inches(2.2),
             fill_color=RGBColor(0xFF, 0xF3, 0xCD))
    add_text_box(slide, "Hyperparameter XGBoost",
                 Inches(0.5), Inches(4.8), Inches(5.8), Inches(0.38),
                 font_size=13, bold=True, color=C_ORANGE)
    xgb_params = [
        "max_depth=6  |  learning_rate=0.1",
        "subsample=0.8  |  colsample_bytree=0.8",
        "Boosting rounds optimal: 313 (early stopping)",
        "10-Fold Cross-Validation",
    ]
    y3 = Inches(5.25)
    for p in xgb_params:
        add_text_box(slide, p, Inches(0.55), y3, Inches(5.7), Inches(0.35),
                     font_size=11, color=C_BLACK)
        y3 += Inches(0.3)

    add_text_box(slide,
                 "Ref: Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, KDD 2016 | "
                 "Davis & Mermelstein, Comparison of Parametric Representations for MFCC, IEEE T-ASSP 1980",
                 Inches(0.3), Inches(7.05), Inches(12.7), Inches(0.35),
                 font_size=9, italic=True, color=C_GRAY)


def slide_11_training_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Hasil Training & Evaluasi",
               "Progres training ketiga model")
    footer(slide, 11)

    # AASIST training table (condensed)
    add_text_box(slide, "AASIST — Progres Training (selected epochs)",
                 Inches(0.4), Inches(1.45), Inches(5.9), Inches(0.38),
                 font_size=13, bold=True, color=C_ACCENT_BLUE)
    headers1 = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
    rows1 = [
        ["1", "0.0942", "96.61%", "2.5823", "68.51%"],
        ["4", "0.0075", "99.80%", "1.1331", "80.61%"],
        ["6 ★", "0.0043", "99.88%", "0.0062", "99.86% ← best"],
        ["8", "0.0023", "99.94%", "43.597", "53.13%"],
        ["11", "0.0012", "99.96%", "0.0308", "99.40%"],
        ["(stop)", "—", "—", "—", "early stop"],
    ]
    add_table(slide, headers1, rows1,
              Inches(0.4), Inches(1.9), Inches(6.2), Inches(2.15),
              font_size=10, header_font_size=11)

    # MoLEx training table (condensed)
    add_text_box(slide, "MoLEx — Progres Training (selected epochs)",
                 Inches(7.0), Inches(1.45), Inches(6.0), Inches(0.38),
                 font_size=13, bold=True, color=C_GREEN)
    headers2 = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
    rows2 = [
        ["1", "0.0469", "98.40%", "0.0159", "99.62%"],
        ["3", "0.0065", "99.86%", "0.0024", "99.93%"],
        ["6", "0.0024", "99.95%", "0.0001", "99.99%"],
        ["8 ★", "0.0004", "99.99%", "0.0001", "100.00% ← best"],
        ["9", "0.0000", "100.00%", "0.0000", "100.00%"],
        ["(stop)", "—", "—", "—", "early stop ep.13"],
    ]
    add_table(slide, headers2, rows2,
              Inches(7.0), Inches(1.9), Inches(6.0), Inches(2.15),
              font_size=10, header_font_size=11)

    # Test results
    add_text_box(slide, "Hasil Test Set",
                 Inches(0.4), Inches(4.2), Inches(12.5), Inches(0.38),
                 font_size=15, bold=True, color=C_DARK_BLUE)
    headers3 = ["Model", "Precision", "Recall", "F1-Score", "ROC-AUC", "EER", "MCC"]
    rows3 = [
        ["AASIST", "1.00", "1.00", "1.00", "1.0000", "0.00%", "0.9958"],
        ["MoLEx", "1.00", "1.00", "1.00", "1.0000", "0.03%", "0.9992"],
        ["XGBoost", "1.00", "1.00", "1.00", "1.0000", "0.10%", "0.9981"],
    ]
    add_table(slide, headers3, rows3,
              Inches(0.4), Inches(4.65), Inches(12.5), Inches(1.5))

    # note
    add_text_box(slide,
                 "★ = Best checkpoint  |  Semua model: test accuracy ≥ 99.79%, ROC-AUC = 1.0000",
                 Inches(0.4), Inches(6.35), Inches(12.5), Inches(0.3),
                 font_size=11, italic=True, color=C_GRAY)


def slide_12_comparison(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Perbandingan Tiga Model",
               "AASIST vs MoLEx vs XGBoost — trade-off performa & efisiensi")
    footer(slide, 12)

    # main comparison table
    headers = ["Metrik", "AASIST", "MoLEx", "XGBoost"]
    rows = [
        ["Accuracy", "99.79%", "99.96% ★", "99.90%"],
        ["ROC-AUC", "1.0000", "1.0000", "1.0000"],
        ["EER", "0.00% ★", "0.03%", "0.10%"],
        ["MCC", "0.9958", "0.9992 ★", "0.9981"],
        ["Inf. (ms/sampel)", "4.5964", "3.2693", "0.0026 ★"],
        ["# Parameter", "2.508.174", "179.491", "N/A (tree)"],
        ["Jenis Input", "Raw waveform", "LFCC features", "MFCC + akustik"],
        ["Training stability", "Volatile val loss", "Stabil ★", "CV stabil ★"],
        ["Batch Inference", "✓ (sel 7.3)", "✓ (sel 7.3)", "✓ (sel 9.6b) ← baru"],
    ]
    add_table(slide, headers, rows,
              Inches(0.4), Inches(1.5), Inches(12.5), Inches(3.4),
              font_size=11, header_font_size=12)

    # insights
    insights = [
        ("MoLEx", C_GREEN, "Accuracy & MCC terbaik, parameter 14× lebih sedikit dari AASIST, training stabil"),
        ("AASIST", C_ACCENT_BLUE, "EER terbaik (0.00%), cocok untuk aplikasi high-security, input langsung raw audio"),
        ("XGBoost", C_ORANGE, "Inferensi 1.770× lebih cepat + kini batch inference independen (sel 9.6b), ideal edge/real-time"),
    ]
    y2 = Inches(5.15)
    for model, color, insight in insights:
        add_rect(slide, Inches(0.4), y2, Inches(2.0), Inches(0.5), fill_color=color)
        add_text_box(slide, model, Inches(0.4), y2 + Inches(0.08),
                     Inches(2.0), Inches(0.38),
                     font_size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_rect(slide, Inches(2.4), y2, Inches(10.5), Inches(0.5), fill_color=C_LIGHT_GRAY)
        add_text_box(slide, insight, Inches(2.5), y2 + Inches(0.1),
                     Inches(10.3), Inches(0.38),
                     font_size=12, color=C_BLACK)
        y2 += Inches(0.6)

    add_text_box(slide,
                 "★ = Best dalam kategori tersebut  |  Ensemble AASIST+MoLEx: keputusan akhir  |  XGBoost: batch inference independen (sel 9.6b)",
                 Inches(0.4), Inches(7.1), Inches(12.5), Inches(0.28),
                 font_size=9, italic=True, color=C_GRAY)


def slide_13_realworld(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Inferensi Real-World",
               "Ensemble AASIST + MoLEx (Bagian 9) · XGBoost independen (sel 9.6b)")
    footer(slide, 13)

    # setup
    add_rect(slide, Inches(0.4), Inches(1.45), Inches(3.5), Inches(1.5),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "Setup Ensemble",
                 Inches(0.5), Inches(1.5), Inches(3.3), Inches(0.38),
                 font_size=13, bold=True, color=C_ACCENT_BLUE)
    setup_items = [
        "• Threshold Low = 0.3",
        "• Threshold High = 0.7",
        "• Jika AASIST & MoLEx beda → TIDAK YAKIN",
        "• Format: .flac .wav .mp3 .m4a .ogg",
        "• XGBoost: batch independen (sel 9.6b)",
    ]
    y2 = Inches(1.9)
    for s in setup_items:
        add_text_box(slide, s, Inches(0.55), y2, Inches(3.25), Inches(0.33),
                     font_size=11, color=C_BLACK)
        y2 += Inches(0.3)

    # sample results table
    headers = ["File", "AASIST", "MoLEx", "Ensemble"]
    rows = [
        ["fake_ardi_ttsfree.mp3", "SPOOF 0.9999", "SPOOF 1.0000", "SPOOF ✓"],
        ["fake_ttsfree_standard-b.mp3", "BONAFIDE 0.0016", "BONAFIDE 0.0253", "BONAFIDE ✗"],
        ["fake_guru_rani_minta_duit.mp3", "SPOOF 0.9956", "BONAFIDE 0.0000", "TIDAK YAKIN"],
        ["real_CNN.mp3", "BONAFIDE 0.0000", "BONAFIDE 0.0000", "BONAFIDE ✓"],
        ["real_Pak_Jokowi.mp3", "BONAFIDE 0.0000", "BONAFIDE 0.0000", "BONAFIDE ✓"],
        ["sample_dewi_putri.mp3", "SPOOF 1.0000", "SPOOF 1.0000", "SPOOF ✓"],
        ["sample_budi_utomo.mp3", "SPOOF 0.9886", "BONAFIDE 0.1563", "TIDAK YAKIN"],
    ]
    add_table(slide, headers, rows,
              Inches(4.1), Inches(1.45), Inches(8.9), Inches(2.6),
              font_size=10, header_font_size=11)

    # summary donut-like
    add_rect(slide, Inches(0.4), Inches(3.1), Inches(3.5), Inches(1.7),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "Ringkasan Batch (25 file)",
                 Inches(0.5), Inches(3.15), Inches(3.3), Inches(0.38),
                 font_size=13, bold=True, color=C_ACCENT_BLUE)
    summary = [
        ("BONAFIDE", "19", "76.0%", C_GREEN),
        ("SPOOF", "3", "12.0%", RGBColor(0xC0, 0x39, 0x2B)),
        ("TIDAK YAKIN", "3", "12.0%", C_ORANGE),
    ]
    y3 = Inches(3.6)
    for label, n, pct, color in summary:
        add_rect(slide, Inches(0.55), y3, Inches(0.2), Inches(0.28), fill_color=color)
        add_text_box(slide, f"{label}: {n} ({pct})",
                     Inches(0.82), y3, Inches(2.9), Inches(0.3),
                     font_size=12, color=C_BLACK)
        y3 += Inches(0.38)

    # key observations
    add_rect(slide, Inches(0.4), Inches(4.95), Inches(12.5), Inches(2.45),
             fill_color=RGBColor(0xFF, 0xF3, 0xCD))
    add_text_box(slide, "Temuan Kritis dari Real-World Test:",
                 Inches(0.55), Inches(5.0), Inches(12.1), Inches(0.38),
                 font_size=13, bold=True, color=C_ORANGE)
    obs = [
        "• fake_ttsfree_standard-b/c: model GAGAL mendeteksi → TTSFree menggunakan model TTS yang lebih realistis",
        "• 3 kasus TIDAK YAKIN: AASIST & MoLEx tidak agree → perlu investigasi lebih lanjut",
        "• Semua audio nyata (real_*): terklasifikasi BONAFIDE dengan benar (precision sempurna)",
        "• XGBoost kini mendukung batch folder inference independen (sel 9.6b) — ~0.006 ms/file, CPU only",
    ]
    y4 = Inches(5.45)
    for o in obs:
        add_text_box(slide, o, Inches(0.55), y4, Inches(12.1), Inches(0.38),
                     font_size=12, color=C_BLACK)
        y4 += Inches(0.38)


def slide_14_findings(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Temuan Penting",
               "Insight teknis dari eksperimen")
    footer(slide, 14)

    findings = [
        {
            "num": "1",
            "title": "Semua Model ROC-AUC = 1.0000",
            "body": "Ketiga model mencapai discriminability sempurna di test set. Penambahan data LibriVox meningkatkan variasi bona-fide secara signifikan.",
            "color": C_GREEN,
        },
        {
            "num": "2",
            "title": "AASIST: Val Loss Volatile tapi Performatif",
            "body": "Lonjakan dramatis di epoch 3, 7–9 menunjukkan instabilitas training. Namun checkpoint epoch 6 berhasil menangkap representasi yang tepat. EER turun dari 0.56% → 0.00%.",
            "color": C_ACCENT_BLUE,
        },
        {
            "num": "3",
            "title": "MoLEx: Efisien & Stabil",
            "body": "Training konsisten tanpa lonjakan loss. Dengan hanya 179.491 parameter (14× lebih sedikit dari AASIST), MoLEx mencapai accuracy tertinggi 99.96%.",
            "color": C_ORANGE,
        },
        {
            "num": "4",
            "title": "Potensi Distributional Leak",
            "body": "Spoof dihasilkan dari kalimat yang SAMA dengan bona-fide. Model mungkin belajar TTS artifacts, bukan pola linguistik umum. Perlu evaluasi dengan data out-of-distribution.",
            "color": RGBColor(0xC0, 0x39, 0x2B),
        },
        {
            "num": "5",
            "title": "TTSFree: False Negative Meningkat",
            "body": "3 file TTSFree lolos deteksi. Kemungkinan menggunakan arsitektur TTS lebih canggih dari training data. Perlu analisis spektral mendalam.",
            "color": RGBColor(0x7B, 0x2D, 0x8B),
        },
        {
            "num": "6",
            "title": "XGBoost: Kecepatan Inferensi Luar Biasa",
            "body": "0.0026 ms/sampel — ~1.770× lebih cepat dari model DL. Kandidat ideal untuk edge deployment dan sistem real-time dengan resource terbatas.",
            "color": RGBColor(0x17, 0x7A, 0x9C),
        },
    ]

    positions = [
        (Inches(0.4), Inches(1.45)),
        (Inches(4.55), Inches(1.45)),
        (Inches(8.7), Inches(1.45)),
        (Inches(0.4), Inches(4.2)),
        (Inches(4.55), Inches(4.2)),
        (Inches(8.7), Inches(4.2)),
    ]
    bw = Inches(4.0)
    bh = Inches(2.5)

    for f, (lx, ty) in zip(findings, positions):
        add_rect(slide, lx, ty, bw, Inches(0.45), fill_color=f["color"])
        add_text_box(slide, f"{f['num']}. {f['title']}",
                     lx + Inches(0.05), ty + Inches(0.07),
                     bw - Inches(0.1), Inches(0.35),
                     font_size=12, bold=True, color=C_WHITE)
        add_rect(slide, lx, ty + Inches(0.45), bw, bh - Inches(0.45),
                 fill_color=C_LIGHT_GRAY)
        add_text_box(slide, f["body"],
                     lx + Inches(0.1), ty + Inches(0.55),
                     bw - Inches(0.2), bh - Inches(0.6),
                     font_size=11, color=C_BLACK, wrap=True)


def slide_15_recommendations(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Rekomendasi Lanjutan",
               "Langkah berikutnya untuk meningkatkan robustness sistem")
    footer(slide, 15)

    priorities = [
        {
            "label": "PRIORITAS TINGGI",
            "color": RGBColor(0xC0, 0x39, 0x2B),
            "items": [
                "1. Evaluasi generalisasi dengan audio YouTube/podcast/call-center",
                "2. Zero-shot TTS evaluation (TTS baru yang tidak ada di training)",
                "3. Pisahkan kalimat bona-fide dan spoof untuk menghindari distributional leak",
                "4. Stabilkan AASIST: gradient clipping, LR warmup, fine-tune dari pre-trained",
            ],
        },
        {
            "label": "PRIORITAS SEDANG",
            "color": C_ORANGE,
            "items": [
                "5. Tambah TTS engine realistis: ElevenLabs, OpenAI TTS, voice conversion (VC) attacks",
                "6. Ensemble cerdas: weighted voting, stacking, Platt scaling",
                "7. Investigasi TTSFree: analisis spektral, identifikasi arsitektur TTS",
                "8. Data augmentation: noise, codec compression, room impulse response",
            ],
        },
        {
            "label": "PRIORITAS RENDAH / EKSPLORATIF",
            "color": C_GREEN,
            "items": [
                "9. Lightweight deployment XGBoost: ONNX / TreeLite untuk mobile/embedded",
                "10. Continual learning: update berkala dengan data baru + deteksi distribution shift",
                "11. Explainability: SHAP values (XGBoost), Grad-CAM (MoLEx) untuk forensik",
            ],
        },
    ]

    y_start = Inches(1.45)
    for p in priorities:
        add_rect(slide, Inches(0.4), y_start, Inches(2.2), Inches(0.42),
                 fill_color=p["color"])
        add_text_box(slide, p["label"],
                     Inches(0.4), y_start + Inches(0.05),
                     Inches(2.2), Inches(0.35),
                     font_size=11, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        dy = y_start + Inches(0.05)
        for item in p["items"]:
            add_rect(slide, Inches(2.65), dy, Inches(10.3), Inches(0.38),
                     fill_color=C_LIGHT_GRAY)
            add_text_box(slide, item, Inches(2.75), dy + Inches(0.05),
                         Inches(10.1), Inches(0.32),
                         font_size=11, color=C_BLACK)
            dy += Inches(0.42)
        y_start = dy + Inches(0.12)


def slide_16_conclusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Kesimpulan")
    footer(slide, 16)

    # big result box
    add_rect(slide, Inches(0.4), Inches(1.45), Inches(12.5), Inches(0.8),
             fill_color=C_DARK_BLUE)
    add_text_box(slide,
                 "Sistem deteksi deepfake audio Bahasa Indonesia end-to-end berhasil dibangun dengan performa sangat tinggi",
                 Inches(0.5), Inches(1.55), Inches(12.3), Inches(0.65),
                 font_size=16, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    # 3 conclusion cards
    cards = [
        {
            "icon": "DATASET",
            "title": "Dataset Komprehensif",
            "points": [
                "35.891 sampel bona-fide",
                "75.512 sampel spoof (4 TTS engine)",
                "Balanced split 80/10/10",
                "Mozilla Common Voice + LibriVox",
            ],
            "color": C_ACCENT_BLUE,
        },
        {
            "icon": "MODEL",
            "title": "3 Model Komplementer",
            "points": [
                "AASIST: EER 0.00%, raw waveform",
                "MoLEx: 99.96% acc, 14× lighter",
                "XGBoost: 0.0026 ms/sampel",
                "Semua ROC-AUC = 1.0000",
            ],
            "color": C_GREEN,
        },
        {
            "icon": "DEPLOY",
            "title": "Siap Dikembangkan",
            "points": [
                "Ensemble real-world inference",
                "XGBoost: edge deployment",
                "AASIST/MoLEx: cloud API",
                "Explainability ready (SHAP)",
            ],
            "color": C_ORANGE,
        },
    ]
    cw = Inches(4.0)
    cx_start = Inches(0.4)
    cy = Inches(2.4)
    gap = Inches(0.25)
    for i, card in enumerate(cards):
        cx = cx_start + i * (cw + gap)
        add_rect(slide, cx, cy, cw, Inches(0.5), fill_color=card["color"])
        add_text_box(slide, card["title"],
                     cx + Inches(0.05), cy + Inches(0.08),
                     cw - Inches(0.1), Inches(0.38),
                     font_size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_rect(slide, cx, cy + Inches(0.5), cw, Inches(1.8), fill_color=C_LIGHT_GRAY)
        y_pt = cy + Inches(0.6)
        for pt in card["points"]:
            add_text_box(slide, f"• {pt}",
                         cx + Inches(0.12), y_pt, cw - Inches(0.2), Inches(0.38),
                         font_size=12, color=C_BLACK)
            y_pt += Inches(0.38)

    # quote box
    add_rect(slide, Inches(0.4), Inches(4.85), Inches(12.5), Inches(0.9),
             fill_color=RGBColor(0xE8, 0xF4, 0xE8))
    add_text_box(slide,
                 '"Dengan ROC-AUC = 1.0000 pada ketiga model, pipeline ini menunjukkan bahwa deteksi '
                 'deepfake audio Bahasa Indonesia dapat dilakukan dengan akurasi sangat tinggi menggunakan '
                 'data open-source dan teknik yang ada."',
                 Inches(0.6), Inches(4.95), Inches(12.1), Inches(0.75),
                 font_size=13, italic=True, color=C_DARK_BLUE, align=PP_ALIGN.CENTER)

    # future outlook
    add_rect(slide, Inches(0.4), Inches(5.9), Inches(12.5), Inches(0.9),
             fill_color=C_LIGHT_GRAY)
    add_text_box(slide, "Langkah Selanjutnya:",
                 Inches(0.5), Inches(5.95), Inches(3.0), Inches(0.35),
                 font_size=13, bold=True, color=C_ORANGE)
    add_text_box(slide,
                 "Evaluasi OOD → Tambah TTS realistis → Pisahkan distribusi kalimat → Deployment edge/cloud",
                 Inches(3.4), Inches(5.95), Inches(9.3), Inches(0.35),
                 font_size=13, color=C_BLACK)


def slide_17_references(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_WHITE)
    header_bar(slide, "Referensi")
    footer(slide, 17)

    refs = [
        ("[1]", "Jung, J. W., et al. (2022). AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal "
                "Graph Attention Networks. ICASSP 2022, pp. 6367–6371. doi:10.1109/ICASSP43922.2022.9747766"),
        ("[2]", "Wang, X., et al. (2020). ASVspoof 2019: A Large-Scale Public Database of Synthesized, "
                "Converted and Replayed Speech. Computer Speech & Language, 64, 101114."),
        ("[3]", "Yi, J., et al. (2022). ADD 2022: The First Audio Deep Synthesis Detection Challenge. "
                "ICASSP 2022, pp. 9226–9230."),
        ("[4]", "Sahidullah, M., et al. (2015). A Comparison of Features for Synthetic Speech Detection. "
                "Interspeech 2015, pp. 2087–2091."),
        ("[5]", "Wu, X., et al. (2018). Light CNN for Deep Face Representation with Noisy Labels. "
                "IEEE Transactions on Information Forensics and Security, 13(11), pp. 2884–2896."),
        ("[6]", "Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. "
                "Proc. 22nd ACM SIGKDD, pp. 785–794. doi:10.1145/2939672.2939785"),
        ("[7]", "Ardila, R., et al. (2020). Common Voice: A Massively-Multilingual Speech Corpus. "
                "LREC 2020, pp. 4218–4222."),
        ("[8]", "Pratap, V., et al. (2023). Scaling Speech Technology to 1,000+ Languages. "
                "arXiv:2305.13516. Meta AI Research."),
        ("[9]", "Ravanelli, M. & Bengio, Y. (2018). Speaker Recognition from Raw Waveform with SincNet. "
                "SLT 2018, pp. 1021–1028. doi:10.1109/SLT.2018.8639585"),
        ("[10]", "Davis, S. B. & Mermelstein, P. (1980). Comparison of Parametric Representations for "
                 "Monosyllabic Word Recognition in Continuously Spoken Sentences. "
                 "IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(4), pp. 357–366."),
        ("[11]", "Wenger, E., et al. (2021). Hello, It's Me: Deep Learning-based Speech Synthesis Attacks "
                 "in the Real World. ACM CCS 2021. doi:10.1145/3460120.3484742"),
        ("[12]", "Bird, J. J. & Lotfi, A. (2023). Real-Time Detection of AI-Generated Speech for Deepfake "
                 "Voice Conversion. arXiv:2310.12204."),
        ("[13]", "Wu, C., et al. (2025). SEA-SPOOF: Bridging the Gap in Multilingual Audio Deepfake Detection "
                 "for South-East Asia. arXiv:2504."),
        ("[14]", "Mawalim, C. O., et al. (2025). InaSAS: Benchmarking Indonesian Speech Antispoofing Systems. "
                 "arXiv:2502."),
    ]

    y = Inches(1.45)
    for num, ref in refs:
        add_rect(slide, Inches(0.35), y, Inches(0.5), Inches(0.35), fill_color=C_ACCENT_BLUE)
        add_text_box(slide, num, Inches(0.35), y + Inches(0.04),
                     Inches(0.5), Inches(0.28),
                     font_size=9, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_text_box(slide, ref, Inches(0.9), y + Inches(0.02),
                     Inches(12.1), Inches(0.35),
                     font_size=9, color=C_BLACK)
        y += Inches(0.38)


def slide_18_thankyou(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, C_DARK_BLUE)

    add_rect(slide, 0, 0, SLIDE_W, Inches(0.18), fill_color=C_ACCENT_BLUE)
    add_rect(slide, 0, Inches(0.18), SLIDE_W, Inches(0.06), fill_color=C_YELLOW)

    add_text_box(slide, "Terima Kasih",
                 Inches(1.0), Inches(1.8), Inches(11.3), Inches(1.2),
                 font_size=52, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    add_rect(slide, Inches(4.0), Inches(3.0), Inches(5.3), Inches(0.05),
             fill_color=C_YELLOW)

    add_text_box(slide, "Reynaldi Pamungkas",
                 Inches(1.0), Inches(3.15), Inches(11.3), Inches(0.55),
                 font_size=22, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, "Institut Teknologi Bandung  |  2026",
                 Inches(1.0), Inches(3.65), Inches(11.3), Inches(0.45),
                 font_size=16, color=RGBColor(0x88, 0xAA, 0xCC), align=PP_ALIGN.CENTER)

    # summary stats
    stats_items = [
        ("35.891", "Sampel\nBona Fide"),
        ("75.512", "Sampel\nSpoof"),
        ("3", "Model\nTerlatih"),
        ("1.0000", "ROC-AUC\nSemua Model"),
    ]
    bw = Inches(2.6)
    bh = Inches(1.5)
    sx = Inches(1.2)
    sy = Inches(4.5)
    gap = Inches(0.6)
    for i, (val, label) in enumerate(stats_items):
        x = sx + i * (bw + gap)
        add_rect(slide, x, sy, bw, bh, fill_color=RGBColor(0x11, 0x35, 0x5C))
        add_text_box(slide, val, x + Inches(0.05), sy + Inches(0.15),
                     bw - Inches(0.1), Inches(0.7),
                     font_size=24, bold=True, color=C_YELLOW, align=PP_ALIGN.CENTER)
        add_text_box(slide, label, x + Inches(0.05), sy + Inches(0.8),
                     bw - Inches(0.1), Inches(0.6),
                     font_size=13, color=RGBColor(0xB0, 0xC8, 0xE8), align=PP_ALIGN.CENTER)

    add_rect(slide, 0, Inches(6.8), SLIDE_W, Inches(0.7), fill_color=C_ACCENT_BLUE)
    add_text_box(slide,
                 "Pipeline & kode tersedia di: /Users/rey/ITB/semester_2/PPT/reformat_create_dataset/",
                 Inches(0.5), Inches(6.88), Inches(12.3), Inches(0.4),
                 font_size=12, color=C_WHITE, align=PP_ALIGN.CENTER)


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    print("Building slides...")
    slide_01_title(prs)
    print("  [1/18] Title")
    slide_02_agenda(prs)
    print("  [2/18] Agenda")
    slide_03_background(prs)
    print("  [3/18] Latar Belakang")
    slide_04_pipeline_overview(prs)
    print("  [4/18] Pipeline Overview")
    slide_05_bonafide(prs)
    print("  [5/18] Bona Fide Dataset")
    slide_06_spoof(prs)
    print("  [6/18] Spoof Generation")
    slide_07_dataset_split(prs)
    print("  [7/18] Dataset Split")
    slide_08_aasist_arch(prs)
    print("  [8/18] AASIST Architecture")
    slide_09_molex_arch(prs)
    print("  [9/18] MoLEx Architecture")
    slide_10_xgboost(prs)
    print("  [10/18] XGBoost Features")
    slide_11_training_results(prs)
    print("  [11/18] Training Results")
    slide_12_comparison(prs)
    print("  [12/18] Model Comparison")
    slide_13_realworld(prs)
    print("  [13/18] Real-World Inference")
    slide_14_findings(prs)
    print("  [14/18] Key Findings")
    slide_15_recommendations(prs)
    print("  [15/18] Recommendations")
    slide_16_conclusion(prs)
    print("  [16/18] Conclusion")
    slide_17_references(prs)
    print("  [17/18] References")
    slide_18_thankyou(prs)
    print("  [18/18] Thank You")

    output_path = "/Users/rey/ITB/semester_2/PPT/reformat_create_dataset/Presentasi_Deepfake_Audio_Bahasa_Indonesia.pptx"
    prs.save(output_path)
    print(f"\nPresentasi disimpan: {output_path}")
    return output_path


if __name__ == "__main__":
    build_presentation()
