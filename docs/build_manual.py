"""Generate a professional PDF manual for nebula3d."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
NAVY = colors.HexColor("#1a2d4f")
BLUE = colors.HexColor("#2c5282")
LIGHT_BLUE = colors.HexColor("#ebf4ff")
STEEL = colors.HexColor("#4a6fa5")
GOLD = colors.HexColor("#d4a017")
LIGHT_GREY = colors.HexColor("#f5f5f5")
MID_GREY = colors.HexColor("#cccccc")
DARK_GREY = colors.HexColor("#555555")
CODE_BG = colors.HexColor("#f0f4f8")
CODE_BORDER = colors.HexColor("#c3d5e8")

PAGE_W, PAGE_H = A4

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
_base = getSampleStyleSheet()


def _ps(**kw):
    """Shorthand ParagraphStyle factory."""
    return ParagraphStyle(**kw)


styles = {
    "title": _ps(
        name="ManTitle",
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=6,
    ),
    "subtitle": _ps(
        name="ManSubtitle",
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        textColor=STEEL,
        alignment=TA_CENTER,
        spaceAfter=4,
    ),
    "version": _ps(
        name="ManVersion",
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=DARK_GREY,
        alignment=TA_CENTER,
        spaceAfter=2,
    ),
    "h1": _ps(
        name="ManH1",
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=NAVY,
        spaceBefore=18,
        spaceAfter=6,
        keepWithNext=1,
    ),
    "h2": _ps(
        name="ManH2",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=BLUE,
        spaceBefore=14,
        spaceAfter=4,
        keepWithNext=1,
    ),
    "h3": _ps(
        name="ManH3",
        fontName="Helvetica-BoldOblique",
        fontSize=11,
        leading=14,
        textColor=STEEL,
        spaceBefore=10,
        spaceAfter=3,
        keepWithNext=1,
    ),
    "body": _ps(
        name="ManBody",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.black,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    ),
    "body_left": _ps(
        name="ManBodyLeft",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.black,
        spaceAfter=6,
        alignment=TA_LEFT,
    ),
    "bullet": _ps(
        name="ManBullet",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.black,
        spaceAfter=3,
        leftIndent=16,
        bulletIndent=0,
        alignment=TA_LEFT,
    ),
    "code": _ps(
        name="ManCode",
        fontName="Courier",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=3,
        leftIndent=6,
        rightIndent=6,
    ),
    "math": _ps(
        name="ManMath",
        fontName="Courier-Bold",
        fontSize=9.5,
        leading=14,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=4,
    ),
    "caption": _ps(
        name="ManCaption",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=DARK_GREY,
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=8,
    ),
    "note": _ps(
        name="ManNote",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=DARK_GREY,
        spaceAfter=4,
        leftIndent=12,
    ),
    "toc_h1": _ps(
        name="TOCH1",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        leftIndent=0,
        spaceAfter=2,
    ),
    "toc_h2": _ps(
        name="TOCH2",
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        leftIndent=12,
        spaceAfter=1,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def H1(text, toc=True):
    p = Paragraph(text, styles["h1"])
    p._bookmarkName = text.replace(" ", "_")
    return p


def H2(text):
    return Paragraph(text, styles["h2"])


def H3(text):
    return Paragraph(text, styles["h3"])


def P(text):
    return Paragraph(text, styles["body"])


def PL(text):
    return Paragraph(text, styles["body_left"])


def B(text):
    return Paragraph(f"•  {text}", styles["bullet"])


def SP(n=6):
    return Spacer(1, n)


def HR():
    return HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=4, spaceBefore=4)


def code_block(text):
    """Return a table-wrapped code block."""
    lines = [Paragraph(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                        styles["code"]) for line in text.split("\n")]
    tbl = Table([[lines]], colWidths=[PAGE_W - 5.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, CODE_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return tbl


def math_block(text):
    return Paragraph(text, styles["math"])


def param_table(rows, col_widths=None):
    """Render a parameter table.  rows = list of (name, type, default, description)."""
    if col_widths is None:
        col_widths = [3.5 * cm, 2.2 * cm, 2.8 * cm, 8.5 * cm]
    header = ["Parameter", "Type", "Default", "Description"]
    data = [header] + rows
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


def two_col_table(rows, header=None, col_widths=None):
    """Render a simple two-column table."""
    if col_widths is None:
        col_widths = [5 * cm, 12 * cm]
    data = ([header] if header else []) + rows
    tbl = Table(data, colWidths=col_widths)
    style = [
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
        ]
    tbl.setStyle(TableStyle(style))
    return tbl


def info_box(title, body_text):
    """Blue info box."""
    content = [
        Paragraph(f"<b>{title}</b>", _ps(name="_ib_t", fontName="Helvetica-Bold",
                                          fontSize=9, leading=12, textColor=NAVY, spaceAfter=2)),
        Paragraph(body_text, _ps(name="_ib_b", fontName="Helvetica", fontSize=9,
                                  leading=13, textColor=colors.black)),
    ]
    tbl = Table([[content]], colWidths=[PAGE_W - 5.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Page template callbacks
# ---------------------------------------------------------------------------
class _CollectorDoc(SimpleDocTemplate):
    """Pass-1 document: writes to BytesIO and collects TOC entries."""

    def __init__(self, **kw):
        import io
        super().__init__(io.BytesIO(), **kw)
        self.toc_entries: list = []  # [(level, text, page)]

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            text = flowable.getPlainText()
            if text == "Table of Contents":
                return
            if style_name == "ManH1":
                self.toc_entries.append((0, text, self.page))
            elif style_name == "ManH2":
                self.toc_entries.append((1, text, self.page))


def _build_static_toc(entries, page_offset: int) -> Table:
    """Build a plain Table TOC from (level, text, page) entries."""
    rows = []
    for level, text, page in entries:
        adj_page = str(page + page_offset)
        dot_fill = "." * max(2, 80 - len(text) - len(adj_page) - level * 2)
        indent = "    " * level  # non-breaking spaces for indent
        label = Paragraph(f"{indent}{text}", styles["toc_h1" if level == 0 else "toc_h2"])
        dots = Paragraph(dot_fill, styles["toc_h2" if level == 1 else "toc_h1"])
        num = Paragraph(adj_page, _ps(name="_toc_num", fontName="Helvetica",
                                       fontSize=10 if level == 1 else 11,
                                       leading=14, alignment=TA_RIGHT))
        rows.append([label, num])
    if not rows:
        return Spacer(1, 0)
    tbl = Table(rows, colWidths=[PAGE_W - 5.5 * cm - 1.5 * cm, 1.5 * cm])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4

    # Header bar
    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.rect(doc.leftMargin, h - 1.5 * cm,
                    w - doc.leftMargin - doc.rightMargin, 0.35 * cm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.white)
        canvas.drawString(doc.leftMargin, h - 1.35 * cm,
                          "nebula3d  |  Manual  v0.1")
        canvas.setFillColor(DARK_GREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - doc.rightMargin, h - 1.35 * cm,
                               f"Page {doc.page}")

    # Footer line
    if doc.page > 1:
        canvas.setStrokeColor(MID_GREY)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 1.8 * cm,
                    w - doc.rightMargin, 1.8 * cm)
        canvas.setFont("Helvetica-Oblique", 7.5)
        canvas.setFillColor(DARK_GREY)
        canvas.drawCentredString(w / 2, 1.4 * cm,
                                 "3D Diffuse Neutron Scattering Analysis Toolkit")

    canvas.restoreState()


# ---------------------------------------------------------------------------
# Build document
# ---------------------------------------------------------------------------
_DOC_KW = dict(
    pagesize=A4,
    leftMargin=2.5 * cm,
    rightMargin=2.5 * cm,
    topMargin=2.2 * cm,
    bottomMargin=2.5 * cm,
    title="nebula3d Manual",
    author="nebula3d Development Team",
    subject="3D Diffuse-Scattering Analysis",
)


def build_manual(output_path: str) -> None:
    # ---- Pass 1: build with a zero-height TOC placeholder to collect pages ----
    _toc_ph = Spacer(1, 0)   # replaced in pass 2
    story = []

    # -----------------------------------------------------------------------
    # COVER PAGE
    # -----------------------------------------------------------------------
    story.append(SP(80))
    story.append(Paragraph("nebula3d", styles["title"]))
    story.append(SP(4))
    story.append(Paragraph("Manual", styles["subtitle"]))
    story.append(SP(8))

    # Gold separator
    story.append(HRFlowable(width="60%", thickness=2, color=GOLD, hAlign="CENTER",
                              spaceAfter=16, spaceBefore=8))
    story.append(Paragraph(
        "3D Reciprocal-Space Diffuse Neutron Scattering Analysis<br/>"
        "Powder Ring Removal · Bragg Punch &amp; Fill · 3D-ΔPDF Transform",
        styles["subtitle"],
    ))
    story.append(SP(24))
    story.append(Paragraph("Version 0.1.0", styles["version"]))
    story.append(Paragraph("June 2026", styles["version"]))
    story.append(SP(80))

    # Bottom metadata box
    _meta_cell = _ps(name="_mc", fontName="Helvetica", fontSize=9, leading=12)
    meta_rows = [
        ["Language",         Paragraph("Python ≥ 3.10", _meta_cell)],
        ["License",          Paragraph("MIT", _meta_cell)],
        ["Dependencies",     Paragraph("numpy, scipy, h5py, matplotlib", _meta_cell)],
        ["Instrument",       Paragraph("Mantid NeXus (MDHistoWorkspace)", _meta_cell)],
        ["Reference data", Paragraph(
            "Mantid NeXus HKL volumes", _meta_cell)],
    ]
    meta_tbl = Table(meta_rows, colWidths=[4.5 * cm, 10 * cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (0, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GREY]),
        ("BOX", (0, 0), (-1, -1), 0.5, STEEL),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0))   # flexible gap fills remaining cover space
    story.append(HRFlowable(width="100%", thickness=0.4, color=MID_GREY,
                             spaceBefore=14, spaceAfter=6))
    story.append(Paragraph(
        "Tsung-Han Yang",
        _ps(name="_cov_name", fontName="Helvetica-Bold", fontSize=10,
            leading=14, textColor=NAVY, alignment=TA_CENTER),
    ))
    story.append(Paragraph(
        "© 2026 Tsung-Han Yang. All rights reserved.",
        _ps(name="_cov_copy", fontName="Helvetica", fontSize=8.5,
            leading=12, textColor=DARK_GREY, alignment=TA_CENTER, spaceAfter=6),
    ))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # TABLE OF CONTENTS
    # -----------------------------------------------------------------------
    story.append(H1("Table of Contents"))
    _toc_story_idx = len(story)   # remember position for pass-2 replacement
    story.append(_toc_ph)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # 1  INTRODUCTION
    # -----------------------------------------------------------------------
    story.append(H1("1.  Introduction"))
    story.append(P(
        "<b>nebula3d</b> is a Python 3.10+ toolkit for the complete "
        "analysis of three-dimensional (3D) diffuse neutron scattering volumes. "
        "It implements a production-ready four-stage pipeline that transforms raw "
        "single-crystal Mantid NeXus output into real-space 3D difference pair "
        "distribution functions (3D-ΔPDF):"
    ))
    stages = [
        ("Stage 1", "Powder-ring removal",
         "Subtracts azimuthally smooth ring contamination from the cryostat, "
         "sample holder, and sample environment without masking real diffuse signal."),
        ("Stage 2", "Bragg punch",
         "Identifies and masks sharp Bragg and satellite peaks using lattice-aware "
         "integer-node detection, data-driven shaping, and an hkl-agnostic search mode "
         "for off-integer satellites."),
        ("Stage 3", "Backfill",
         "Replaces masked voxels with physically reasonable estimates derived from the "
         "surrounding radial background or crystal symmetry equivalents."),
        ("Stage 4", "3D-ΔPDF transform",
         "Converts the cleaned reciprocal-space volume to a real-space pair-correlation "
         "map via a correctly centred 3D discrete Fourier transform with apodization and "
         "smooth-background subtraction."),
    ]
    for s_id, s_name, s_desc in stages:
        story.append(KeepTogether([
            P(f"<b>{s_id} — {s_name}.</b>  {s_desc}"),
        ]))

    story.append(SP(6))
    story.append(P(
        "The package is designed around a single core data structure "
        "(<tt>HKLVolume</tt>) that carries the 3D intensity array, per-voxel "
        "standard deviations, a validity mask, HKL coordinate axes, and the "
        "crystal orientation (UB) matrix. All pipeline stages operate on this "
        "type, making it straightforward to inspect intermediate results, apply "
        "individual stages, or extend the pipeline with custom algorithms."
    ))
    story.append(P(
        "Visualization is provided through a thin, scriptable layer of Matplotlib "
        "primitives: 2D plane slices, radial profiles, azimuthal ring-texture maps, "
        "a 2×2 overview diagnostic, and three interactive orthogonal-cut real-space "
        "viewers for quality assurance at every stage."
    ))

    story.append(H2("1.1  Scope of This Manual"))
    story.append(P(
        "This manual covers the background physics and mathematics of 3D diffuse "
        "neutron scattering analysis, the detailed algorithms and their theoretical "
        "basis, a step-by-step workflow guide, the full Python API, configuration "
        "parameters, generic worked examples for single-volume and multi-volume "
        "workflows, interactive visualization, known limitations, and "
        "a complete reference list."
    ))

    story.append(H2("1.2  What nebula3d Does Not Do"))
    story.append(P(
        "The package does not perform raw detector reduction, normalization to an "
        "absolute cross-section, or Reverse Monte Carlo (RMC) structural refinement. "
        "It assumes the input is a Mantid background-subtracted MDHistoWorkspace volume "
        "(<tt>*_cc_sub_bkg.nxs</tt>). For the reduction steps that precede this, use "
        "Mantid directly."
    ))

    # -----------------------------------------------------------------------
    # 2  BACKGROUND THEORY
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("2.  Background Theory"))

    story.append(H2("2.1  Neutron Scattering Fundamentals"))
    story.append(P(
        "In a neutron scattering experiment a monochromatic beam of neutrons with "
        "wavevector <b>k</b><sub>i</sub> is scattered by the sample. Neutrons with "
        "final wavevector <b>k</b><sub>f</sub> are detected as a function of the "
        "momentum transfer"
    ))
    story.append(math_block("Q = k_f − k_i   ( |Q| = Q = 4π sinθ / λ )"))
    story.append(P(
        "For an elastic measurement (|<b>k</b><sub>i</sub>| = |<b>k</b><sub>f</sub>|) "
        "on a single crystal, the scattered intensity <i>I</i>(<b>Q</b>) is recorded "
        "over a 3D grid in reciprocal space indexed by the fractional Miller indices "
        "(<i>h</i>, <i>k</i>, <i>l</i>)."
    ))
    story.append(P(
        "The total measured intensity separates into two components:"
    ))
    story.append(math_block("I_total(Q) = I_Bragg(Q) + I_diffuse(Q)"))
    story.append(P(
        "The <b>Bragg contribution</b> <i>I</i><sub>Bragg</sub> consists of sharp "
        "peaks at integer lattice nodes (and satellite peaks for modulated structures). "
        "It encodes the average periodic structure. The <b>diffuse contribution</b> "
        "<i>I</i><sub>diffuse</sub> is the remaining broad, continuous signal that "
        "encodes deviations from the average: atomic displacements, occupancy "
        "disorder, short-range magnetic correlations, and phonon/magnon scattering."
    ))

    story.append(H2("2.2  The 3D Difference Pair Distribution Function"))
    story.append(P(
        "The three-dimensional difference pair distribution function (3D-ΔPDF) is "
        "defined as the Fourier transform of the diffuse scattering intensity:"
    ))
    story.append(math_block("Δρ(r) = FT[ I_diffuse(Q) ] = FT[ I_total(Q) − I_Bragg(Q) ]"))
    story.append(P(
        "In real space, Δρ(<b>r</b>) is a three-dimensional map of interatomic "
        "correlation deviations. A positive value at vector <b>r</b> means there are "
        "<i>more</i> interatomic pairs separated by <b>r</b> than the average structure "
        "predicts; a negative value means <i>fewer</i>. Peaks in Δρ(<b>r</b>) "
        "at specific real-space positions directly identify the geometry of correlated "
        "displacements or ordered nano-domains."
    ))
    story.append(P(
        "The key practical relation connecting measurement to analysis is:"
    ))
    story.append(math_block(
        "Δρ(r) ≈ FT[ I_total(Q) − I_Bragg(Q) ]"
        "  =  FT[ I_cleaned(Q) ]"
    ))
    story.append(P(
        "where <i>I</i><sub>cleaned</sub>(<b>Q</b>) is the diffuse volume after "
        "ring removal, Bragg punching, and backfill. The accuracy of Δρ "
        "therefore depends directly on the quality of each preprocessing stage."
    ))
    story.append(info_box(
        "Key Reference",
        "Weber & Simonov, Z. Kristallogr. 227, 238–247 (2012). "
        "Simonov, Weber & Steurer, J. Appl. Cryst. 47, 2011–2018 (2014). "
        "These two papers establish the 3D-ΔPDF formalism and the punch-and-fill "
        "data-reduction strategy implemented in this package."
    ))

    story.append(H2("2.3  Coordinate Systems and Conventions"))
    story.append(P(
        "Two coordinate systems are used throughout the package:"
    ))
    story.append(B(
        "<b>Fractional HKL.</b>  Reciprocal-lattice coordinates (<i>h</i>, <i>k</i>, "
        "<i>l</i>) measured in units of the reciprocal basis vectors "
        "<b>a</b>*, <b>b</b>*, <b>c</b>*. Integer values correspond to Bragg "
        "positions; fractional values index the continuous diffuse volume."
    ))
    story.append(B(
        "<b>Cartesian Q.</b>  Physical momentum transfer in units of Å⁻¹, "
        "related to HKL by the UB matrix: <b>Q</b> = UB · [<i>h</i>, <i>k</i>, "
        "<i>l</i>]ᵀ. The magnitude |<b>Q</b>| = 2π/<i>d</i> (physics convention, "
        "as opposed to the crystallographic 1/<i>d</i> used in Mantid)."
    ))
    story.append(P(
        "The <b>UB matrix</b> combines the orientation matrix U (aligning the crystal "
        "axes to the laboratory frame) and the reciprocal-metric matrix B "
        "(encoding the lattice parameters). Mantid files store the crystallographic "
        "UB (no 2π factor); nebula3d reads this and multiplies by 2π "
        "to convert to the physics convention."
    ))
    story.append(math_block("Q = 2π · UB_cryst · [h, k, l]ᵀ   ( Å⁻¹ )"))

    story.append(H2("2.4  Powder Rings: Physical Origin"))
    story.append(P(
        "Polycrystalline material in the beam path — the sample cryostat, aluminium "
        "heat shields, sample holder, and sample capsule walls — produces "
        "<i>powder diffraction rings</i>: bands of elevated intensity at fixed "
        "|<b>Q</b>| values corresponding to the <i>d</i>-spacings of the polycrystalline "
        "material. For aluminium (FCC, <i>a</i> = 4.046 Å), the strongest ring "
        "is Al(111) at |<b>Q</b>| = 2.69 Å⁻¹."
    ))
    story.append(P(
        "Rings are not isotropic. Detector solid-angle coverage, absorption path "
        "length, and normalization artifacts modulate the ring intensity with azimuthal "
        "direction φ. A useful physical model for the ring contribution at a voxel "
        "with scattering vector magnitude |<b>Q</b>| and azimuthal angle φ is:"
    ))
    story.append(math_block(
        "I_ring(Q, φ) = T(φ) × Σ_i  A_i · G( |Q| − q_i, σ_i )"
    ))
    story.append(P(
        "where <i>G</i>(|<b>Q</b>|−<i>q</i><sub>i</sub>, σ<sub>i</sub>) is "
        "a Gaussian radial profile for ring <i>i</i>, <i>A</i><sub>i</sub> is the "
        "per-ring amplitude, and <i>T</i>(φ) is the shared azimuthal texture "
        "function. The measured signal is:"
    ))
    story.append(math_block(
        "I_measured(Q, φ) = I_diffuse(Q) + I_ring(Q, φ)"
    ))
    story.append(P(
        "Since the diffuse signal of interest does <i>not</i> share the ring's radial "
        "peak structure or azimuthal texture, it is in principle separable from the "
        "ring contribution."
    ))

    # -----------------------------------------------------------------------
    # 3  PACKAGE ARCHITECTURE
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("3.  Package Architecture"))

    story.append(H2("3.1  Overview"))
    story.append(P(
        "The package is installed from source under <tt>src/nebula3d/</tt> and follows "
        "a strict separation between the core data structure, I/O, the four algorithmic "
        "stages, an inpainting library, and visualization:"
    ))
    arch_rows = [
        ["<tt>nebula3d/core.py</tt>", "HKLVolume dataclass — the universal data carrier"],
        ["<tt>nebula3d/io/</tt>", "Mantid NeXus reader, HDF5 load/save, ASCII HKL I/O"],
        ["<tt>nebula3d/preprocessing/</tt>", "Powder-ring models, radial background, backfill"],
        ["<tt>nebula3d/analysis/</tt>", "Bragg punch/backfill, 3D-ΔPDF FFT"],
        ["<tt>nebula3d/inpainting/</tt>", "Symmetry fill, TV (Chambolle-Pock), RBF, biharmonic"],
        ["<tt>nebula3d/visualization/</tt>", "Slice plots, radial profiles, interactive viewers"],
        ["<tt>nebula3d/utils/</tt>", "UB matrix, d-spacing, Q↔HKL conversion utilities"],
        ["<tt>examples/</tt>", "Pipeline scripts and interactive viewer entry points"],
    ]
    story.append(two_col_table(
        [[Paragraph(r, styles["body_left"]), Paragraph(d, styles["body_left"])]
         for r, d in arch_rows],
        header=[Paragraph("<b>Module</b>", styles["body_left"]),
                Paragraph("<b>Responsibility</b>", styles["body_left"])],
        col_widths=[5 * cm, 12 * cm],
    ))

    story.append(H2("3.2  The HKLVolume Data Structure"))
    story.append(P(
        "Every stage of the pipeline operates on an <tt>HKLVolume</tt> dataclass "
        "defined in <tt>src/nebula3d/core.py</tt>:"
    ))
    story.append(code_block(
        "@dataclass\n"
        "class HKLVolume:\n"
        "    data   : NDArray[np.float64]   # shape (nh, nk, nl) intensity\n"
        "    sigma  : NDArray[np.float64]   # shape (nh, nk, nl) standard deviation\n"
        "    mask   : NDArray[np.bool_]     # True = valid (not masked)\n"
        "    h_axis : NDArray               # 1D fractional-HKL H coordinates\n"
        "    k_axis : NDArray               # 1D fractional-HKL K coordinates\n"
        "    l_axis : NDArray               # 1D fractional-HKL L coordinates\n"
        "    ub_matrix : NDArray[np.float64]# (3, 3) physics-convention UB (Angstrom^-1)\n"
        "    instrument: str                # provenance label"
    ))
    story.append(P("Key methods provided by HKLVolume:"))
    hkl_methods = [
        ["<tt>hkl_grid()</tt>", "Returns (H, K, L) meshgrids of shape (nh, nk, nl)"],
        ["<tt>q_magnitude()</tt>", "Returns |Q| in Å⁻¹ for every voxel"],
        ["<tt>masked_data()</tt>", "Returns data with masked voxels set to NaN"],
        ["<tt>apply_mask(m)</tt>", "Updates the mask in place"],
    ]
    story.append(two_col_table(
        [[Paragraph(m, styles["code"]), Paragraph(d, styles["body_left"])]
         for m, d in hkl_methods],
        header=[Paragraph("<b>Method</b>", styles["body_left"]),
                Paragraph("<b>Description</b>", styles["body_left"])],
        col_widths=[5 * cm, 12 * cm],
    ))

    story.append(H2("3.3  File Formats"))
    story.append(H3("Input: Mantid NeXus"))
    story.append(P(
        "Raw input is a Mantid <tt>MDHistoWorkspace</tt> saved as a NeXus file "
        "(<tt>*_cc_sub_bkg.nxs</tt>). The reader (<tt>nebula3d/io/mantid_nxs.py</tt>) "
        "extracts the signal, variance, mask, bin-edge arrays, and the UB matrix from "
        "the standard Mantid HDF5 hierarchy. The Mantid crystallographic UB is "
        "rescaled by 2π on read."
    ))
    story.append(H3("Native HDF5"))
    story.append(P(
        "All intermediate and output files are stored as HDF5 (<tt>.h5</tt>). "
        "Load with <tt>vol = nebula3d.load(path)</tt>; save with <tt>nebula3d.save(vol, path)</tt>. "
        "The ΔPDF output additionally stores the transform configuration and direct-lattice "
        "constants as HDF5 attributes for use by the interactive viewers."
    ))

    # -----------------------------------------------------------------------
    # 4  ALGORITHMS
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("4.  Algorithms"))

    # 4.1 Ring removal
    story.append(H2("4.1  Stage 1: Powder Ring Removal"))
    story.append(H3("4.1.1  Design Principle"))
    story.append(P(
        "Ring removal is strictly <i>subtractive</i>. The algorithm estimates only "
        "the azimuthally smooth ring intensity and subtracts it. It never masks or "
        "replaces voxels solely because they exhibit radial excess, because radial "
        "excess can be genuine anisotropic diffuse scattering."
    ))

    story.append(H3("4.1.2  Non-Parametric Patch-Based Method (Production)"))
    story.append(P(
        "The production algorithm is implemented in "
        "<tt>nebula3d/preprocessing/radial_background.py</tt> as "
        "<tt>PatchedRadialRingModel</tt>. It processes the volume one H-slice "
        "(0kl plane) at a time and builds a non-parametric radial ring model in "
        "azimuthal patches:"
    ))
    steps = [
        ("1. Azimuthal patches",
         "The reference plane (default hk0) defines the azimuthal angle "
         "φ = atan2(k_Q, h_Q). The full range φ ∈ [0, 2π) is divided "
         "into N overlapping patches, each weighted by a Hann window to ensure smooth "
         "blending across patch boundaries."),
        ("2. Robust radial profile",
         "Within each azimuthal patch, voxels are binned by |Q| and a trimmed-mean "
         "profile is computed. The trimmed mean rejects high-tail outliers (Bragg peaks) "
         "and low-tail outliers (detector gaps), yielding a clean estimate of the combined "
         "diffuse + ring signal in each bin."),
        ("3. SNIP baseline estimation",
         "A smooth diffuse baseline is estimated from the robust profile using a "
         "morphological opening: successive minimum and maximum filters with a kernel "
         "wider than the expected ring width. Light polynomial smoothing is applied. "
         "This is equivalent to the Sensitive Nonlinear Iterative Peak (SNIP) "
         "clipping algorithm common in nuclear spectroscopy."),
        ("4. Ring component",
         "The ring component for each patch is: ring(|Q|) = max(0, prof − base). "
         "Taking the positive part ensures no diffuse signal is subtracted when "
         "the profile lies below the baseline (noise fluctuations)."),
        ("5. Subtraction",
         "Each voxel receives a Hann-weighted blend of the ring estimates from its two "
         "nearest azimuthal patches, interpolated to the voxel’s exact |Q|. "
         "Cross-H confirmed ring shells and amplitude ceilings from adjacent H planes "
         "are propagated to prevent integer-H Bragg artifacts from masquerading as "
         "powder rings in nearby planes."),
    ]
    for step_name, step_desc in steps:
        story.append(KeepTogether([
            P(f"<b>{step_name}.</b>  {step_desc}"),
        ]))

    story.append(H3("4.1.3  Factored Gaussian/SVD Model (Legacy)"))
    story.append(P(
        "The older algorithm (<tt>PatchedRingModel</tt>) remains in the package and "
        "is useful background. It fits a rank-1 factored model to the amplitude matrix:"
    ))
    story.append(math_block("A[i, P] ≈ A_i × T[P]"))
    story.append(P(
        "where index <i>i</i> runs over rings and <i>P</i> over azimuthal patches. "
        "A rank-1 SVD extracts per-ring amplitudes A<sub>i</sub> and patch textures T[P]. "
        "A Fourier series <i>T</i>(φ) is then fitted to give a smooth, periodic "
        "texture function. The quality of the rank-1 approximation is monitored via "
        "<tt>rank1_variance</tt>; values ≥ 0.90 confirm the shared-texture assumption."
    ))

    story.append(H3("4.1.4  Diagnostic Metrics"))
    story.append(two_col_table(
        [
            [Paragraph("<tt>rank1_variance</tt>", styles["code"]),
             Paragraph("Fraction of amplitude-matrix variance explained by rank-1 SVD. "
                        "≥ 0.90 confirms shared T(φ). Lower values suggest per-ring "
                        "texture fitting is needed.", styles["body_left"])],
            [Paragraph("Per-patch counts", styles["body_left"]),
             Paragraph("Sparse patches receive lower weight; uniform coverage is ideal.",
                        styles["body_left"])],
            [Paragraph("Raw vs baseline profiles", styles["body_left"]),
             Paragraph("Plot with plot_radial_profile() to visually verify ring isolation.",
                        styles["body_left"])],
        ],
        header=[Paragraph("<b>Diagnostic</b>", styles["body_left"]),
                Paragraph("<b>Interpretation</b>", styles["body_left"])],
        col_widths=[4.5 * cm, 12.5 * cm],
    ))

    # 4.2 Bragg punch
    story.append(H2("4.2  Stage 2: Bragg Peak Punching"))
    story.append(P(
        "Bragg punching is implemented in <tt>nebula3d/analysis/bragg.py</tt> by the "
        "<tt>BraggRemover</tt> class. The recommended production mode is "
        "<tt>mode=\"both\"</tt>, which combines a lattice-aware integer-node pass "
        "followed by an hkl-agnostic search pass."
    ))

    story.append(H3("4.2.1  Integer-Node Path"))
    story.append(P(
        "The integer-node path exploits the known lattice geometry:"
    ))
    int_steps = [
        ("Enumerate", "All integer (h, k, l) nodes within the HKL volume extent are listed."),
        ("Detect",
         "A local HKL window of half-width <tt>detect_window_hkl</tt> is inspected around "
         "each node. The node is retained only if the local peak exceeds "
         "<tt>min_intensity</tt> and the local median by <tt>min_prominence</tt>."),
        ("Recentre",
         "The punch centre is moved to the measured local maximum within the detection window."),
        ("Fit shape (optional)",
         "If <tt>integer_optimize_shape=True</tt>, the anisotropic covariance of the peak "
         "excess above a threshold fraction of the peak height is computed, yielding "
         "per-peak HKL ellipsoid radii. These are clipped by "
         "<tt>integer_fit_max_radius_hkl</tt> to prevent over-punching in sparse data."),
        ("Guard (H-slab)",
         "The parameter <tt>integer_h_guard_hkl</tt> clips each integer-node punch to a "
         "slab of half-width H<sub>guard</sub> centred on the integer-H plane. This "
         "prevents strong Bragg holes at H=0 from extending into the diffuse planes at "
         "H=±1/3 or H=±2/3."),
    ]
    for s_name, s_desc in int_steps:
        story.append(B(f"<b>{s_name}.</b>  {s_desc}"))

    story.append(H3("4.2.2  Weak Bragg at Integer Nodes"))
    story.append(P(
        "Weak peaks at integer nodes can fall below the absolute "
        "<tt>min_intensity</tt> and <tt>min_prominence</tt> floors but still be "
        "sharp local outliers. The parameter <tt>integer_local_prominence_n_mad</tt> "
        "catches these by requiring the prominence to exceed a multiple of the local "
        "Median Absolute Deviation (MAD) within the detection window. Because this "
        "criterion is <i>locked to integer lattice nodes</i> (never applied at fractional "
        "positions), it cannot punch the q=1/3 diffuse planes. An additional small "
        "absolute floor <tt>integer_local_min_prominence</tt> avoids false positives "
        "in very flat regions."
    ))

    story.append(H3("4.2.3  Search Path (HKL-Agnostic)"))
    story.append(P(
        "The search path detects off-integer satellites and any sharp feature "
        "missed by the integer pass. At each |<b>Q</b>| shell, a robust background "
        "level is estimated as:"
    ))
    story.append(math_block("bg_threshold = median(I_shell) + n_mad × MAD(I_shell)"))
    story.append(P(
        "Local maxima above this threshold and the absolute floor "
        "<tt>search_min_intensity</tt> are retained as punch centres. Because the "
        "search does not know the crystal lattice, structured diffuse planes must be "
        "explicitly protected:"
    ))
    story.append(B(
        "<b>Explicit exclusion.</b>  A list of H-plane centres "
        "(<tt>search_exclude_h_centers</tt>) defines fixed exclusion slabs of "
        "half-width <tt>search_exclude_h_half_width</tt>."
    ))
    story.append(B(
        "<b>Periodic exclusion (preferred).</b>  "
        "<tt>search_exclude_h_fractions</tt> defines fractional parts modulo 1. "
        "For example, fractions [0.3333, 0.6667] protect every H = n ± 1/3 and "
        "n ± 2/3 plane across the full H range, covering higher-order satellites "
        "at H = ±4/3, ±5/3, … that a fixed-centre list would miss."
    ))

    story.append(H3("4.2.4  Punch Shape"))
    story.append(P(
        "Each identified peak is punched as an anisotropic ellipsoid in HKL space:"
    ))
    story.append(math_block(
        "(h − h_0)^2/r_h^2 + (k − k_0)^2/r_k^2 + (l − l_0)^2/r_l^2 ≤ 1"
    ))
    story.append(P(
        "where (h<sub>0</sub>, k<sub>0</sub>, l<sub>0</sub>) is the fitted punch centre "
        "and (r<sub>h</sub>, r<sub>k</sub>, r<sub>l</sub>) are the HKL semi-axes. "
        "A guard margin is added to all radii. Optionally, the radii are scaled by "
        "the cube root of the peak intensity (relative to a reference), capped at "
        "<tt>max_radius_scale</tt>."
    ))

    story.append(H3("4.2.5  Direct Beam"))
    story.append(P(
        "The direct beam at <b>Q</b> = 0 is much broader than ordinary Bragg peaks "
        "and is treated separately. It is punched after all other peaks, using "
        "independent radii specified by <tt>incident_beam_ellipsoid_radii_hkl</tt>. "
        "The backfill of the direct-beam hole uses a specially shifted radial shell "
        "just outside the punch boundary, preventing the over-subtraction halo from "
        "contaminating the fill."
    ))

    # 4.3 Backfill
    story.append(H2("4.3  Stage 3: Bragg-Hole Backfill"))
    story.append(P(
        "After punching, masked voxels must be replaced with physically reasonable "
        "estimates before the Fourier transform. The dedicated wrapper is "
        "<tt>backfill_bragg()</tt> in <tt>nebula3d/analysis/bragg_fill.py</tt>."
    ))

    story.append(H3("4.3.1  Q-Shell Fill (Recommended)"))
    story.append(P(
        "For ordinary Bragg holes, the robust diffuse level at the same |<b>Q</b>| "
        "provides the best estimate. For each |<b>Q</b>| bin (step size "
        "<tt>q_shell_step</tt>), the background is estimated as:"
    ))
    story.append(math_block(
        "I_fill = median(I_valid_in_bin) + n_mad × MAD(I_valid_in_bin)"
    ))
    story.append(P(
        "where the median and MAD are computed over all unmasked voxels in the "
        "|<b>Q</b>| shell. If a shell has fewer than <tt>q_shell_min_count</tt> valid "
        "voxels, the algorithm falls back to local shell interpolation. This method is "
        "physically motivated: the diffuse intensity at a given |<b>Q</b>| is "
        "relatively smooth and isotropic, so the shell level is a good estimate for "
        "all punched voxels at that |<b>Q</b>|."
    ))

    story.append(H3("4.3.2  Local Fill"))
    story.append(P(
        "For fast visual checks or sparse synthetic volumes, "
        "<tt>method=\"local\"</tt> fills each connected punched component from a "
        "dilated-shell median of nearby valid voxels, falling back to the global "
        "median if too few neighbours are available."
    ))

    story.append(H3("4.3.3  General Inpainting Methods"))
    story.append(P(
        "For complex cases, the general inpainting pipeline in "
        "<tt>nebula3d/inpainting/pipeline.py</tt> provides:"
    ))
    inpaint_rows = [
        ["Symmetry", "Crystal Laue-symmetry equivalents (inverse-variance weighted)",
         "Exact; no smoothing; preserves all features",
         "Fails when all equivalents are also masked"],
        ["TV", "Total-variation Chambolle-Pock primal-dual",
         "Preserves sharp features; piecewise smooth",
         "Slower for large masks; may over-smooth"],
        ["RBF", "scipy RBFInterpolator, thin-plate spline",
         "Fast for small isolated masks",
         "Intrinsic smoothing"],
        ["Biharmonic", "∇⁴u = 0 iterative relaxation",
         "Very smooth fills",
         "Slow for large masks"],
    ]
    tbl = Table(
        [["Method", "Algorithm", "Strengths", "Limitations"]] + inpaint_rows,
        colWidths=[2.5 * cm, 4.5 * cm, 5 * cm, 5 * cm],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(tbl)

    story.append(H3("4.3.4  TV Inpainting: Mathematical Formulation"))
    story.append(P(
        "Total-variation inpainting solves the following constrained minimisation:"
    ))
    story.append(math_block(
        "min_u  (1/2) || W(u - f) ||<super>2</super>  +  λ ||∇u||<sub>1</sub>"
    ))
    story.append(P(
        "where <b>f</b> is the observed data (arbitrary in the masked region), "
        "<b>W</b> is the diagonal mask operator (1 for valid, 0 for masked), "
        "∇<b>u</b> is the 3D forward finite-difference gradient, and "
        "λ is the regularisation parameter. This is solved by the "
        "Chambolle–Pock primal-dual algorithm with step sizes "
        "τσ = 1/6 and projection onto the ℓ∞ ball. "
        "TV preserves piecewise-smooth structures (sharp diffuse sheets and streaks) "
        "while suppressing noise in the filled region."
    ))

    # 4.4 3D-DPDF
    story.append(PageBreak())
    story.append(H2("4.4  Stage 4: 3D-ΔPDF Transform"))

    story.append(H3("4.4.1  Correct FFT Recipe"))
    story.append(P(
        "The cleaned volume stores <b>Q</b>=0 at the <i>array centre</i> (index "
        "n//2), but NumPy’s <tt>fftn</tt> expects the origin at index [0,0,0]. "
        "A correct, centred Fourier transform of the real, centrosymmetric "
        "I(<b>Q</b>) must therefore be:"
    ))
    story.append(math_block(
        "Δρ = fftshift( fftn( ifftshift( I_windowed ) ) ).real"
    ))
    story.append(P(
        "Applied step by step in <tt>nebula3d/analysis/delta_pdf.py</tt>:"
    ))
    fft_steps = [
        ("Fill masked voxels",
         "Replace NaN (masked) voxels with 0. The backfilled volume should already be "
         "NaN-free; this is a safety guard."),
        ("Optional Q-space crop",
         "Symmetrically crop to |H| ≤ h_max, |K| ≤ k_max, |L| ≤ l_max, "
         "keeping the array centred on Q=0."),
        ("Subtract smooth background",
         "Compute I_bg = GaussianBlur(I, σ) with σ ≈ 1.5 r.l.u. and subtract: "
         "I_new = I − I_bg. This removes the broad separable diffuse envelope that "
         "would otherwise produce an axis cross in real space (see Section 4.4.3)."),
        ("Apodize",
         "Multiply by a separable window W(H,K,L) = w(H)×w(K)×w(L). "
         "The Hann window w(x) = cos²(πx/(2x_max)) (default) suppresses "
         "termination ripples from the finite |Q| range. "
         "Alternatives: Gaussian with width gaussian_sigma × Q_max, or no window."),
        ("Remove DC",
         "Subtract the mean <i>after</i> windowing: I_w −= mean(I_w). "
         "This ensures Σ I = 0 exactly, zeroing the r=0 self-correlation spike. "
         "(Subtracting before windowing leaves a nonzero windowed sum, creating a "
         "spurious large peak at r=0.)"),
        ("Zero-pad symmetrically",
         "Pad to the next fast FFT length (5-smooth) in each dimension, keeping Q=0 on the new centre. "
         "One-sided padding shifts the origin and breaks the ifftshift below. "
         "Zero-padding provides sinc-interpolation of the real-space grid; it does not "
         "increase intrinsic resolution, which is set by the Q_max and apodization window."),
        ("Transform",
         "ifftshift → fftn → fftshift: move Q=0 to the corner, transform, "
         "then recentre r=0."),
        ("Take real part",
         "The transform of centrosymmetric data I(Q)=I(−Q) is real; "
         "the imaginary part is numerical noise and a useful diagnostic."),
    ]
    for i, (s, d) in enumerate(fft_steps, 1):
        story.append(B(f"<b>Step {i}: {s}.</b>  {d}"))

    story.append(H3("4.4.2  Real-Space Axes"))
    story.append(P(
        "The real-space coordinate along axis <i>a</i> is computed from the FFT "
        "frequency grid and the UB matrix:"
    ))
    story.append(math_block(
        "Δh = (h_max − h_min) / (n_h − 1)"
    ))
    story.append(math_block(
        "freq_h = fftshift( fftfreq(n_h_padded, d=Δh) )   [cycles per HKL unit]"
    ))
    story.append(math_block(
        "x_a = freq_h × 2π / |UB[:,0]|   [Å]"
    ))
    story.append(P(
        "The factor 2π/|<b>a</b>*| converts cycles per HKL unit to Å in "
        "direct space, where <b>a</b>* = UB[:,0] is the first column of the UB matrix "
        "(the reciprocal basis vector along H)."
    ))

    story.append(H3("4.4.3  The Axis-Cross Artifact and Its Removal"))
    story.append(P(
        "A common artifact in 3D-ΔPDF maps is a bright cross along the "
        "<i>y</i><sub>K</sub>=0 and <i>z</i><sub>L</sub>=0 axes in the "
        "<i>y</i><sub>K</sub>–<i>z</i><sub>L</sub> plane. Diagnosis showed:"
    ))
    for d in [
        "The cross is present on planes with <i>no</i> Bragg peaks (e.g. H=1/3).",
        "The input has 0% masked voxels along the axis lines.",
        "Replacing the exact K=0 or L=0 input lines with neighbour averages has no effect.",
    ]:
        story.append(B(d))
    story.append(P(
        "The root cause is a broad, slowly-varying <i>diffuse envelope</i> centred "
        "near K=L=0. This envelope is approximately separable "
        "(I ≈ f(K) + g(L)), and the Fourier transform of a separable function "
        "concentrates on the two principal axes. The Hann window, itself a centred "
        "separable hump, multiplies this effect. Subtraction of the scalar mean "
        "only removes the DC term; it leaves the envelope shape untouched."
    ))
    story.append(P(
        "The fix is smooth-background subtraction before windowing. Using a "
        "Gaussian blur with σ ≈ 1.5 r.l.u. (per-axis control available):"
    ))
    story.append(math_block(
        "I_new = I − GaussianBlur(I, σ)"
    ))
    story.append(P(
        "This removes the separable envelope while preserving the oscillatory "
        "modulation (the genuine diffuse structure). As with all ΔPDF background "
        "subtractions, genuine very-long-period / low-<i>r</i> correlations at the "
        "same length scale as the background cannot be cleanly separated and are "
        "also attenuated."
    ))
    story.append(info_box(
        "Parameter recommendation",
        "Use SUBTRACT_BG=\"0,1.5,1.5\" to apply a per-axis Gaussian blur with "
        "σ_H=0 (slice-wise background, preserves H-layering), "
        "σ_K=1.5, σ_L=1.5 r.l.u. Tune the blur widths for the length scale "
        "of the smooth background in your volume."
    ))

    story.append(H3("4.4.4  Near-Origin Spike"))
    story.append(P(
        "A strong feature at r &lt; ~3 Å remains in all 3D-ΔPDF maps. It "
        "originates from residual high-|Q| Bragg leakage, discontinuities at the "
        "punch-hole boundaries, and the direct-beam punch. Colour scales are "
        "therefore set from the p99 of |Δρ| at r &gt; 3 Å so this "
        "near-origin spike does not dominate the display."
    ))

    story.append(H3("4.4.5  The Centring Bug (Fixed 2026-06-05)"))
    story.append(P(
        "Earlier code computed <tt>fftshift(fftn(data))</tt> without "
        "<tt>ifftshift</tt> and used one-sided zero-padding. With Q=0 at the array "
        "centre, the missing <tt>ifftshift</tt> introduces a linear phase ramp "
        "e<sup>−iπk</sup> = (−1)<sup>k</sup> across the output. "
        "Taking the real part then <i>flips the sign of real-space features by pixel "
        "parity</i>, so each correlation peak splits into mixed positive/negative lobes. "
        "This has been fixed in all current code; the regression test "
        "<tt>test_delta_pdf_centring_positive_peak</tt> guards against reintroduction."
    ))

    # -----------------------------------------------------------------------
    # 5  INSTALLATION
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("5.  Installation and Setup"))

    story.append(H2("5.1  Requirements"))
    req_rows = [
        ["Python", "≥ 3.10", "Core language"],
        ["numpy", "≥ 1.24", "Array operations, FFT"],
        ["scipy", "≥ 1.10", "Interpolation, Gaussian filters"],
        ["h5py", "≥ 3.8", "HDF5 file I/O"],
        ["matplotlib", "≥ 3.7", "Visualization"],
    ]
    tbl = Table([["Package", "Version", "Purpose"]] + req_rows,
                colWidths=[3.5 * cm, 3 * cm, 10.5 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)

    story.append(H2("5.2  Install from Source"))
    story.append(code_block(
        "git clone https://github.com/user/nebula3d\n"
        "cd nebula3d\n"
        "pip install -e \".[dev]\""
    ))
    story.append(P(
        "The <tt>[dev]</tt> extra installs pytest, ruff, and mypy for the test "
        "suite and linters."
    ))

    story.append(H2("5.3  Recommended Runtime Environment"))
    story.append(P(
        "The recommended environment uses the <tt>sci-general</tt> conda "
        "environment with all dependencies pre-installed. Set the following "
        "shell variables before running any script:"
    ))
    story.append(code_block(
        "export PY=/opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python\n"
        "export PYTHONPATH=src\n"
        "export MPLCONFIGDIR=/private/tmp/nebula3d-mpl"
    ))
    story.append(P(
        "<tt>MPLCONFIGDIR</tt> keeps Matplotlib cache files outside the repository. "
        "If your Python 3.10+ environment already has the dependencies active, "
        "replace <tt>$PY</tt> with <tt>python3</tt>."
    ))

    story.append(H2("5.4  Verifying the Installation"))
    story.append(code_block(
        "PYTHONPATH=src python -c \"import nebula3d; print(nebula3d.__version__)\"\n"
        "PYTHONPATH=src python -m pytest -o addopts='' tests/"
    ))

    # -----------------------------------------------------------------------
    # 6  WORKFLOW
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("6.  End-to-End Workflow"))

    story.append(H2("6.1  One-Command Pipeline"))
    story.append(P(
        "The script <tt>examples/run_pipeline.py</tt> orchestrates all four stages "
        "automatically. It detects which output files already exist and skips those "
        "stages, enabling fast incremental reprocessing:"
    ))
    story.append(code_block(
        "PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\\n"
        "  python examples/run_pipeline.py"
    ))
    story.append(P("Key environment overrides:"))
    pipe_rows = [
        [Paragraph("<tt>DATA_FILE</tt>", styles["code"]),
         Paragraph("Path to input .nxs file", styles["body_left"])],
        [Paragraph("<tt>NO_VIEWER=1</tt>", styles["code"]),
         Paragraph("Skip GUI stages; stop after writing _delta_pdf.h5", styles["body_left"])],
        [Paragraph("<tt>FORCE=1</tt>", styles["code"]),
         Paragraph("Recompute every stage even if output exists", styles["body_left"])],
        [Paragraph("<tt>FORCE_FROM=rings|punch|backfill|pdf</tt>", styles["code"]),
         Paragraph("Recompute from the named stage onward", styles["body_left"])],
        [Paragraph("<tt>SLICE_AXIS=H|K|L</tt>", styles["code"]),
         Paragraph("Direction along which ring removal iterates planes (default H)", styles["body_left"])],
    ]
    story.append(two_col_table(pipe_rows,
                               header=[Paragraph("<b>Variable</b>", styles["body_left"]),
                                       Paragraph("<b>Effect</b>", styles["body_left"])],
                               col_widths=[5.5 * cm, 11.5 * cm]))

    story.append(H2("6.2  Stage-by-Stage Commands"))

    story.append(H3("Stage 1: Ring Removal"))
    story.append(code_block(
        "PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RING_PRESET=cc_on \\\n"
        "  python examples/remove_rings_3d.py"
    ))
    story.append(PL("Output: <tt>data/processed/*_ringremoved.h5</tt>"))
    story.append(PL("Use <tt>RING_PRESET=cc_off</tt> for more aggressive subtraction on noisy data."))

    story.append(H3("Stage 2: Bragg Punch"))
    story.append(code_block(
        "PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\\n"
        "  PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \\\n"
        "  INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 \\\n"
        "  INTEGER_H_GUARD=0.12 \\\n"
        "  SEARCH_EXCLUDE_H_FRACTIONS=0.3333,0.6667 \\\n"
        "  SEARCH_EXCLUDE_H_WIDTH=0.08 \\\n"
        "  python examples/punch_bragg_3d.py"
    ))
    story.append(PL("Output: <tt>data/processed/*_braggpunched.h5</tt>"))

    story.append(H3("Stage 3: Backfill"))
    story.append(code_block(
        "PYTHONPATH=src METHOD=q_shell \\\n"
        "  python examples/backfill_bragg_3d.py"
    ))
    story.append(PL("Output: <tt>data/processed/*_braggpunched_backfilled.h5</tt>"))

    story.append(H3("Stage 4: 3D-ΔPDF"))
    story.append(code_block(
        "PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\\n"
        "  SUBTRACT_BG=\"0,1.5,1.5\" CROP_H=4 CROP_K=8 CROP_L=15 \\\n"
        "  APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \\\n"
        "  python examples/delta_pdf.py"
    ))
    story.append(PL("Outputs: <tt>examples/_delta_pdf.h5</tt> and PNG central-cut images."))

    story.append(H2("6.3  Multi-Volume Workflow"))
    story.append(P(
        "Run the pipeline once per input volume by overriding <tt>DATA_FILE</tt>:"
    ))
    story.append(code_block(
        "# condition A\n"
        "NO_VIEWER=1 \\\n"
        "DATA_FILE=\"data/raw/condition_a_cc_sub_bkg.nxs\" \\\n"
        "python examples/run_pipeline.py\n"
        "\n"
        "# condition B\n"
        "NO_VIEWER=1 \\\n"
        "DATA_FILE=\"data/raw/condition_b_cc_sub_bkg.nxs\" \\\n"
        "python examples/run_pipeline.py\n"
        "\n"
        "# condition C\n"
        "NO_VIEWER=1 \\\n"
        "DATA_FILE=\"data/raw/condition_c_cc_sub_bkg.nxs\" \\\n"
        "python examples/run_pipeline.py"
    ))
    story.append(P(
        "To generate persistent per-condition ΔPDF files in "
        "<tt>data/processed/</tt> use <tt>OUT_FILE</tt> and <tt>PROC_FILE</tt>:"
    ))
    story.append(code_block(
        "PROC_FILE=\"data/processed/condition_a_backfilled.h5\" \\\n"
        "OUT_FILE=\"data/processed/condition_a_delta_pdf.h5\" \\\n"
        "SUBTRACT_BG=\"0,1.5,1.5\" CROP_H=4 CROP_K=8 CROP_L=15 \\\n"
        "APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \\\n"
        "python examples/delta_pdf.py"
    ))

    story.append(H2("6.4  Python API"))
    story.append(P(
        "All pipeline stages can be called programmatically:"
    ))
    story.append(code_block(
        "import nebula3d\n"
        "from nebula3d.analysis import BraggRemover, backfill_bragg, compute_delta_pdf\n"
        "\n"
        "# Load ring-removed volume\n"
        "vol = nebula3d.load(\"data/processed/sample_ringremoved.h5\")\n"
        "\n"
        "# Bragg punch\n"
        "remover = BraggRemover(\n"
        "    mode=\"both\",\n"
        "    punch_radii=(0.09, 0.12, 0.45),\n"
        "    min_intensity=0.8,\n"
        "    min_prominence=0.8,\n"
        "    integer_optimize_position=True,\n"
        "    integer_optimize_shape=True,\n"
        "    integer_h_guard_hkl=0.12,\n"
        "    integer_local_prominence_n_mad=8.0,\n"
        "    search_n_mad=4.0,\n"
        "    search_exclude_h_fractions=(1/3, 2/3),\n"
        "    search_exclude_h_half_width=0.08,\n"
        "    incident_beam_ellipsoid_radii_hkl=(0.15, 0.50, 1.00),\n"
        ")\n"
        "punched = remover.apply(vol)\n"
        "\n"
        "# Backfill\n"
        "filled = backfill_bragg(punched, method=\"q_shell\")\n"
        "\n"
        "# 3D-DeltaPDF\n"
        "dpdf = compute_delta_pdf(\n"
        "    filled,\n"
        "    apodization=\"gaussian\",\n"
        "    gaussian_sigma=0.4,\n"
        "    crop_hkl=(4, 8, 15),\n"
        "    subtract_smooth_bg=(0, 1.5, 1.5),\n"
        ")\n"
        "print(f\"DeltaPDF shape: {dpdf.data.shape}\")\n"
        "print(f\"Real-space range: x +/-{dpdf.x_axis.max():.1f} A\")"
    ))

    # -----------------------------------------------------------------------
    # 7  CONFIGURATION REFERENCE
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("7.  Configuration Reference"))

    story.append(H2("7.1  Ring Removal Parameters"))
    ring_params = [
        ["RING_PRESET", "str", "cc_on", "Preset: cc_on (conservative) or cc_off (aggressive)"],
        ["Q_STEP", "float", "0.02", "Radial bin width for profile computation (Å⁻¹)"],
        ["N_FOURIER", "int", "8", "Number of Fourier terms for T(φ) fit"],
        ["PROFILE_METHOD", "str", "trimmed_mean", "Bin statistic: trimmed_mean or median"],
        ["TEXTURE_Q_SMOOTH", "float", "0.1", "Smoothing σ for texture along |Q| (Å⁻¹)"],
        ["SLICE_AXIS", "str", "H", "Axis along which to iterate slices: H, K, or L"],
    ]
    story.append(param_table(
        [[Paragraph(p, styles["code"]),
          Paragraph(t, styles["body_left"]),
          Paragraph(d, styles["code"]),
          Paragraph(desc, styles["body_left"])]
         for p, t, d, desc in ring_params]
    ))

    story.append(H2("7.2  Bragg Punch Parameters"))
    punch_params = [
        ["MODE", "str", "both", "Detection: integer, search (auto), or both"],
        ["PUNCH_PRESET", "str", "cc_on", "Preset for punch radii and thresholds"],
        ["MIN_I", "float", "0.8", "Minimum intensity threshold for Bragg detection"],
        ["MIN_PROM", "float", "0.8", "Minimum local-median prominence"],
        ["INTEGER_FIT_POSITION", "0/1", "1", "Optimize punch centre to measured maximum"],
        ["INTEGER_FIT_SHAPE", "0/1", "1", "Fit anisotropic punch radii from data"],
        ["INTEGER_H_GUARD", "float", "0.12", "H-slab half-width for integer-node punches"],
        ["SEARCH_EXCLUDE_H_FRACTIONS", "str", "0.3333,0.6667", "Periodic H exclusions (fractional parts mod 1)"],
        ["SEARCH_EXCLUDE_H_WIDTH", "float", "0.08", "Half-width of search-exclusion slabs"],
        ["INCIDENT_ELLIPSOID_R_HKL", "str", "0.15,0.50,1.00", "Direct-beam punch ellipsoid semi-axes (HKL)"],
    ]
    story.append(param_table(
        [[Paragraph(p, styles["code"]),
          Paragraph(t, styles["body_left"]),
          Paragraph(d, styles["code"]),
          Paragraph(desc, styles["body_left"])]
         for p, t, d, desc in punch_params]
    ))

    story.append(H2("7.3  Backfill Parameters"))
    fill_params = [
        ["METHOD", "str", "q_shell", "Fill method: q_shell, local, tv, symmetry, symmetry+tv"],
        ["Q_SHELL_STEP", "float", "0.05", "Bin width for q-shell fill (Å⁻¹)"],
        ["Q_SHELL_MIN_COUNT", "int", "10", "Min valid voxels per bin; fall back to local if fewer"],
        ["TV_LAM", "float", "0.1", "TV regularisation λ (higher = smoother fill)"],
        ["TV_ITER", "int", "300", "Chambolle-Pock iterations"],
    ]
    story.append(param_table(
        [[Paragraph(p, styles["code"]),
          Paragraph(t, styles["body_left"]),
          Paragraph(d, styles["code"]),
          Paragraph(desc, styles["body_left"])]
         for p, t, d, desc in fill_params]
    ))

    story.append(H2("7.4  3D-ΔPDF Transform Parameters"))
    dpdf_params = [
        ["APODIZE", "str", "gaussian", "Window function: hann, gaussian, or none"],
        ["GAUSSIAN_SIGMA", "float", "0.4", "Gaussian window width as fraction of Q_max"],
        ["CROP_H / CROP_K / CROP_L", "float", "4 / 8 / 15", "Symmetric Q-crop limits in HKL"],
        ["SUBTRACT_BG", "str", "0,1.5,1.5", "Per-axis Gaussian blur σ for background subtraction (H,K,L r.l.u.)"],
        ["REAL_SPACE_ANGSTROM", "0/1", "1", "Output real-space axes in Å (1) or HKL units (0)"],
        ["OUT_FILE", "str", "examples/_delta_pdf.h5", "Output HDF5 path for the ΔPDF volume"],
    ]
    story.append(param_table(
        [[Paragraph(p, styles["code"]),
          Paragraph(t, styles["body_left"]),
          Paragraph(d, styles["code"]),
          Paragraph(desc, styles["body_left"])]
         for p, t, d, desc in dpdf_params]
    ))

    # -----------------------------------------------------------------------
    # 8  VISUALIZATION
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("8.  Visualization and Interactive Exploration"))

    story.append(H2("8.1  Visualization API Overview"))
    story.append(P(
        "All visualization functions live in <tt>nebula3d.visualization</tt> and follow "
        "a primitive-first design: each function accepts an <tt>HKLVolume</tt>, "
        "draws into a caller-supplied Matplotlib Axes or Figure, and returns it. "
        "This makes the same calls work in one-shot scripts, IPython sessions, "
        "Jupyter notebooks, and the interactive viewer scripts."
    ))
    story.append(code_block(
        "from nebula3d.visualization import (\n"
        "    extract_slice, plot_slice,\n"
        "    plot_radial_profile, plot_azimuthal_map,\n"
        "    plot_overview, SliceData,\n"
        ")"
    ))

    story.append(H2("8.2  plot_slice()"))
    story.append(P(
        "Two-dimensional intensity slice through an HKLVolume. The <tt>plane</tt> "
        "argument is read as <tt>(horizontal, vertical)</tt> and selects the two "
        "displayed axes; the remaining axis is cut at <tt>value</tt>:"
    ))
    plane_rows = [
        ["<tt>'kl'</tt> or <tt>'0kl'</tt>", "K", "L", "H"],
        ["<tt>'hl'</tt> or <tt>'h0l'</tt>", "H", "L", "K"],
        ["<tt>'hk'</tt> or <tt>'hk0'</tt>", "H", "K", "L"],
    ]
    tbl = Table(
        [["Plane", "x-axis", "y-axis", "Fixed (cut by value)"]] + plane_rows,
        colWidths=[4.5 * cm, 3 * cm, 3 * cm, 6.5 * cm],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(SP(4))
    story.append(code_block(
        "# Log-scale background ring view\n"
        "plot_slice(bkg, \"kl\", value=0.0, log_scale=True)\n"
        "\n"
        "# Exact fractional plane with manual colour limits\n"
        "plot_slice(data, \"hk\", value=0.3333, interp=True, vmin=0.0, vmax=0.4)"
    ))
    story.append(P(
        "<b>Off-grid interpolation.</b>  Pass <tt>interp=True</tt> to linearly "
        "interpolate between the two bracketing planes so the exact value is "
        "honoured. The interpolation is NaN-aware: where only one bracketing plane "
        "is valid, that value is used."
    ))

    story.append(H2("8.3  plot_radial_profile() and plot_azimuthal_map()"))
    story.append(code_block(
        "# Radial profile with Al(111) ring marker\n"
        "plot_radial_profile(data, mark_q=[2.69])\n"
        "\n"
        "# Azimuthal texture of the Al(111) ring\n"
        "plot_azimuthal_map(data, q_center=2.69)"
    ))
    story.append(P(
        "<tt>plot_radial_profile</tt> bins voxels by |<b>Q</b>| and plots the mean "
        "or median intensity. <tt>plot_azimuthal_map</tt> bins by azimuthal angle "
        "φ within a thin |<b>Q</b>| shell, showing the ring texture T(φ)."
    ))

    story.append(H2("8.4  plot_overview()"))
    story.append(code_block(
        "fig = plot_overview(data, log_scale=True)"
    ))
    story.append(P(
        "A 2×2 diagnostic figure: KL (H=0), HL (K=0), HK (L=0) slices and a "
        "radial profile. The first look at any new volume."
    ))

    story.append(H2("8.5  Interactive Viewers"))

    story.append(H3("Cleanup QA Viewer"))
    story.append(P(
        "Four-panel view (raw → ring-removed → Bragg-punched → backfilled) "
        "with H/K/L plane selector and cut-position slider. Use to verify that integer-H "
        "Bragg peaks are cleanly removed and fractional-H diffuse planes are preserved:"
    ))
    story.append(code_block(
        "PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\\n"
        "  PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \\\n"
        "  INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \\\n"
        "  SEARCH_EXCLUDE_H_FRACTIONS=0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \\\n"
        "  H_VALUE=0.3333 \\\n"
        "  python examples/explore_slice.py"
    ))

    story.append(H3("3D-ΔPDF Orthoslice Viewer (Recommended)"))
    story.append(P(
        "All three orthogonal real-space planes simultaneously "
        "(x<sub>H</sub>–y<sub>K</sub>, x<sub>H</sub>–z<sub>L</sub>, "
        "y<sub>K</sub>–z<sub>L</sub>) with independent cut sliders, a contrast "
        "multiplier, and a unit-cell gridline toggle:"
    ))
    story.append(code_block(
        "PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\\n"
        "  PDF_FILE=data/processed/condition_a_delta_pdf.h5 RMAX=50 \\\n"
        "  python examples/explore_delta_pdf_ortho.py"
    ))

    story.append(H3("Multi-Volume Comparison Grid"))
    story.append(P(
        "A 3×3 grid (rows = three related volumes, columns = three orthogonal "
        "cuts) with shared cut sliders and a single global colour scale so "
        "intensities are directly comparable:"
    ))
    story.append(code_block(
        "PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\\n"
        "  PDF_FILES=data/processed/condition_a_delta_pdf.h5,data/processed/condition_b_delta_pdf.h5,data/processed/condition_c_delta_pdf.h5 \\\n"
        "  PDF_LABELS=\"condition A,condition B,condition C\" \\\n"
        "  python examples/explore_delta_pdf_multi.py"
    ))

    # -----------------------------------------------------------------------
    # 9  EXAMPLES
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("9.  Worked Examples"))

    story.append(H2("9.1  Quick Inspection of a New Dataset"))
    story.append(P(
        "Load a raw volume and display the 2×2 diagnostic overview:"
    ))
    story.append(code_block(
        "import nebula3d\n"
        "from nebula3d.visualization import plot_overview\n"
        "\n"
        "vol = nebula3d.load(\"path/to/sample.nxs\")   # or .h5\n"
        "fig = plot_overview(vol, log_scale=True)\n"
        "fig.savefig(\"overview.png\", dpi=120)"
    ))

    story.append(H2("9.2  Locating and Profiling a Powder Ring"))
    story.append(code_block(
        "from nebula3d.visualization import plot_radial_profile, plot_azimuthal_map\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "# Step 1: identify ring positions from radial profile\n"
        "fig, ax = plt.subplots()\n"
        "plot_radial_profile(vol, ax=ax, mark_q=[2.69, 4.39, 5.07])\n"
        "plt.title(\"Radial profile: Al ring positions\")\n"
        "\n"
        "# Step 2: check azimuthal texture T(phi) at Al(111)\n"
        "plot_azimuthal_map(vol, q_center=2.69, q_width=0.05)\n"
        "plt.title(\"Al(111) ring texture\")"
    ))

    story.append(H2("9.3  Checking Bragg Punch Quality"))
    story.append(P(
        "Compare an integer-H plane (Bragg present) to a fractional-H diffuse plane:"
    ))
    story.append(code_block(
        "import nebula3d\n"
        "from nebula3d.visualization import plot_slice\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "punched = nebula3d.load(\"data/processed/..._braggpunched.h5\")\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
        "\n"
        "# Integer-H: check all Bragg holes are punched\n"
        "plot_slice(punched, \"kl\", value=0.0, ax=axes[0],\n"
        "           log_scale=True, title=\"H=0 (integer): Bragg punched\")\n"
        "\n"
        "# Fractional-H: check diffuse is preserved\n"
        "plot_slice(punched, \"kl\", value=0.3333, interp=True, ax=axes[1],\n"
        "           log_scale=True, title=\"H=1/3 (fractional): diffuse intact\")\n"
        "\n"
        "fig.tight_layout()\n"
        "fig.savefig(\"bragg_check.png\", dpi=110)"
    ))

    story.append(H2("9.4  Computing and Displaying the 3D-ΔPDF"))
    story.append(code_block(
        "import nebula3d\n"
        "from nebula3d.analysis import compute_delta_pdf\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "filled = nebula3d.load(\"data/processed/..._backfilled.h5\")\n"
        "\n"
        "dpdf = compute_delta_pdf(\n"
        "    filled,\n"
        "    apodization=\"gaussian\",\n"
        "    gaussian_sigma=0.4,\n"
        "    crop_hkl=(4, 8, 15),\n"
        "    subtract_smooth_bg=(0, 1.5, 1.5),\n"
        ")\n"
        "nebula3d.save(dpdf, \"output_delta_pdf.h5\")\n"
        "\n"
        "# Plot the central hk0 slice\n"
        "hk0 = dpdf.slice_hk0()   # h-k plane at l=0\n"
        "vmax = abs(hk0.data).max()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 6))\n"
        "im = ax.imshow(hk0.data.T, origin=\"lower\",\n"
        "               extent=[hk0.x_axis[0], hk0.x_axis[-1],\n"
        "                       hk0.y_axis[0], hk0.y_axis[-1]],\n"
        "               cmap=\"RdBu_r\", vmin=-vmax, vmax=vmax)\n"
        "ax.set_xlabel(r\"$x_H$ (Å)\")\n"
        "ax.set_ylabel(r\"$y_K$ (Å)\")\n"
        "fig.colorbar(im, ax=ax, label=\"Δρ (arb.)\")\n"
        "fig.savefig(\"delta_pdf_hk0.png\", dpi=150)"
    ))

    # -----------------------------------------------------------------------
    # 10  TESTING
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("10.  Testing"))

    story.append(H2("10.1  Running the Test Suite"))
    story.append(code_block(
        "# Full suite with coverage\n"
        "PYTHONPATH=src python -m pytest -o addopts='' tests/\n"
        "\n"
        "# Lint (ruff)\n"
        "python -m ruff check src/ tests/\n"
        "\n"
        "# Type check (mypy)\n"
        "python -m mypy src/nebula3d --ignore-missing-imports"
    ))
    story.append(P(
        "Note: The <tt>-o addopts=''</tt> flag is required because the project "
        "uses <tt>conda</tt> rather than a virtualenv; without it pytest may pick "
        "up unexpected plugins."
    ))

    story.append(H2("10.2  Key Regression Tests"))
    test_rows = [
        [Paragraph("<tt>test_delta_pdf_centring_positive_peak</tt>", styles["code"]),
         Paragraph("Guards the FFT centring fix: verifies a positive cosine input "
                    "produces a positive real-space peak (not a sign-flipped lobe).",
                    styles["body_left"])],
        [Paragraph("Ring subtraction (synthetic)", styles["body_left"]),
         Paragraph("Verifies that a known Gaussian ring on a flat background is "
                    "cleanly subtracted without affecting the flat region.", styles["body_left"])],
        [Paragraph("Bragg punch / search modes", styles["body_left"]),
         Paragraph("Checks that integer-node and search-mode masks are consistent "
                    "with expected punch radii and guard conditions.", styles["body_left"])],
        [Paragraph("Symmetry fill", styles["body_left"]),
         Paragraph("Verifies that inverse-variance-weighted symmetry averaging "
                    "correctly fills masked voxels using Laue equivalents.", styles["body_left"])],
        [Paragraph("TV inpainting convergence", styles["body_left"]),
         Paragraph("Checks that the Chambolle-Pock primal-dual solver converges "
                    "within the specified iteration budget.", styles["body_left"])],
    ]
    story.append(two_col_table(
        test_rows,
        header=[Paragraph("<b>Test</b>", styles["body_left"]),
                Paragraph("<b>What it verifies</b>", styles["body_left"])],
        col_widths=[5.5 * cm, 11.5 * cm],
    ))

    # -----------------------------------------------------------------------
    # 11  TROUBLESHOOTING
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("11.  Known Limitations and Troubleshooting"))

    story.append(H2("11.1  Near-Origin Spike"))
    story.append(P(
        "<b>Symptom.</b>  A very strong feature at r &lt; ~3 Å in all "
        "ΔPDF maps that dominates the colour scale."
    ))
    story.append(P(
        "<b>Cause.</b>  Residual high-|Q| Bragg leakage, discontinuities at "
        "punch-hole boundaries, and the direct-beam punch."
    ))
    story.append(P(
        "<b>Mitigation.</b>  Set colour scales from the p99 of |Δρ| at "
        "r &gt; 3 Å (this is the default in all viewer scripts). For "
        "quantitative analysis of short-range correlations, tapered punch boundaries "
        "and softer high-|Q| windows are planned for a future release."
    ))

    story.append(H2("11.2  Axis Cross in ΔPDF"))
    story.append(P(
        "<b>Symptom.</b>  A bright cross along y<sub>K</sub>=0 and z<sub>L</sub>=0 "
        "axes in the real-space map."
    ))
    story.append(P(
        "<b>Cause.</b>  Residual separable diffuse envelope (see Section 4.4.3)."
    ))
    story.append(P(
        "<b>Fix.</b>  Use <tt>SUBTRACT_BG=\"0,1.5,1.5\"</tt> (or the Python API "
        "parameter <tt>subtract_smooth_bg=(0, 1.5, 1.5)</tt>)."
    ))

    story.append(H2("11.3  Residual Ring After Subtraction"))
    story.append(P(
        "<b>Symptom.</b>  Faint ring pattern still visible after ring removal."
    ))
    story.append(P(
        "<b>Causes and mitigations.</b>"
    ))
    story.append(B("Gaussian width <tt>σ_i</tt> too narrow: widen via Q_STEP and re-run detection."))
    story.append(B("Texture mismatch (rank1_variance &lt; 0.90): check diagnostic and "
                   "consider per-ring T_i(φ) fitting (planned feature)."))
    story.append(B("Ring preset mismatch: try RING_PRESET=cc_off for noisier or "
                   "more strongly ringed data."))

    story.append(H2("11.4  Search Mode Punching Diffuse Structure"))
    story.append(P(
        "<b>Symptom.</b>  Structured diffuse on fractional-H planes is partially masked."
    ))
    story.append(P(
        "<b>Fix.</b>  Add the fractional H values to the periodic exclusion list: "
        "<tt>SEARCH_EXCLUDE_H_FRACTIONS=0.3333,0.6667</tt>. "
        "Use the periodic form rather than an explicit centre list to catch "
        "higher-order satellites automatically."
    ))

    story.append(H2("11.5  Rank-1 SVD Failure in Ring Model"))
    story.append(P(
        "<b>Symptom.</b>  <tt>rank1_variance</tt> &lt; 0.90 after ring fitting."
    ))
    story.append(P(
        "<b>Meaning.</b>  Different rings have significantly different azimuthal "
        "textures; the shared T(φ) model is insufficient."
    ))
    story.append(P(
        "<b>Current workaround.</b>  Use the non-parametric production path "
        "(<tt>PatchedRadialRingModel</tt>), which handles per-patch textures without "
        "assuming a shared T(φ). Per-ring T_i(φ) fitting is a planned improvement "
        "for the factored model."
    ))

    story.append(H2("11.6  Performance Notes"))
    perf_rows = [
        ["Ring removal (per-slice)", "~2–5 min", "Sequential per-H-slice; embarrassingly parallel"],
        ["Bragg punch (detection + mask)", "~1–2 min", "Scales with n_peaks"],
        ["Backfill (q-shell)", "~30 sec", "Linear in n_voxels"],
        ["3D-ΔPDF (FFT + windowing)", "~10 sec", "Dominated by FFT of zero-padded array"],
    ]
    tbl = Table(
        [["Operation", "Typical time", "Notes"]] + perf_rows,
        colWidths=[5 * cm, 3.5 * cm, 8.5 * cm],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(P(
        "Benchmarks are for a 401×401×301 volume (~48 M voxels) on a "
        "typical laptop. Ring removal is the bottleneck; it is embarrassingly "
        "parallel per H-slice and a future release will add optional "
        "<tt>concurrent.futures.ProcessPoolExecutor</tt> parallelisation."
    ))
    story.append(tbl)

    # -----------------------------------------------------------------------
    # 12  GLOSSARY
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("12.  Glossary"))

    glossary = [
        ("Apodization", "A window function applied to I(Q) before FFT to suppress "
         "termination ripples from the finite Q range. Common choices: Hann, Gaussian."),
        ("Backfill", "Replacement of masked (punched) voxels with estimated background "
         "intensities derived from surrounding valid voxels."),
        ("Biharmonic", "A fill based on iterative solution of ∇⁴u=0; produces "
         "very smooth interpolations."),
        ("Bragg peaks", "Sharp peaks at integer (and satellite) reciprocal-lattice nodes; "
         "encode the average periodic structure."),
        ("Centrosymmetric", "Having inversion symmetry I(Q)=I(−Q); ensures the "
         "Fourier transform is real-valued."),
        ("Diffuse scattering", "Broad, continuous intensity encoding structural disorder, "
         "short-range order, and dynamic correlations."),
        ("Fractional HKL", "Reciprocal-lattice coordinates (h,k,l) in units of "
         "a*, b*, c*; integers are Bragg positions."),
        ("Hann window", "A cosine-squared taper w(x)=cos²(πx/2x_max); "
         "smoothly suppresses the signal to zero at the Q boundary."),
        ("HKLVolume", "The central data structure: a 3D array plus axes, mask, σ, "
         "UB matrix, and instrument label."),
        ("Laue class", "Crystal point-group symmetry class (e.g. mmm, m3m, 4/mmm) "
         "determining equivalent Q positions."),
        ("MAD", "Median Absolute Deviation; a robust measure of spread: "
         "MAD = median(|x − median(x)|)."),
        ("Powder ring", "Azimuthally smooth band of intensity at fixed |Q| from "
         "polycrystalline material in the beam."),
        ("Punch", "Mask and remove a Bragg peak by setting a region of voxels invalid."),
        ("Q", "Cartesian momentum transfer vector in Å⁻¹; "
         "Q = UB·[h,k,l]ᵀ; |Q| = 2π/d (physics convention)."),
        ("SNIP", "Sensitive Nonlinear Iterative Peak clipping; a morphological method "
         "for baseline estimation in 1D profiles."),
        ("Total Variation (TV)", "Regulariser ||∇u||<sub>1</sub> promoting "
         "piecewise-smooth solutions; preserves sharp diffuse sheets."),
        ("UB matrix", "Orientation (U) × metric (B) matrix; Q = UB·[h,k,l]ᵀ."),
        ("3D-ΔPDF", "Three-dimensional difference pair distribution function; "
         "the Fourier transform of diffuse scattering, mapping to real-space pair correlations."),
    ]
    for term, definition in glossary:
        story.append(KeepTogether([
            P(f"<b>{term}.</b>  {definition}"),
        ]))

    # -----------------------------------------------------------------------
    # 13  REFERENCES
    # -----------------------------------------------------------------------
    story.append(PageBreak())
    story.append(H1("13.  References"))
    story.append(SP(4))

    refs = [
        ("[1]",
         "T. Weber and A. Simonov, "
         "&ldquo;The three-dimensional pair distribution function analysis of "
         "disordered single crystals: basic concepts,&rdquo; "
         "<i>Z. Kristallogr.</i> <b>227</b>, 238–247 (2012). "
         "DOI: 10.1524/zkri.2012.1504. "
         "<b>Establishes the 3D-ΔPDF formalism.</b>"),
        ("[2]",
         "A. Simonov, T. Weber, and W. Steurer, "
         "&ldquo;Diffuse scattering from the disordered intermetallic compound "
         "TbFe<sub>2</sub>(Si<sub>1-x</sub>Al<sub>x</sub>)<sub>4</sub> (0 ≤ x ≤ 0.67),&rdquo; "
         "<i>J. Appl. Cryst.</i> <b>47</b>, 2011–2018 (2014). "
         "DOI: 10.1107/S1600576714023668. "
         "<b>3D-ΔPDF punch-and-fill strategy.</b>"),
        ("[3]",
         "A. Chambolle and T. Pock, "
         "&ldquo;A First-Order Primal-Dual Algorithm for Convex Problems with "
         "Applications to Imaging,&rdquo; "
         "<i>J. Math. Imaging Vision</i> <b>40</b>, 120–145 (2011). "
         "DOI: 10.1007/s10851-010-0251-1. "
         "<b>Primal-dual TV inpainting algorithm used in nebula3d.</b>"),
        ("[4]",
         "M. Bertalmio, G. Sapiro, V. Caselles, and C. Ballester, "
         "&ldquo;Image inpainting,&rdquo; "
         "<i>Proc. ACM SIGGRAPH</i>, 417–424 (2000). "
         "DOI: 10.1145/344779.344972. "
         "<b>Origin of PDE-based diffusion inpainting.</b>"),
        ("[5]",
         "M. Bertero and P. Boccacci, "
         "<i>Introduction to Inverse Problems in Imaging</i>, "
         "IOP Publishing, Bristol, UK (1998). ISBN: 978-0750304351. "
         "<b>General theory of regularised inverse problems.</b>"),
        ("[6]",
         "Mantid Project, "
         "&ldquo;Mantid — Manipulation and Analysis Toolkit for Instrument Data,&rdquo; "
         "<i>Mantid</i> (2013–present). "
         "www.mantidproject.org. "
         "<b>Data reduction and MDHistoWorkspace I/O.</b>"),
        ("[7]",
         "M. Ryan and D. Giannakis, "
         "&ldquo;Sensitive nonlinear iterative peak-clipping algorithm (SNIP) for "
         "background estimation in spectrometry,&rdquo; "
         "<i>Nucl. Instrum. Methods Phys. Res. B</i> <b>34</b>, 396–402 (1988). "
         "<b>SNIP baseline algorithm used in radial profile processing.</b>"),
    ]

    for ref_id, ref_text in refs:
        data = [[Paragraph(ref_id, styles["body_left"]),
                 Paragraph(ref_text, styles["body"])]]
        ref_tbl = Table(data, colWidths=[1.2 * cm, PAGE_W - 5.5 * cm - 1.2 * cm])
        ref_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(ref_tbl)
        story.append(SP(2))

    # -----------------------------------------------------------------------
    # BUILD
    # -----------------------------------------------------------------------
    import copy

    # ---- Stash a deep copy of the fresh story BEFORE pass 1 mutates it ----
    story_pass2 = copy.deepcopy(story)   # fresh objects for pass 2

    # ---- Pass 1: collect heading page numbers ----
    col_doc = _CollectorDoc(**_DOC_KW)
    col_doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    # ---- Estimate how many extra pages the actual TOC will add ----
    # Spacer(1,0) in pass 1 takes 0 extra pages.  Compute actual TOC height:
    toc_height_pt = sum(20 if lvl == 0 else 18 for lvl, _, _ in col_doc.toc_entries) + 8
    usable_h = PAGE_H - _DOC_KW["topMargin"] - _DOC_KW["bottomMargin"] - 14
    n_toc_pages = max(0, int(toc_height_pt / usable_h))  # extra pages TOC adds

    # ---- Build static TOC table with adjusted page numbers ----
    toc_table = _build_static_toc(col_doc.toc_entries, page_offset=n_toc_pages)
    story_pass2[_toc_story_idx] = toc_table

    # ---- Pass 2: final build using fresh story copy ----
    real_doc = SimpleDocTemplate(output_path, **_DOC_KW)
    real_doc.build(story_pass2, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"Manual written to: {output_path}")


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "nebula3d_manual.pdf")
    build_manual(out)
