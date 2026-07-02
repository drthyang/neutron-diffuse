// build_manual.js — generates the nebula3d manual as a styled .docx
// Optimized for import into Google Docs (File ▸ Import / drag-drop), then
// Download ▸ PDF for distribution.
//
//   node docs/build_manual.js
//
// Design system lives at the top; content sections (sec1..sec13) below.

const GLOBAL_NODE_MODULES = "/Users/thyang/.nvm/versions/node/v24.16.0/lib/node_modules";
let docx;
try {
  docx = require("docx");
} catch {
  docx = require(`${GLOBAL_NODE_MODULES}/docx`);
}
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, TableOfContents,
  LevelFormat, TabStopType,
} = docx;
const fs = require("fs");
const path = require("path");

// ═══════════════════════════════════════════════════════════════════════════
//  DESIGN TOKENS
// ═══════════════════════════════════════════════════════════════════════════

// Page (A4, narrow side margins for a manual feel)
const PAGE_W = 11906, PAGE_H = 16838;
const MX = 1418, MT = 1500, MB = 1320;
const CONTENT_W = PAGE_W - 2 * MX;          // 9070 DXA ≈ 16.0 cm

// Palette
const INK   = "1C1F26";   // body text
const NAVY  = "112A4E";   // primary
const BLUE  = "1E4E8C";   // secondary
const STEEL = "5A739B";   // tertiary / muted headings
const GOLD  = "BC8A1E";   // accent
const TEAL  = "0E7C6B";
const GREEN = "1C7A4E";
const AMBER = "B07415";
const PLUM  = "6A4A93";
const MUTED = "5C6470";

const WHITE = "FFFFFF";
const RULE  = "DCE2EC";   // hairline borders
const ZEBRA = "F4F7FB";   // alt row
const CODE_BG = "F4F6FA";
const BAND  = "EAF1FB";   // soft blue band (equations)
const NOTE_BG = "EAF2FB", TIP_BG = "E9F6EF", WARN_BG = "FBF2E2", KEY_BG = "F2EDFA";

// Type
const SANS = "Arial", MONO = "Courier New", SERIF = "Georgia";

// ═══════════════════════════════════════════════════════════════════════════
//  LOW-LEVEL HELPERS
// ═══════════════════════════════════════════════════════════════════════════

function run(text, o = {}) {
  return new TextRun({
    text, font: o.font || SANS, size: o.size || 21,
    bold: o.bold, italic: o.italic, color: o.color || INK,
    allCaps: o.caps, smallCaps: o.smallCaps,
    superScript: o.sup, subScript: o.sub, characterSpacing: o.tracking,
    underline: o.underline,
  });
}

function toRuns(content, base = {}) {
  if (typeof content === "string") return [run(content, base)];
  if (Array.isArray(content))
    return content.map(c => (c instanceof TextRun ? c : run(c.text, { ...base, ...c })));
  return [content];
}

/** Body / generic paragraph */
function p(content, o = {}) {
  return new Paragraph({
    children: toRuns(content, { size: o.size, color: o.color, italic: o.italic, bold: o.bold }),
    alignment: o.align,
    indent: o.indent,
    spacing: { before: o.before ?? 0, after: o.after ?? 150, line: o.line ?? 278, lineRule: "auto" },
    pageBreakBefore: o.pageBreak,
    keepNext: o.keepNext,
    border: o.border,
  });
}

/** Section-opening lead sentence (larger, muted) */
function lead(text) {
  return p(text, { size: 24, color: STEEL, italic: true, after: 200, line: 300 });
}

/** Small italic caption */
function caption(text) {
  return p(text, { size: 17, color: MUTED, italic: true, after: 140 });
}

/** Near-empty vertical spacer */
function gap(after = 90) {
  return new Paragraph({ spacing: { before: 0, after }, children: [new TextRun({ text: "", size: 2 })] });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

// Borders
const NB = { style: BorderStyle.NONE, size: 0, color: WHITE };
function bAll(color, size = 2) { const b = { style: BorderStyle.SINGLE, size, color }; return { top: b, bottom: b, left: b, right: b }; }

// ═══════════════════════════════════════════════════════════════════════════
//  HEADINGS  (real Heading styles → TOC works after refresh in Google Docs)
// ═══════════════════════════════════════════════════════════════════════════

function h1(num, title) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    pageBreakBefore: true,
    children: [
      run(String(num), { size: 38, bold: true, color: GOLD }),
      run("  ", { size: 38 }),
      run(title, { size: 38, bold: true, color: NAVY }),
    ],
  });
}
function h2(num, title) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [
      run(num + " ", { size: 28, bold: true, color: STEEL }),
      run(title, { size: 28, bold: true, color: BLUE }),
    ],
  });
}
function h3(title) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [run(title, { size: 23, bold: true, italic: true, color: STEEL })],
  });
}

// ═══════════════════════════════════════════════════════════════════════════
//  LISTS
// ═══════════════════════════════════════════════════════════════════════════

function li(content, o = {}) {
  return new Paragraph({
    numbering: { reference: o.ref || "bul", level: o.level || 0 },
    spacing: { before: 24, after: 70, line: 268, lineRule: "auto" },
    children: toRuns(content),
  });
}

// ═══════════════════════════════════════════════════════════════════════════
//  CODE CARD  (single-cell table, accent left bar, continuous background)
// ═══════════════════════════════════════════════════════════════════════════

function code(text, label) {
  const inner = text.split("\n").map(line =>
    new Paragraph({
      spacing: { before: 0, after: 0, line: 248, lineRule: "auto" },
      children: [run(line.length ? line : " ", { font: MONO, size: 18, color: "16213A" })],
    }));
  const cell = new TableCell({
    width: { size: CONTENT_W, type: WidthType.DXA },
    shading: { fill: CODE_BG, type: ShadingType.CLEAR },
    margins: { top: 150, bottom: 150, left: 220, right: 170 },
    borders: {
      left:   { style: BorderStyle.SINGLE, size: 26, color: BLUE },
      top:    { style: BorderStyle.SINGLE, size: 3, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 3, color: RULE },
      right:  { style: BorderStyle.SINGLE, size: 3, color: RULE },
    },
    children: inner,
  });
  const out = [];
  if (label)
    out.push(p([{ text: label, caps: true, bold: true, size: 15, color: STEEL, tracking: 36 }],
               { before: 40, after: 46 }));
  out.push(new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [cell] })],
  }));
  out.push(gap(110));
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  CALLOUT  (single cell, thick colored left bar, tinted background)
// ═══════════════════════════════════════════════════════════════════════════

function callout(kind, title, body) {
  const M = {
    note: { bg: NOTE_BG, bar: BLUE,  label: "NOTE" },
    tip:  { bg: TIP_BG,  bar: GREEN, label: "TIP" },
    warn: { bg: WARN_BG, bar: AMBER, label: "IMPORTANT" },
    key:  { bg: KEY_BG,  bar: PLUM,  label: "KEY IDEA" },
  };
  const c = M[kind] || M.note;
  const kids = [];
  const head = [run(c.label, { caps: true, bold: true, size: 17, color: c.bar, tracking: 34 })];
  if (title) head.push(run(" — " + title, { bold: true, size: 20, color: NAVY }));
  kids.push(new Paragraph({ spacing: { before: 0, after: body ? 70 : 0 }, children: head }));
  if (body) {
    const paras = Array.isArray(body) && typeof body[0] !== "object"
      ? [body] : (Array.isArray(body) ? body : [body]);
    // normalize: allow string | runs[] | array-of-(string|runs[])
    const items = Array.isArray(body) && (typeof body[0] === "string" || body[0] instanceof TextRun || (body[0] && body[0].text))
      ? [body] : (Array.isArray(body) ? body : [body]);
    items.forEach((t, i) =>
      kids.push(new Paragraph({
        spacing: { before: 0, after: i === items.length - 1 ? 0 : 60, line: 272, lineRule: "auto" },
        children: toRuns(t, { size: 20 }),
      })));
  }
  const cell = new TableCell({
    width: { size: CONTENT_W, type: WidthType.DXA },
    shading: { fill: c.bg, type: ShadingType.CLEAR },
    margins: { top: 160, bottom: 160, left: 230, right: 210 },
    borders: {
      left:   { style: BorderStyle.SINGLE, size: 30, color: c.bar },
      top:    { style: BorderStyle.SINGLE, size: 2, color: c.bg },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: c.bg },
      right:  { style: BorderStyle.SINGLE, size: 2, color: c.bg },
    },
    children: kids,
  });
  return [new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [cell] })],
  }), gap(120)];
}

// ═══════════════════════════════════════════════════════════════════════════
//  DISPLAY EQUATION  (centered, soft blue band)
// ═══════════════════════════════════════════════════════════════════════════

function math(lines) {
  const arr = Array.isArray(lines) ? lines : [lines];
  const kids = arr.map((l, i) =>
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: i ? 50 : 0, after: 0, line: 268, lineRule: "auto" },
      children: [run(l, { font: SERIF, italic: true, size: 25, color: NAVY })],
    }));
  const cell = new TableCell({
    width: { size: CONTENT_W, type: WidthType.DXA },
    shading: { fill: BAND, type: ShadingType.CLEAR },
    margins: { top: 170, bottom: 170, left: 220, right: 220 },
    borders: bAll(BAND, 2),
    children: kids,
  });
  return [new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [cell] })],
  }), gap(110)];
}

// ═══════════════════════════════════════════════════════════════════════════
//  DATA TABLE  (navy header, zebra rows, optional mono columns)
// ═══════════════════════════════════════════════════════════════════════════

function dataTable({ head, rows, widths, mono = [], align = [], after = 150 }) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headRow = new TableRow({
    tableHeader: true,
    children: head.map((t, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { fill: NAVY, type: ShadingType.CLEAR },
      margins: { top: 95, bottom: 95, left: 140, right: 140 },
      verticalAlign: VerticalAlign.CENTER,
      borders: bAll(NAVY, 2),
      children: [new Paragraph({ alignment: align[i], spacing: { after: 0 },
        children: [run(t, { bold: true, color: WHITE, size: 19 })] })],
    })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, ci) => {
      const fill = ri % 2 === 0 ? WHITE : ZEBRA;
      const isMono = mono.includes(ci);
      let kids;
      if (c instanceof Paragraph) kids = [c];
      else if (Array.isArray(c))
        kids = [new Paragraph({ spacing: { after: 0, line: 256, lineRule: "auto" }, alignment: align[ci],
          children: c.map(x => (x instanceof TextRun ? x : run(x.text, x))) })];
      else
        kids = [new Paragraph({ spacing: { after: 0, line: 256, lineRule: "auto" }, alignment: align[ci],
          children: [run(String(c), { size: 18, font: isMono ? MONO : SANS, color: isMono ? "16213A" : INK })] })];
      return new TableCell({
        width: { size: widths[ci], type: WidthType.DXA },
        shading: { fill, type: ShadingType.CLEAR },
        margins: { top: 82, bottom: 82, left: 140, right: 140 },
        verticalAlign: VerticalAlign.TOP, borders: bAll(RULE, 2), children: kids,
      });
    }),
  }));
  return [new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [headRow, ...bodyRows] }), gap(after)];
}

// ═══════════════════════════════════════════════════════════════════════════
//  STAGE CARD  (numbered process flow)
// ═══════════════════════════════════════════════════════════════════════════

function stageCard(num, title, desc, accent) {
  const numCell = new TableCell({
    width: { size: 920, type: WidthType.DXA },
    shading: { fill: accent, type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 130, bottom: 130, left: 60, right: 60 },
    borders: bAll(accent, 2),
    children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: [run(String(num), { bold: true, color: WHITE, size: 44 })] })],
  });
  const txtCell = new TableCell({
    width: { size: CONTENT_W - 920, type: WidthType.DXA },
    shading: { fill: ZEBRA, type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 120, bottom: 120, left: 220, right: 190 },
    borders: { top: { style: BorderStyle.SINGLE, size: 2, color: RULE },
               bottom: { style: BorderStyle.SINGLE, size: 2, color: RULE },
               right: { style: BorderStyle.SINGLE, size: 2, color: RULE }, left: NB },
    children: [
      new Paragraph({ spacing: { after: 46 }, children: [run(title, { bold: true, size: 23, color: NAVY })] }),
      new Paragraph({ spacing: { after: 0, line: 262, lineRule: "auto" }, children: [run(desc, { size: 19, color: INK })] }),
    ],
  });
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [920, CONTENT_W - 920],
    rows: [new TableRow({ children: [numCell, txtCell] })] });
}
function stageFlow(stages) {
  const out = [];
  stages.forEach(s => { out.push(stageCard(s[0], s[1], s[2], s[3])); out.push(gap(70)); });
  out.push(gap(60));
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  COVER
// ═══════════════════════════════════════════════════════════════════════════

function hero() {
  const cell = new TableCell({
    width: { size: CONTENT_W, type: WidthType.DXA },
    shading: { fill: NAVY, type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 820, bottom: 820, left: 520, right: 520 },
    borders: bAll(NAVY, 2),
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 150 },
        children: [run("NEUTRON SCATTERING ANALYSIS TOOLKIT", { color: "9DB7DD", size: 18, bold: true, tracking: 48 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
        children: [run("nebula3d", { color: WHITE, size: 100, bold: true })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 190 },
        indent: { left: 2700, right: 2700 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 20, color: GOLD, space: 1 } },
        children: [run(" ", { size: 6 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
        children: [run("User & Reference Manual", { color: "DCE7F6", size: 32, tracking: 24 })] }),
    ],
  });
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [cell] })] });
}

function metaRow(label, valueRuns, L, R) {
  return new TableRow({ children: [
    new TableCell({ width: { size: L, type: WidthType.DXA }, shading: { fill: ZEBRA, type: ShadingType.CLEAR },
      margins: { top: 78, bottom: 78, left: 150, right: 150 }, borders: bAll(RULE, 2),
      children: [new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 0 },
        children: [run(label, { bold: true, color: NAVY, size: 18 })] })] }),
    new TableCell({ width: { size: R, type: WidthType.DXA }, shading: { fill: WHITE, type: ShadingType.CLEAR },
      margins: { top: 78, bottom: 78, left: 170, right: 150 }, borders: bAll(RULE, 2),
      children: [new Paragraph({ spacing: { after: 0 }, children: valueRuns })] }),
  ] });
}

function coverMeta() {
  const W = 6600, L = 2500, R = W - L;
  const rows = [
    metaRow("Language", [run("Python ≥ 3.10", { size: 19 })], L, R),
    metaRow("License", [run("MIT", { size: 19 })], L, R),
    metaRow("Core stack", [run("numpy · scipy · h5py · matplotlib", { size: 19 })], L, R),
    metaRow("Input format", [run("Mantid NeXus (MDHistoWorkspace)", { size: 19 })], L, R),
    metaRow("Reference data", [run("Mantid NeXus HKL volumes", { size: 19 })], L, R),
  ];
  return new Table({ alignment: AlignmentType.CENTER, width: { size: W, type: WidthType.DXA },
    columnWidths: [L, R], rows });
}

function cover() {
  return [
    gap(620),
    hero(),
    gap(300),
    new Paragraph({ alignment: AlignmentType.CENTER, indent: { left: 520, right: 520 },
      spacing: { after: 170, line: 300, lineRule: "auto" },
      children: [run(
        "Three-dimensional reciprocal-space diffuse neutron scattering analysis — powder ring removal, Bragg punch-and-fill, and the 3D-ΔPDF transform, in one reproducible Python pipeline.",
        { size: 23, color: STEEL, italic: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
      children: [run("Version 0.1.0", { bold: true, color: NAVY, size: 21 }),
                 run("        •        ", { bold: true, color: GOLD, size: 21 }),
                 run("June 2026", { bold: true, color: NAVY, size: 21 })] }),
    gap(260),
    coverMeta(),
    gap(470),
    new Paragraph({ alignment: AlignmentType.CENTER, indent: { left: 2000, right: 2000 },
      spacing: { after: 80 }, border: { top: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 8 } },
      children: [run(" ", { size: 4 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 36 },
      children: [run("Tsung-Han Yang", { bold: true, color: NAVY, size: 25 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: [run("© 2026 Tsung-Han Yang.  All rights reserved.", { color: MUTED, size: 18 })] }),
    pageBreak(),
  ];
}

function toc() {
  return [
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [run("Contents", { size: 38, bold: true, color: NAVY })] }),
    gap(40),
    caption("If the list below is blank after importing into Google Docs, click it and choose “Update table of contents”, or use Insert ▸ Table of contents."),
    gap(40),
    new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  ];
}

// ═══════════════════════════════════════════════════════════════════════════
//  HEADER / FOOTER
// ═══════════════════════════════════════════════════════════════════════════

function pageHeader() {
  return new Header({ children: [new Paragraph({
    spacing: { after: 0 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: NAVY, space: 5 } },
    tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W }],
    children: [
      run("nebula3d", { bold: true, color: NAVY, size: 16 }),
      run("  ·  User & Reference Manual", { color: STEEL, size: 16 }),
      new TextRun({ text: "\t" }),
      run("v0.1.0", { color: STEEL, size: 16 }),
    ],
  })] });
}

function pageFooter() {
  return new Footer({ children: [new Paragraph({
    spacing: { before: 0 },
    border: { top: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 6 } },
    tabStops: [{ type: TabStopType.CENTER, position: Math.floor(CONTENT_W / 2) },
               { type: TabStopType.RIGHT, position: CONTENT_W }],
    children: [
      run("© 2026 Tsung-Han Yang", { color: MUTED, size: 15 }),
      new TextRun({ text: "\t" }),
      run("Page ", { color: MUTED, size: 15 }),
      new TextRun({ children: [PageNumber.CURRENT], color: MUTED, size: 15, font: SANS }),
      run(" of ", { color: MUTED, size: 15 }),
      new TextRun({ children: [PageNumber.TOTAL_PAGES], color: MUTED, size: 15, font: SANS }),
      new TextRun({ text: "\t" }),
      run("nebula3d", { bold: true, color: STEEL, size: 15 }),
    ],
  })] });
}
const emptyHeader = new Header({ children: [new Paragraph({ children: [] })] });
const emptyFooter = new Footer({ children: [new Paragraph({ children: [] })] });

// ═══════════════════════════════════════════════════════════════════════════
//  CONTENT — SECTION 1  Introduction
// ═══════════════════════════════════════════════════════════════════════════

function sec1() {
  return [
    h1(1, "Introduction"),
    lead("From a raw single-crystal neutron volume to a real-space map of atomic correlations — one structure, four stages, fully reproducible."),
    p([{ text: "nebula3d", bold: true }, { text: " is a Python 3.10+ toolkit for the complete analysis of three-dimensional (3D) diffuse neutron scattering volumes. It implements a production-ready four-stage pipeline that transforms a Mantid NeXus volume into a real-space 3D difference pair distribution function (3D-ΔPDF)." }]),

    h2("1.1", "The Pipeline at a Glance"),
    ...stageFlow([
      [1, "Powder ring removal", "Subtracts azimuthally smooth powder-ring contamination from the cryostat and sample environment without masking genuine diffuse signal.", "153A66"],
      [2, "Bragg punch", "Masks sharp Bragg and satellite peaks using lattice-aware integer-node detection plus an hkl-agnostic search pass for off-integer satellites.", "1E4E8C"],
      [3, "Backfill", "Replaces masked voxels with physically reasonable estimates from the surrounding radial background or crystal-symmetry equivalents.", "2C6E8F"],
      [4, "3D-ΔPDF transform", "Converts the cleaned reciprocal-space volume to a real-space pair-correlation map via a correctly centred 3D DFT with apodization and smooth-background subtraction.", "0E7C6B"],
    ]),
    ...callout("key", "One structure carries everything",
      [[{ text: "Every stage reads and writes a single " }, { text: "HKLVolume", font: MONO, size: 19 },
        { text: " — the 3D intensity array, per-voxel σ, a validity mask, the HKL axes, and the crystal orientation (UB) matrix travel together from raw input to ΔPDF output." }]]),

    h2("1.2", "Scope of This Manual"),
    p("This manual covers the background physics and mathematics of 3D diffuse scattering analysis, the algorithms and their theoretical basis, a step-by-step workflow, the Python API, every configuration parameter, generic worked examples for single-volume and multi-volume workflows, interactive visualization, known limitations, and a complete reference list."),

    h2("1.3", "What nebula3d Does Not Do"),
    p("The package does not perform raw detector reduction, normalization to an absolute cross-section, or Reverse Monte Carlo (RMC) structural refinement. It assumes the input is a Mantid background-subtracted MDHistoWorkspace volume."),
    ...callout("note", null,
      [[{ text: "Input is the *_cc_sub_bkg.nxs", font: MONO, size: 19 },
        { text: " file produced by Mantid. For the reduction steps that precede it, use Mantid directly.", }]]),
  ];
}

// ───────────────────────────── SECTION 2  Background Theory ─────────────────

function sec2() {
  return [
    h1(2, "Background Theory"),
    lead("What the measurement contains, and why a Fourier transform of the diffuse part maps disorder into real space."),

    h2("2.1", "Neutron Scattering Fundamentals"),
    p("In a neutron scattering experiment a monochromatic beam with wavevector kᵢ is scattered by the sample. The momentum transfer is:"),
    ...math(["Q = k_f − k_i        |Q| = 4π · sin θ / λ"]),
    p("For an elastic single-crystal measurement, the scattered intensity I(Q) is recorded over a 3D grid in reciprocal space indexed by fractional Miller indices (h, k, l). The total intensity separates into two parts:"),
    ...math(["I_total(Q)  =  I_Bragg(Q)  +  I_diffuse(Q)"]),
    p("The Bragg contribution is sharp peaks at integer lattice nodes, encoding the average periodic structure. The diffuse contribution is the broad, continuous signal encoding atomic displacements, occupancy disorder, short-range magnetic correlations, and phonon / magnon scattering."),

    h2("2.2", "The 3D Difference Pair Distribution Function"),
    p("The 3D-ΔPDF is the Fourier transform of the diffuse scattering intensity:"),
    ...math(["Δρ(r) = FT[ I_diffuse(Q) ] = FT[ I_total(Q) − I_Bragg(Q) ]"]),
    p("In real space, Δρ(r) is a 3D map of interatomic-correlation deviations. A positive value at vector r means more interatomic pairs are separated by r than the average structure predicts; a negative value means fewer. The practical relation linking measurement to analysis is:"),
    ...math(["Δρ(r)  ≈  FT[ I_cleaned(Q) ]"]),
    p("where I_cleaned(Q) is the diffuse volume after ring removal, Bragg punching, and backfill."),
    ...callout("note", "Foundational references",
      "Weber & Simonov, Z. Kristallogr. 227, 238–247 (2012); Simonov, Weber & Steurer, J. Appl. Cryst. 47, 2011–2018 (2014). These establish the 3D-ΔPDF formalism and the punch-and-fill data-reduction strategy implemented here."),

    h2("2.3", "Coordinate Systems and Conventions"),
    li([{ text: "Fractional HKL.  ", bold: true }, { text: "Reciprocal-lattice coordinates (h, k, l) in units of a*, b*, c*. Integer values are Bragg positions." }]),
    li([{ text: "Cartesian Q.  ", bold: true }, { text: "Physical momentum transfer in Å⁻¹, related to HKL by the UB matrix. Magnitude |Q| = 2π/d (physics convention)." }]),
    li([{ text: "UB matrix.  ", bold: true }, { text: "Combines orientation U and reciprocal metric B. Mantid stores the crystallographic UB (no 2π); nebula3d multiplies by 2π on read." }]),
    ...math(["Q = 2π · UB_cryst · [h, k, l]ᵀ      ( Å⁻¹ )"]),

    h2("2.4", "Powder Rings: Physical Origin"),
    p("Polycrystalline material in the beam path — cryostat, aluminium heat shields, sample holder, capsule walls — produces powder diffraction rings: bands of elevated intensity at fixed |Q|. For aluminium (FCC, a = 4.046 Å), the strongest is Al(111) at |Q| = 2.69 Å⁻¹."),
    p("Rings are not isotropic. Detector solid-angle coverage, absorption path length, and normalization artifacts modulate ring intensity with azimuthal direction φ. A useful physical model is:"),
    ...math(["I_ring(Q, φ) = T(φ) · Σᵢ Aᵢ · G( |Q| − qᵢ , σᵢ )",
             "I_measured(Q, φ) = I_diffuse(Q) + I_ring(Q, φ)"]),
    p("where G is a Gaussian radial profile for ring i, Aᵢ is the per-ring amplitude, and T(φ) is the shared azimuthal texture."),
  ];
}

// ───────────────────────────── SECTION 3  Architecture ──────────────────────

function sec3() {
  return [
    h1(3, "Package Architecture"),
    lead("A thin, composable layer over NumPy — one data class, modules per pipeline concern, scripts as entry points."),

    h2("3.1", "Module Map"),
    ...dataTable({
      head: ["Module", "Responsibility"],
      widths: [3200, CONTENT_W - 3200], mono: [0],
      rows: [
        ["nebula3d/core.py", "HKLVolume dataclass — the universal data carrier"],
        ["nebula3d/io/", "Mantid NeXus reader, HDF5 load/save, ASCII HKL I/O"],
        ["nebula3d/preprocessing/", "Powder-ring models, radial background, backfill"],
        ["nebula3d/analysis/", "Bragg punch / backfill, 3D-ΔPDF FFT"],
        ["nebula3d/inpainting/", "Symmetry fill, TV (Chambolle–Pock), RBF, biharmonic"],
        ["nebula3d/visualization/", "Slice plots, radial profiles, interactive viewers"],
        ["nebula3d/utils/", "UB matrix, d-spacing, Q↔HKL conversion utilities"],
        ["examples/", "Pipeline scripts and interactive viewer entry points"],
      ],
    }),

    h2("3.2", "The HKLVolume Data Structure"),
    p("Every stage operates on an HKLVolume dataclass defined in src/nebula3d/core.py:"),
    ...code(
`@dataclass
class HKLVolume:
    data   : NDArray[np.float64]   # (nh, nk, nl) intensity
    sigma  : NDArray[np.float64]   # (nh, nk, nl) standard deviation
    mask   : NDArray[np.bool_]     # True = valid (not masked)
    h_axis : NDArray               # 1D fractional-HKL H coordinates
    k_axis : NDArray               # 1D fractional-HKL K coordinates
    l_axis : NDArray               # 1D fractional-HKL L coordinates
    ub_matrix : NDArray[np.float64]  # (3, 3) physics-convention UB (Å⁻¹)
    instrument: str                # provenance label`, "core.py"),
    ...dataTable({
      head: ["Method", "Description"],
      widths: [2900, CONTENT_W - 2900], mono: [0],
      rows: [
        ["hkl_grid()", "Returns (H, K, L) meshgrids of shape (nh, nk, nl)"],
        ["q_magnitude()", "Returns |Q| in Å⁻¹ for every voxel"],
        ["masked_data()", "Returns data with masked voxels set to NaN"],
        ["apply_mask(m)", "Updates the validity mask in place"],
      ],
    }),

    h2("3.3", "File Formats"),
    h3("Input: Mantid NeXus"),
    p("Raw input is a Mantid MDHistoWorkspace saved as NeXus (*_cc_sub_bkg.nxs). The reader (nebula3d/io/mantid_nxs.py) extracts the signal, variance, mask, bin-edge arrays, and the UB matrix; the crystallographic UB is rescaled by 2π on read."),
    h3("Native HDF5"),
    p("All intermediate and output files are HDF5 (.h5). Load with vol = nebula3d.load(path); save with nebula3d.save(vol, path). The ΔPDF output also stores the transform configuration and direct-lattice constants as HDF5 attributes for the interactive viewers."),
  ];
}

// ───────────────────────────── SECTION 4  Algorithms ────────────────────────

function sec4() {
  return [
    h1(4, "Algorithms"),
    lead("The mathematics and design decisions behind each of the four stages."),

    // 4.1
    h2("4.1", "Stage 1 — Powder Ring Removal"),
    h3("4.1.1  Design Principle"),
    p("Ring removal is strictly subtractive. The algorithm estimates only the azimuthally smooth ring intensity and subtracts it. It never masks or replaces a voxel merely for showing radial excess — that excess can be genuine anisotropic diffuse scattering."),
    ...callout("warn", "Subtract, never mask",
      "Masking radial excess would destroy real diffuse signal. Stage 1 only removes the smooth, φ-modulated ring component; everything anisotropic survives to Stage 2."),
    h3("4.1.2  Non-Parametric Patch Method (production)"),
    p("The production algorithm (PatchedRadialRingModel) processes one H-slice (0kl plane) at a time, building a non-parametric radial ring model in azimuthal patches:"),
    li([{ text: "Azimuthal patches.  ", bold: true }, { text: "φ = atan2(k_Q, h_Q) ∈ [0, 2π) is divided into N overlapping Hann-weighted patches." }]),
    li([{ text: "Robust radial profile.  ", bold: true }, { text: "Within each patch, voxels are binned by |Q| and a trimmed-mean profile rejects Bragg peaks and detector gaps." }]),
    li([{ text: "SNIP baseline.  ", bold: true }, { text: "A smooth diffuse baseline is estimated via morphological opening (min then max filters wider than the ring), equivalent to SNIP clipping." }]),
    li([{ text: "Ring component.  ", bold: true }, { text: "ring(|Q|) = max(0, profile − baseline); the positive part prevents over-subtraction." }]),
    li([{ text: "Subtraction.  ", bold: true }, { text: "Each voxel gets a Hann-weighted blend of ring estimates from the two nearest patches, interpolated to its exact |Q|." }]),
    h3("4.1.3  Factored Gaussian / SVD Model (legacy)"),
    p("The older PatchedRingModel fits a rank-1 factored model to the amplitude matrix:"),
    ...math(["A[i, P]  ≈  Aᵢ · T[P]"]),
    p("A rank-1 SVD extracts per-ring amplitudes Aᵢ and patch textures T[P]; a Fourier series gives a smooth periodic T(φ). The rank1_variance diagnostic monitors the approximation — values ≥ 0.90 confirm the shared-texture assumption."),

    // 4.2
    h2("4.2", "Stage 2 — Bragg Peak Punching"),
    p([{ text: "Implemented in nebula3d/analysis/bragg.py by ", }, { text: "BraggRemover", font: MONO, size: 20 },
       { text: ". The recommended production mode is " }, { text: "mode=\"both\"", font: MONO, size: 20 },
       { text: ": a lattice-aware integer-node pass followed by an hkl-agnostic search pass." }]),
    h3("4.2.1  Integer-Node Path"),
    li([{ text: "Enumerate.  ", bold: true }, { text: "All integer (h, k, l) nodes within the volume are listed." }]),
    li([{ text: "Detect.  ", bold: true }, { text: "A local HKL window is inspected; the node is kept only if the local peak exceeds min_intensity and the local median by min_prominence." }]),
    li([{ text: "Recentre.  ", bold: true }, { text: "The punch centre moves to the measured local maximum." }]),
    li([{ text: "Fit shape (optional).  ", bold: true }, { text: "With integer_optimize_shape, the anisotropic covariance of the peak excess yields per-peak ellipsoid radii, clipped by integer_fit_max_radius_hkl." }]),
    li([{ text: "H-slab guard.  ", bold: true }, { text: "integer_h_guard_hkl clips each punch to a slab around the integer-H plane, keeping holes out of diffuse planes at H = ±1/3, ±2/3." }]),
    h3("4.2.2  Punch Ellipsoid"),
    p("Each identified peak is punched as an anisotropic ellipsoid in HKL space:"),
    ...math(["(h−h₀)²/rₕ²  +  (k−k₀)²/rₖ²  +  (l−l₀)²/r_l²   ≤   1"]),
    p("where (h₀, k₀, l₀) is the fitted centre and (rₕ, rₖ, r_l) the HKL semi-axes; a guard margin is added to all radii."),
    h3("4.2.3  Search Path (hkl-agnostic)"),
    p("The search path detects off-integer satellites. At each |Q| shell the background threshold is:"),
    ...math(["bg_threshold = median(I_shell) + n_mad · MAD(I_shell)"]),
    p("Local maxima above this threshold become punch centres. Structured diffuse planes are protected via search_exclude_h_fractions (periodic, preferred) or search_exclude_h_centers (explicit). For example, fractions [0.3333, 0.6667] protect every H = n ± 1/3 and n ± 2/3 plane across the full H range."),

    // 4.3
    h2("4.3", "Stage 3 — Bragg-Hole Backfill"),
    h3("4.3.1  Q-Shell Fill (recommended)"),
    p("For ordinary Bragg holes, the robust diffuse level at the same |Q| is the best estimate. For each |Q| bin:"),
    ...math(["I_fill = median(I_valid) + n_mad · MAD(I_valid)"]),
    h3("4.3.2  TV Inpainting"),
    p("For complex cases, Total-Variation inpainting solves:"),
    ...math(["min_u   ½ ‖ W(u − f) ‖²  +  λ ‖∇u‖₁"]),
    p("where f is the observed data, W the diagonal mask operator, ∇u the 3D forward-difference gradient, and λ the regularisation. The Chambolle–Pock primal-dual algorithm (step sizes τσ = 1/6) preserves piecewise-smooth structures — sharp diffuse sheets and streaks — while suppressing noise."),

    // 4.4
    h2("4.4", "Stage 4 — 3D-ΔPDF Transform"),
    h3("4.4.1  The Correct FFT Recipe"),
    p("The cleaned volume stores Q = 0 at the array centre (index n//2), but NumPy's fftn expects the origin at [0,0,0]. The correct centred transform of real, centrosymmetric I(Q) is:"),
    ...math(["Δρ = fftshift( fftn( ifftshift( I_windowed ) ) ).real"]),
    p("Applied step by step:"),
    li([{ text: "Fill masked voxels", bold: true }, { text: " with 0." }]),
    li([{ text: "Optional Q-crop:", bold: true }, { text: " symmetrically crop to |H| ≤ h_max, |K| ≤ k_max, |L| ≤ l_max." }]),
    li([{ text: "Subtract smooth background:", bold: true }, { text: " I_new = I − GaussianBlur(I, σ) — see 4.4.3." }]),
    li([{ text: "Apodize:", bold: true }, { text: " multiply by a separable Hann / Gaussian window to suppress termination ripples." }]),
    li([{ text: "Remove DC:", bold: true }, { text: " subtract the post-window mean so ΣI = 0 exactly." }]),
    li([{ text: "Zero-pad", bold: true }, { text: " symmetrically to the next fast FFT length (5-smooth), keeping Q = 0 centred." }]),
    li([{ text: "Transform:", bold: true }, { text: " ifftshift → fftn → fftshift." }]),
    li([{ text: "Take the real part", bold: true }, { text: " — valid because I(Q) = I(−Q) for centrosymmetric data." }]),
    h3("4.4.2  Real-Space Axes"),
    ...math(["Δh = (h_max − h_min) / (nₕ − 1)",
             "freq_h = fftshift( fftfreq(nₕ_padded, d=Δh) )",
             "x_a = freq_h · 2π / |UB[:,0]|        ( Å )"]),
    h3("4.4.3  Axis-Cross Artifact"),
    p("A bright cross along the y_K = 0 and z_L = 0 axes can appear in ΔPDF maps. Root cause: a broad separable diffuse envelope whose Fourier transform concentrates energy on the principal axes. Fix: subtract a Gaussian-blurred background before windowing,"),
    ...math(["I_new = I − GaussianBlur(I, σ ≈ 1.5 r.l.u.)"]),
    ...callout("tip", "Validated default",
      [[{ text: "Use " }, { text: "SUBTRACT_BG=\"0,1.5,1.5\"", font: MONO, size: 19 },
        { text: " — per-axis Gaussian blur with σH = 0 (slice-wise, preserves H-layering) and σK = σL = 1.5 r.l.u. Tune the blur widths for the smooth background in your volume." }]]),
    h3("4.4.4  Centring Bug (fixed 2026-06-05)"),
    p("Earlier code computed fftshift(fftn(data)) without ifftshift and used one-sided zero-padding. The missing ifftshift introduces a linear phase ramp e^(−iπk) = (−1)ᵏ, flipping the sign of real-space features by pixel parity. Fixed throughout; regression test: test_delta_pdf_centring_positive_peak."),
  ];
}

// ───────────────────────────── SECTION 5  Installation ──────────────────────

function sec5() {
  return [
    h1(5, "Installation and Setup"),
    h2("5.1", "Requirements"),
    ...dataTable({
      head: ["Package", "Version", "Purpose"],
      widths: [2200, 1900, CONTENT_W - 4100], mono: [0, 1],
      rows: [
        ["Python", "≥ 3.10", "Core language"],
        ["numpy", "≥ 1.24", "Array operations, FFT"],
        ["scipy", "≥ 1.10", "Interpolation, Gaussian filters"],
        ["h5py", "≥ 3.8", "HDF5 file I/O"],
        ["matplotlib", "≥ 3.7", "Visualization"],
      ],
    }),
    h2("5.2", "Install from Source"),
    ...code(`git clone https://github.com/user/nebula3d
cd nebula3d
pip install -e ".[dev]"`, "bash"),
    h2("5.3", "Recommended Runtime Environment"),
    ...code(`export PY=/opt/homebrew/Caskroom/miniforge/base/envs/sci-general/bin/python
export PYTHONPATH=src
export MPLCONFIGDIR=/private/tmp/nebula3d-mpl`, "bash"),
    p("MPLCONFIGDIR keeps the Matplotlib cache outside the repository. If your Python 3.10+ environment already has the dependencies active, replace $PY with python3."),
    h2("5.4", "Verifying the Installation"),
    ...code(`PYTHONPATH=src python -c "import nebula3d; print(nebula3d.__version__)"
PYTHONPATH=src python -m pytest -o addopts='' tests/`, "bash"),
    ...callout("note", null,
      [[{ text: "The ", }, { text: "-o addopts=''", font: MONO, size: 19 },
        { text: " flag is required because the project runs under conda rather than a virtualenv." }]]),
  ];
}

// ───────────────────────────── SECTION 6  Workflow ──────────────────────────

function sec6() {
  return [
    h1(6, "End-to-End Workflow"),
    lead("One command for the whole pipeline, or run each stage on its own — outputs are cached and skipped when present."),

    h2("6.1", "One-Command Pipeline"),
    p("examples/run_pipeline.py orchestrates all four stages, skipping any whose output already exists for fast incremental reprocessing:"),
    ...code(`PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
  python examples/run_pipeline.py`, "bash"),
    ...dataTable({
      head: ["Environment variable", "Effect"],
      widths: [3700, CONTENT_W - 3700], mono: [0],
      rows: [
        ["DATA_FILE=/path/to/file.nxs", "Override the auto-detected input"],
        ["NO_VIEWER=1", "Skip the GUI; stop after writing _delta_pdf.h5"],
        ["FORCE=1", "Recompute every stage even if output exists"],
        ["FORCE_FROM=rings|punch|backfill|pdf", "Recompute from the named stage onward"],
        ["SLICE_AXIS=H|K|L", "Direction for ring-removal iteration (default H)"],
      ],
    }),

    h2("6.2", "Stage-by-Stage Commands"),
    ...code(`PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl RING_PRESET=cc_on \\
  python examples/remove_rings_3d.py`, "Stage 1 · Ring removal"),
    ...code(`PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
  PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \\
  INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \\
  SEARCH_EXCLUDE_H_FRACTIONS=0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \\
  python examples/punch_bragg_3d.py`, "Stage 2 · Bragg punch"),
    ...code(`PYTHONPATH=src METHOD=q_shell \\
  python examples/backfill_bragg_3d.py`, "Stage 3 · Backfill"),
    ...code(`PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
  SUBTRACT_BG="0,1.5,1.5" CROP_H=4 CROP_K=8 CROP_L=15 \\
  APODIZE=gaussian GAUSSIAN_SIGMA=0.4 \\
  python examples/delta_pdf.py`, "Stage 4 · 3D-ΔPDF"),

    h2("6.3", "Multi-Volume Workflow"),
    p("Run the pipeline once per input volume by pointing DATA_FILE at each raw volume:"),
    ...code(`# condition A
NO_VIEWER=1 DATA_FILE="data/raw/condition_a_cc_sub_bkg.nxs" \\
  python examples/run_pipeline.py

# condition B
NO_VIEWER=1 DATA_FILE="data/raw/condition_b_cc_sub_bkg.nxs" \\
  python examples/run_pipeline.py

# condition C
NO_VIEWER=1 DATA_FILE="data/raw/condition_c_cc_sub_bkg.nxs" \\
  python examples/run_pipeline.py`, "bash"),

    h2("6.4", "Python API"),
    ...code(`import nebula3d
from nebula3d.analysis import BraggRemover, backfill_bragg, compute_delta_pdf

vol = nebula3d.load("data/processed/sample_ringremoved.h5")

remover = BraggRemover(
    mode="both", punch_radii=(0.09, 0.12, 0.45),
    min_intensity=0.8, min_prominence=0.8,
    integer_optimize_position=True, integer_optimize_shape=True,
    integer_h_guard_hkl=0.12, integer_local_prominence_n_mad=8.0,
    search_n_mad=4.0,
    search_exclude_h_fractions=(1/3, 2/3),
    search_exclude_h_half_width=0.08,
    incident_beam_ellipsoid_radii_hkl=(0.15, 0.50, 1.00),
)
punched = remover.apply(vol)
filled  = backfill_bragg(punched, method="q_shell")

dpdf = compute_delta_pdf(
    filled, apodization="gaussian", gaussian_sigma=0.4,
    crop_hkl=(4, 8, 15), subtract_smooth_bg=(0, 1.5, 1.5),
)
nebula3d.save(dpdf, "output_delta_pdf.h5")`, "python"),
  ];
}

// ───────────────────────────── SECTION 7  Configuration ─────────────────────

function sec7() {
  const W = [2350, 1000, 1500, CONTENT_W - 4850];
  const cfg = (rows) => dataTable({
    head: ["Parameter", "Type", "Default", "Description"],
    widths: W, mono: [0, 2], rows,
  });
  return [
    h1(7, "Configuration Reference"),
    lead("Every environment variable, its type, default, and effect — grouped by stage."),
    h2("7.1", "Ring Removal"),
    ...cfg([
      ["RING_PRESET", "str", "cc_on", "Preset: cc_on (conservative) or cc_off (aggressive)"],
      ["Q_STEP", "float", "0.02", "Radial bin width for the profile (Å⁻¹)"],
      ["N_FOURIER", "int", "8", "Number of Fourier terms for the T(φ) fit"],
      ["PROFILE_METHOD", "str", "trimmed_mean", "Bin statistic: trimmed_mean or median"],
      ["TEXTURE_Q_SMOOTH", "float", "0.1", "Smoothing σ for texture along |Q| (Å⁻¹)"],
      ["SLICE_AXIS", "str", "H", "Axis along which slices iterate: H, K, or L"],
    ]),
    h2("7.2", "Bragg Punch"),
    ...cfg([
      ["MODE", "str", "both", "Detection: integer, search, or both"],
      ["PUNCH_PRESET", "str", "cc_on", "Preset for radii and thresholds"],
      ["MIN_I", "float", "0.8", "Minimum intensity for Bragg detection"],
      ["MIN_PROM", "float", "0.8", "Minimum local-median prominence"],
      ["INTEGER_FIT_POSITION", "0/1", "1", "Optimize punch centre to the measured max"],
      ["INTEGER_FIT_SHAPE", "0/1", "1", "Fit anisotropic punch radii from data"],
      ["INTEGER_H_GUARD", "float", "0.12", "H-slab half-width for integer-node punches"],
      ["SEARCH_EXCLUDE_H_FRACTIONS", "str", "0.3333,0.6667", "Periodic H exclusions (fractional parts mod 1)"],
      ["SEARCH_EXCLUDE_H_WIDTH", "float", "0.08", "Half-width of search-exclusion slabs"],
      ["INCIDENT_ELLIPSOID_R_HKL", "str", "0.15,0.50,1.00", "Direct-beam punch ellipsoid semi-axes (HKL)"],
    ]),
    h2("7.3", "Backfill"),
    ...cfg([
      ["METHOD", "str", "q_shell", "q_shell, local, tv, symmetry, symmetry+tv"],
      ["Q_SHELL_STEP", "float", "0.05", "Bin width for q-shell fill (Å⁻¹)"],
      ["Q_SHELL_MIN_COUNT", "int", "10", "Min valid voxels/bin before falling back to local"],
      ["TV_LAM", "float", "0.1", "TV regularisation λ (higher = smoother)"],
      ["TV_ITER", "int", "300", "Chambolle–Pock iterations"],
    ]),
    h2("7.4", "3D-ΔPDF Transform"),
    ...cfg([
      ["APODIZE", "str", "gaussian", "Window: hann, gaussian, or none"],
      ["GAUSSIAN_SIGMA", "float", "0.4", "Gaussian window width as fraction of Q_max"],
      ["CROP_H/K/L", "float", "4/8/15", "Symmetric Q-crop limits in HKL"],
      ["SUBTRACT_BG", "str", "0,1.5,1.5", "Per-axis Gaussian-blur σ for background subtraction"],
      ["REAL_SPACE_ANGSTROM", "0/1", "1", "Real-space axes in Å (1) or HKL units (0)"],
      ["OUT_FILE", "str", "_delta_pdf.h5", "Output HDF5 path for the ΔPDF volume"],
    ]),
  ];
}

// ───────────────────────────── SECTION 8  Visualization ─────────────────────

function sec8() {
  return [
    h1(8, "Visualization & Interactive Exploration"),
    lead("Primitive-first plotting: every function takes an HKLVolume, draws into a Matplotlib Axes, and returns it."),
    h2("8.1", "Visualization API"),
    p("All functions live in nebula3d.visualization and work identically in scripts, IPython, and Jupyter:"),
    ...code(`from nebula3d.visualization import (
    extract_slice, plot_slice,
    plot_radial_profile, plot_azimuthal_map,
    plot_overview, SliceData,
)`, "python"),

    h2("8.2", "plot_slice()"),
    p("A 2D intensity slice. The plane argument is read as (horizontal, vertical); the remaining axis is cut at value:"),
    ...dataTable({
      head: ["plane", "x-axis", "y-axis", "Fixed axis (cut by value)"],
      widths: [2500, 1900, 1900, CONTENT_W - 6300], mono: [0],
      align: [undefined, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER],
      rows: [
        ["'kl' / '0kl'", "K", "L", "H"],
        ["'hl' / 'h0l'", "H", "L", "K"],
        ["'hk' / 'hk0'", "H", "K", "L"],
      ],
    }),
    ...code(`# Log-scale view including the Al(111) ring
plot_slice(bkg, "kl", value=0.0, log_scale=True)

# Exact fractional plane with manual colour limits
plot_slice(data, "hk", value=0.3333, interp=True, vmin=0.0, vmax=0.4)`, "python"),

    h2("8.3", "Radial Profile & Azimuthal Map"),
    ...code(`# Radial profile with an Al(111) ring marker
plot_radial_profile(data, mark_q=[2.69])

# Azimuthal texture T(phi) of the Al(111) ring
plot_azimuthal_map(data, q_center=2.69)`, "python"),

    h2("8.4", "Interactive Viewers"),
    h3("Cleanup QA viewer"),
    p("Four panels — raw → ring-removed → Bragg-punched → backfilled — with an H/K/L plane selector and a cut slider. Use it to confirm integer-H Bragg peaks are removed and fractional-H diffuse planes are preserved."),
    ...code(`PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
  PUNCH_PRESET=cc_on MODE=both MIN_I=0.8 MIN_PROM=0.8 \\
  INTEGER_FIT_POSITION=1 INTEGER_FIT_SHAPE=1 INTEGER_H_GUARD=0.12 \\
  SEARCH_EXCLUDE_H_FRACTIONS=0.3333,0.6667 SEARCH_EXCLUDE_H_WIDTH=0.08 \\
  H_VALUE=0.3333 \\
  python examples/explore_slice.py`, "bash"),
    h3("3D-ΔPDF orthoslice viewer (recommended)"),
    p("All three orthogonal real-space planes (x_H–y_K, x_H–z_L, y_K–z_L) with independent cut sliders, a contrast multiplier, and a unit-cell gridline toggle:"),
    ...code(`PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
  PDF_FILE=data/processed/condition_a_delta_pdf.h5 RMAX=50 \\
  python examples/explore_delta_pdf_ortho.py`, "bash"),
    h3("Multi-volume comparison grid"),
    p("A 3×3 grid — rows are three related volumes, columns are the three orthogonal cuts — with shared cut sliders and one global colour scale:"),
    ...code(`PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\
  PDF_FILES=data/processed/condition_a_delta_pdf.h5,data/processed/condition_b_delta_pdf.h5,data/processed/condition_c_delta_pdf.h5 \\
  PDF_LABELS="condition A,condition B,condition C" \\
  python examples/explore_delta_pdf_multi.py`, "bash"),
  ];
}

// ───────────────────────────── SECTION 9  Examples ──────────────────────────

function sec9() {
  return [
    h1(9, "Worked Examples"),
    h2("9.1", "Quick Overview of a New Dataset"),
    ...code(`import nebula3d
from nebula3d.visualization import plot_overview

vol = nebula3d.load("path/to/sample.nxs")
fig = plot_overview(vol, log_scale=True)
fig.savefig("overview.png", dpi=120)`, "python"),
    h2("9.2", "Locating and Profiling a Powder Ring"),
    ...code(`from nebula3d.visualization import plot_radial_profile, plot_azimuthal_map
import matplotlib.pyplot as plt

# Identify ring positions
fig, ax = plt.subplots()
plot_radial_profile(vol, ax=ax, mark_q=[2.69, 4.39, 5.07])

# Check the azimuthal texture T(phi)
plot_azimuthal_map(vol, q_center=2.69, q_width=0.05)`, "python"),
    h2("9.3", "Checking Bragg Punch Quality"),
    ...code(`import nebula3d
from nebula3d.visualization import plot_slice
import matplotlib.pyplot as plt

punched = nebula3d.load("data/processed/..._braggpunched.h5")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Integer-H: Bragg holes should be punched
plot_slice(punched, "kl", value=0.0, ax=axes[0],
           log_scale=True, title="H=0 (integer)")

# Fractional-H: diffuse should be preserved
plot_slice(punched, "kl", value=0.3333, interp=True, ax=axes[1],
           log_scale=True, title="H=1/3 (fractional)")
fig.savefig("bragg_check.png", dpi=110)`, "python"),
    h2("9.4", "Computing and Displaying the 3D-ΔPDF"),
    ...code(`import nebula3d
from nebula3d.analysis import compute_delta_pdf
import matplotlib.pyplot as plt

filled = nebula3d.load("data/processed/..._backfilled.h5")
dpdf = compute_delta_pdf(
    filled, apodization="gaussian", gaussian_sigma=0.4,
    crop_hkl=(4, 8, 15), subtract_smooth_bg=(0, 1.5, 1.5),
)
nebula3d.save(dpdf, "output_delta_pdf.h5")

hk0 = dpdf.slice_hk0()
vmax = abs(hk0.data).max()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(hk0.data.T, origin="lower",
               extent=[hk0.x_axis[0], hk0.x_axis[-1],
                       hk0.y_axis[0], hk0.y_axis[-1]],
               cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xlabel("x_H (A)"); ax.set_ylabel("y_K (A)")
fig.colorbar(im, ax=ax, label="delta-rho (arb.)")
fig.savefig("delta_pdf_hk0.png", dpi=150)`, "python"),
  ];
}

// ───────────────────────────── SECTION 10  Testing ──────────────────────────

function sec10() {
  return [
    h1(10, "Testing"),
    h2("10.1", "Running the Suite"),
    ...code(`# Full suite
PYTHONPATH=src python -m pytest -o addopts='' tests/

# Lint
python -m ruff check src/ tests/

# Type check
python -m mypy src/nebula3d --ignore-missing-imports`, "bash"),
    h2("10.2", "Key Regression Tests"),
    ...dataTable({
      head: ["Test", "What it verifies"],
      widths: [3700, CONTENT_W - 3700], mono: [0],
      rows: [
        ["test_delta_pdf_centring_positive_peak", "FFT centring: a positive cosine input gives a positive real-space peak."],
        ["Ring subtraction (synthetic)", "A known Gaussian ring on a flat background is cleanly subtracted."],
        ["Bragg punch / search modes", "Masks match the expected punch radii and guard conditions."],
        ["Symmetry fill", "Inverse-variance-weighted averaging fills masked voxels from Laue equivalents."],
        ["TV inpainting convergence", "The Chambolle–Pock solver converges within the iteration budget."],
      ],
    }),
  ];
}

// ───────────────────────────── SECTION 11  Troubleshooting ──────────────────

function sec11() {
  const issue = (symptom, cause, fix) => [
    p([{ text: "Symptom.  ", bold: true, color: NAVY }, { text: symptom }], { after: 50 }),
    p([{ text: "Cause.  ", bold: true, color: NAVY }, { text: cause }], { after: 50 }),
    p([{ text: "Fix.  ", bold: true, color: NAVY }, { text: fix }], { after: 150 }),
  ];
  return [
    h1(11, "Known Limitations & Troubleshooting"),
    h2("11.1", "Near-Origin Spike"),
    ...issue(
      "A very strong feature at r < ~3 Å that dominates the colour scale.",
      "Residual high-|Q| Bragg leakage, discontinuities at punch-hole boundaries, and the direct-beam punch.",
      "Set colour scales from the p99 of |Δρ| at r > 3 Å — the default in all viewer scripts."),
    h2("11.2", "Axis Cross in ΔPDF"),
    ...issue(
      "A bright cross along the y_K = 0 and z_L = 0 axes in the real-space map.",
      "A residual separable diffuse envelope (see §4.4.3).",
      "Use SUBTRACT_BG=\"0,1.5,1.5\" (API: subtract_smooth_bg=(0, 1.5, 1.5))."),
    h2("11.3", "Residual Ring After Subtraction"),
    ...issue(
      "A faint ring pattern still visible after ring removal.",
      "Gaussian width σᵢ too narrow, texture mismatch (rank1_variance < 0.90), or wrong preset.",
      "Widen σᵢ via Q_STEP; try RING_PRESET=cc_off for noisier data; inspect rank1_variance."),
    h2("11.4", "Search Mode Punching Diffuse Structure"),
    ...issue(
      "Structured diffuse on fractional-H planes is partially masked.",
      "Search mode does not know the crystal lattice.",
      "Add SEARCH_EXCLUDE_H_FRACTIONS=0.3333,0.6667; the periodic form also catches higher-order satellites."),
    h2("11.5", "Performance Notes"),
    ...dataTable({
      head: ["Operation", "Typical time", "Notes"],
      widths: [3200, 1900, CONTENT_W - 5100],
      rows: [
        ["Ring removal (per-slice)", "~2–5 min", "Sequential per H-slice; embarrassingly parallel"],
        ["Bragg punch", "~1–2 min", "Scales with the number of peaks"],
        ["Backfill (q-shell)", "~30 sec", "Linear in the number of voxels"],
        ["3D-ΔPDF (FFT)", "~10 sec", "Dominated by the FFT of the zero-padded array"],
      ],
    }),
    caption("Benchmarks for a 401×401×301 volume (~48 M voxels) on a typical laptop."),
  ];
}

// ───────────────────────────── SECTION 12  Glossary ─────────────────────────

function sec12() {
  const terms = [
    ["Apodization", "A window applied to I(Q) before the FFT to suppress termination ripples (Hann, Gaussian)."],
    ["Backfill", "Replacement of masked (punched) voxels with estimated background intensities."],
    ["Bragg peaks", "Sharp peaks at integer (and satellite) reciprocal-lattice nodes encoding the average structure."],
    ["Centrosymmetric", "Having inversion symmetry I(Q) = I(−Q); ensures the Fourier transform is real."],
    ["Diffuse scattering", "Broad, continuous intensity encoding disorder, short-range order, and dynamics."],
    ["Fractional HKL", "Reciprocal-lattice coordinates in units of a*, b*, c*; integers are Bragg positions."],
    ["Hann window", "A cosine-squared taper that smoothly suppresses the signal to zero at the Q boundary."],
    ["HKLVolume", "The central data structure: 3D array plus axes, mask, σ, UB matrix, and instrument label."],
    ["Laue class", "Crystal point-group symmetry class (mmm, m3m, 4/mmm…) determining equivalent Q positions."],
    ["MAD", "Median Absolute Deviation; a robust spread measure, median(|x − median(x)|)."],
    ["Powder ring", "An azimuthally smooth band at fixed |Q| from polycrystalline material in the beam."],
    ["Punch", "To mask and remove a Bragg peak by setting a region of voxels invalid."],
    ["Q", "Cartesian momentum transfer (Å⁻¹); Q = UB·[h,k,l]ᵀ, |Q| = 2π/d (physics convention)."],
    ["SNIP", "Sensitive Nonlinear Iterative Peak clipping; a morphological 1D baseline method."],
    ["Total Variation", "The regulariser ‖∇u‖₁, promoting piecewise-smooth solutions; preserves sharp sheets."],
    ["UB matrix", "Orientation (U) × metric (B) matrix; Q = UB·[h, k, l]ᵀ."],
    ["3D-ΔPDF", "Three-dimensional difference pair distribution function; the FT of diffuse scattering."],
  ];
  return [
    h1(12, "Glossary"),
    ...dataTable({
      head: ["Term", "Definition"],
      widths: [2500, CONTENT_W - 2500],
      rows: terms.map(t => [[run(t[0], { bold: true, size: 18, color: NAVY })], t[1]]),
    }),
  ];
}

// ───────────────────────────── SECTION 13  References ───────────────────────

function sec13() {
  const refs = [
    ['T. Weber and A. Simonov, "The three-dimensional pair distribution function analysis of disordered single crystals: basic concepts," Z. Kristallogr. 227, 238–247 (2012). DOI: 10.1524/zkri.2012.1504.'],
    ['A. Simonov, T. Weber, and W. Steurer, "Diffuse scattering from the disordered intermetallic compound TbFe₂(Si,Al)₄," J. Appl. Cryst. 47, 2011–2018 (2014). DOI: 10.1107/S1600576714023668.'],
    ['A. Chambolle and T. Pock, "A First-Order Primal–Dual Algorithm for Convex Problems with Applications to Imaging," J. Math. Imaging Vision 40, 120–145 (2011). DOI: 10.1007/s10851-010-0251-1.'],
    ['M. Bertalmío, G. Sapiro, V. Caselles, and C. Ballester, "Image inpainting," Proc. ACM SIGGRAPH, 417–424 (2000). DOI: 10.1145/344779.344972.'],
    ['M. Bertero and P. Boccacci, Introduction to Inverse Problems in Imaging, IOP Publishing, Bristol (1998). ISBN: 978-0750304351.'],
    ['Mantid Project, "Mantid — Manipulation and Analysis Toolkit for Instrument Data," (2013–present). www.mantidproject.org.'],
    ['C. G. Ryan et al., "SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra," Nucl. Instrum. Methods B 34, 396–402 (1988).'],
  ];
  const W = 760;
  return [
    h1(13, "References"),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [W, CONTENT_W - W],
      rows: refs.map((r, i) => new TableRow({ children: [
        new TableCell({ width: { size: W, type: WidthType.DXA }, borders: { top: NB, bottom: NB, left: NB, right: NB },
          margins: { top: 60, bottom: 60, left: 0, right: 120 },
          children: [new Paragraph({ children: [run(`[${i + 1}]`, { bold: true, color: GOLD, size: 19 })] })] }),
        new TableCell({ width: { size: CONTENT_W - W, type: WidthType.DXA },
          borders: { top: NB, left: NB, right: NB, bottom: { style: BorderStyle.SINGLE, size: 2, color: RULE } },
          margins: { top: 60, bottom: 80, left: 0, right: 0 },
          children: [new Paragraph({ spacing: { line: 268, lineRule: "auto" }, children: [run(r[0], { size: 18 })] })] }),
      ] })),
    }),
  ];
}

// ═══════════════════════════════════════════════════════════════════════════
//  ASSEMBLE
// ═══════════════════════════════════════════════════════════════════════════

const doc = new Document({
  creator: "Tsung-Han Yang",
  title: "nebula3d — User & Reference Manual",
  description: "3D diffuse neutron scattering analysis toolkit manual",
  numbering: {
    config: [{
      reference: "bul",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { run: { color: BLUE }, paragraph: { indent: { left: 580, hanging: 290 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
          style: { run: { color: STEEL }, paragraph: { indent: { left: 1080, hanging: 290 } } } },
      ],
    }],
  },
  styles: {
    default: { document: { run: { font: SANS, size: 21, color: INK }, paragraph: { spacing: { line: 278, lineRule: "auto" } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 38, bold: true, font: SANS, color: NAVY },
        paragraph: { spacing: { before: 240, after: 200 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 10, color: STEEL, space: 6 } },
          outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: SANS, color: BLUE },
        paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, italic: true, font: SANS, color: STEEL },
        paragraph: { spacing: { before: 220, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  sections: [{
    properties: {
      titlePage: true,
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MT, right: MX, bottom: MB, left: MX, header: 680, footer: 540 },
      },
    },
    headers: { default: pageHeader(), first: emptyHeader },
    footers: { default: pageFooter(), first: emptyFooter },
    children: [
      ...cover(),
      ...toc(),
      ...sec1(), ...sec2(), ...sec3(), ...sec4(), ...sec5(), ...sec6(), ...sec7(),
      ...sec8(), ...sec9(), ...sec10(), ...sec11(), ...sec12(), ...sec13(),
    ],
  }],
});

const outPath = path.join(__dirname, "nebula3d_manual.docx");
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log(`Written: ${outPath}  (${(buffer.length / 1024).toFixed(0)} KB)`);
});
