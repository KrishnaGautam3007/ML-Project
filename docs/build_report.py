"""
Build the project report (Word + PDF) in the department's report format.

  - Starts from docs/assets/report_template.docx and keeps its cover page.
  - Replaces the template's instructions with the real report:
      Abstract, Table of Contents, Chapters 1-4, References
    (no UML diagrams, kept short).
  - Uses Microsoft Word (if installed) to fill in the Table of Contents page
    numbers and to export a PDF copy.

Run from the project root:
    python docs/build_report.py

Needs: python-docx, pypdfium2, Pillow (figures are cut from the explainer PDF).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
TEMPLATE = ASSETS / "report_template.docx"
EXPLAINER_PDF = HERE / "GNN_Vulnerability_Scanner_Explained.pdf"
OUT_DOCX = HERE / "GNN_Vulnerability_Scanner_Report.docx"
OUT_PDF = HERE / "GNN_Vulnerability_Scanner_Report.pdf"

COURSE = "UML501: Machine Learning"
PROJECT_TITLE = "GNN-Based Vulnerability Scanner for C/C++ on GitHub"
GUIDE_NAME = "Mrs. Reeta"
GUIDE_TITLE = None                # e.g. "Assistant Professor"; None removes the line
STUDENTS = [                      # (roll number, name) as shown on the cover
    ("1024030733", "Krishna Gautam"),
    ("1024031149", "Prateek Yadav"),
]
FONT = "Times New Roman"
MONO = "Consolas"
CONTENT_WIDTH_MM = 160          # A4 (210) minus 30 mm left and 20 mm right margin
BULLET_NUM_ID = 90


# =============================================================================
# Figures: cut from the explainer PDF (rendered at 3x = ~216 dpi)
# =============================================================================

# figure -> list of (page index, (left, top, right, bottom) in pixels at scale 3);
# several regions are stacked top-to-bottom into one image
FIGURE_CROPS = {
    "report_fig_architecture.png": [(3, (120, 478, 1662, 1228))],
    "report_fig_graph.png": [(5, (800, 488, 1648, 1284))],
    "report_fig_data.png": [(4, (120, 488, 1662, 874)), (4, (120, 1496, 1662, 1654))],
    "report_fig_features.png": [(6, (120, 556, 1662, 856))],
    "report_fig_model.png": [(7, (225, 494, 795, 1302))],
    "report_fig_lines.png": [(9, (120, 1348, 1662, 1776))],
}


def make_figures():
    import pypdfium2 as pdfium
    from PIL import Image

    import build_pdf
    import report_diagrams

    pdf = pdfium.PdfDocument(str(EXPLAINER_PDF))
    pages = {}
    for name, regions in FIGURE_CROPS.items():
        parts = []
        for page_idx, box in regions:
            if page_idx not in pages:
                pages[page_idx] = pdf[page_idx].render(scale=3).to_pil()
            parts.append(pages[page_idx].crop(box))
        gap = 30
        width = max(p.width for p in parts)
        height = sum(p.height for p in parts) + gap * (len(parts) - 1)
        out = Image.new("RGB", (width, height), "white")
        y = 0
        for p in parts:
            out.paste(p, (0, y))
            y += p.height + gap
        out.save(ASSETS / name)

    # the two flowcharts are drawn from SVG (see report_diagrams.py)
    report_diagrams.make_all(ASSETS, build_pdf.find_browser())


# =============================================================================
# Low-level helpers
# =============================================================================

def set_run_font(rpr_owner, name: str):
    """Force a font on a style/run (removes theme fonts, which would win)."""
    rpr = rpr_owner.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)


def add_field(paragraph, instruction: str, placeholder: str = ""):
    """Insert a Word field (e.g. TOC, PAGE) into a paragraph."""
    def fld_char(kind):
        r = paragraph.add_run()
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
        r._r.append(el)

    fld_char("begin")
    r = paragraph.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    r._r.append(instr)
    fld_char("separate")
    if placeholder:
        paragraph.add_run(placeholder)
    fld_char("end")


def insert_in_order(parent, child, successors):
    """Word requires a fixed child order: put `child` before the first successor found."""
    for tag in successors:
        found = parent.find(qn(tag))
        if found is not None:
            found.addprevious(child)
            return
    parent.append(child)


# elements that must come AFTER <w:shd> inside a paragraph / table-cell property block
PPR_AFTER_SHD = ("w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap",
                 "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN",
                 "w:bidi", "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
                 "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
                 "w:textDirection", "w:textAlignment", "w:textboxTightWrap", "w:outlineLvl",
                 "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange")
TCPR_AFTER_SHD = ("w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign",
                  "w:hideMark", "w:headers", "w:cellIns", "w:cellDel", "w:cellMerge",
                  "w:tcPrChange")
TBLPR_AFTER_BORDERS = ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
                       "w:tblCaption", "w:tblDescription", "w:tblPrChange")
SECTPR_AFTER_PGNUM = ("w:cols", "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg",
                      "w:textDirection", "w:bidi", "w:rtlGutter", "w:docGrid",
                      "w:printerSettings", "w:sectPrChange")


def shade(element_pr, fill_hex: str, successors=PPR_AFTER_SHD):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    insert_in_order(element_pr, shd, successors)


def set_table_borders(table, color="808080", size=4):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    insert_in_order(tbl_pr, borders, TBLPR_AFTER_BORDERS)


def ensure_bullet_numbering(document):
    """Add a simple bullet list definition (the template has none)."""
    numbering = document.part.numbering_part.element
    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(BULLET_NUM_ID))
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    for tag, val in (("w:start", "1"), ("w:numFmt", "bullet"), ("w:lvlText", "•"), ("w:lvlJc", "left")):
        el = OxmlElement(tag)
        el.set(qn("w:val"), val)
        lvl.append(el)
    ppr = OxmlElement("w:pPr")
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "567")
    ind.set(qn("w:hanging"), "284")
    ppr.append(ind)
    lvl.append(ppr)
    rpr = OxmlElement("w:rPr")
    rfonts = OxmlElement("w:rFonts")
    rfonts.set(qn("w:ascii"), "Arial")
    rfonts.set(qn("w:hAnsi"), "Arial")
    rpr.append(rfonts)
    lvl.append(rpr)
    abstract.append(lvl)

    # schema order: every abstractNum must come before the first num
    first_num = numbering.find(qn("w:num"))
    if first_num is not None:
        first_num.addprevious(abstract)
    else:
        numbering.append(abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(BULLET_NUM_ID))
    ref = OxmlElement("w:abstractNumId")
    ref.set(qn("w:val"), str(BULLET_NUM_ID))
    num.append(ref)
    numbering.append(num)


# =============================================================================
# The report writer
# =============================================================================

class Report:
    def __init__(self, document):
        self.d = document

    # ---- inline markup: **bold** and `code` ----
    def _runs(self, p, text, size=None):
        parts = []
        for chunk in text.split("**"):
            parts.append(chunk)
        bold = False
        for i, chunk in enumerate(parts):
            bold = i % 2 == 1
            code_parts = chunk.split("`")
            for j, piece in enumerate(code_parts):
                if not piece:
                    continue
                run = p.add_run(piece)
                run.bold = bold
                if j % 2 == 1:
                    set_run_font(run._r, MONO)
                    run.font.size = Pt(10.5)
                elif size:
                    run.font.size = Pt(size)

    def title(self, text, page_break=True):
        p = self.d.add_paragraph(style="Report Title")
        p.paragraph_format.page_break_before = page_break
        p.add_run(text)
        return p

    def h1(self, text):
        return self.d.add_paragraph(text, style="Heading 1")

    def h2(self, text):
        return self.d.add_paragraph(text, style="Heading 2")

    def para(self, text, align=None):
        p = self.d.add_paragraph(style="Report Body")
        self._runs(p, text)
        if align is not None:
            p.alignment = align
        return p

    def bullet(self, text):
        p = self.d.add_paragraph(style="Report Bullet")
        num_pr = OxmlElement("w:numPr")
        ilvl = OxmlElement("w:ilvl")
        ilvl.set(qn("w:val"), "0")
        num_id = OxmlElement("w:numId")
        num_id.set(qn("w:val"), str(BULLET_NUM_ID))
        num_pr.append(ilvl)
        num_pr.append(num_id)
        p._p.get_or_add_pPr().append(num_pr)
        self._runs(p, text)
        return p

    def bullets(self, items):
        for item in items:
            self.bullet(item)

    def code(self, lines):
        # keep the sentence that introduces the code on the same page as the code
        prev = self.d.paragraphs[-1] if self.d.paragraphs else None
        if prev is not None and prev.style.name == "Report Body":
            prev.paragraph_format.keep_with_next = True
        for i, line in enumerate(lines):
            p = self.d.add_paragraph(style="Report Code")
            p.add_run(line if line else " ")
            if i == len(lines) - 1:
                p.paragraph_format.space_after = Pt(8)
        return p

    def caption(self, text):
        return self.d.add_paragraph(text, style="Report Caption")

    def figure(self, path, width_mm, caption):
        p = self.d.add_paragraph(style="Report Body")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(str(path), width=Mm(width_mm))
        self.caption(caption)

    def table(self, rows, widths_mm, caption=None, font_size=10.5):
        """rows[0] is the header. Widths must add up to the content width."""
        if caption:
            cap = self.caption(caption)
            cap.paragraph_format.keep_with_next = True
            cap.paragraph_format.space_after = Pt(3)
        t = self.d.add_table(rows=len(rows), cols=len(rows[0]))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        set_table_borders(t)
        for c, w in enumerate(widths_mm):          # grid widths (used by some viewers)
            t.columns[c].width = Mm(w)
        for r, row in enumerate(rows):
            # a row never splits across pages
            tr_pr = t.rows[r]._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:cantSplit"))
            for c, value in enumerate(row):
                cell = t.cell(r, c)
                cell.width = Mm(widths_mm[c])       # cell widths (used by Word)
                p = cell.paragraphs[0]
                p.style = self.d.styles["Report Table"]
                # keep the whole table on one page (short tables only)
                p.paragraph_format.keep_with_next = r < len(rows) - 1
                self._runs(p, str(value), size=font_size)
                if r == 0:
                    for run in p.runs:
                        run.bold = True
                    shade(cell._tc.get_or_add_tcPr(), "D9E2F3", TCPR_AFTER_SHD)
        # spacing after the table
        self.d.add_paragraph(style="Report Body").paragraph_format.space_after = Pt(2)
        return t


# =============================================================================
# Template preparation: keep the cover, drop the instructions
# =============================================================================

def prepare_template(document):
    body = document.element.body
    children = list(body.iterchildren())

    # the cover ends at the paragraph that holds the first section's properties
    cover_end = None
    for i, el in enumerate(children):
        if el.tag == qn("w:p") and el.find(qn("w:pPr")) is not None \
                and el.find(qn("w:pPr")).find(qn("w:sectPr")) is not None:
            cover_end = i
            break
    if cover_end is None:
        raise SystemExit("Could not find the end of the cover page in the template.")

    # delete the template's instruction pages (keep the final body sectPr)
    for el in children[cover_end + 1:]:
        if el.tag != qn("w:sectPr"):
            body.remove(el)

    # fill in the project name on the cover. The template uses an exact line
    # height there (for a one-line name); our name wraps to two lines, so give
    # it normal single spacing and remove two of the blank lines under the logo
    # to keep the cover on one page.
    paragraphs = document.paragraphs
    for p in paragraphs:
        if p.text.strip() == "Name of the Project":
            runs = p.runs
            runs[0].text = PROJECT_TITLE
            for r in runs[1:]:
                r.text = ""
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            break
    def set_text(p, text):
        """Replace a paragraph's text but keep the formatting of its first run."""
        runs = p.runs
        runs[0].text = text
        for r in runs[1:]:
            r.text = ""

    # guide name and students (the template has 3 student lines; drop unused ones)
    student_lines = []
    for p in paragraphs:
        t = p.text.strip()
        if t.startswith("UCS553"):                     # the template's course line
            set_text(p, COURSE)
        elif t == "Mini Project Report":                # (your edit: no "Mini")
            set_text(p, "Project Report")
        elif t.startswith("Dr.") and "ABC" in t and "XYZ" in t:
            set_text(p, GUIDE_NAME)
        elif t == "Assistant Professor":
            if GUIDE_TITLE:
                set_text(p, GUIDE_TITLE)
            else:
                p._p.getparent().remove(p._p)
        elif t.startswith("Roll Number") and "Student" in t:
            student_lines.append(p)
    for p, (roll, name) in zip(student_lines, STUDENTS):
        set_text(p, f"{roll}    {name}")
    for p in student_lines[len(STUDENTS):]:
        p._p.getparent().remove(p._p)

    for i, p in enumerate(paragraphs):
        if p._p.xpath(".//*[local-name()='drawing']"):          # the logo
            blanks = [q for q in paragraphs[i + 1:i + 5] if not q.text.strip()]
            for q in blanks[1:3]:
                q._p.getparent().remove(q._p)
            break

    # the report pages: A4, new page, report margins, page numbers in the footer
    sec = document.sections[-1]
    sec.start_type = WD_SECTION.NEW_PAGE
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin, sec.right_margin = Mm(30), Mm(20)
    sec.top_margin, sec.bottom_margin = Mm(25), Mm(22)
    sec.header_distance, sec.footer_distance = Mm(12), Mm(10)

    sec.header.is_linked_to_previous = False
    for p in sec.header.paragraphs:
        p.text = ""
    sec.footer.is_linked_to_previous = False
    fp = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
    fp.text = ""
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(fp, "PAGE", "1")
    for r in fp.runs:
        set_run_font(r._r, FONT)
        r.font.size = Pt(10)
    # number the report pages from 1 (the cover is unnumbered)
    old = sec._sectPr.find(qn("w:pgNumType"))
    if old is not None:
        sec._sectPr.remove(old)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:start"), "1")
    insert_in_order(sec._sectPr, pg, SECTPR_AFTER_PGNUM)


def define_styles(document):
    styles = document.styles

    def para_style(name, base="Normal"):
        if name in [s.name for s in styles]:
            return styles[name]
        s = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = styles[base]
        return s

    black = RGBColor(0, 0, 0)

    h1 = styles["Heading 1"]
    set_run_font(h1.element, FONT)
    h1.font.size, h1.font.bold, h1.font.color.rgb = Pt(16), True, black
    h1.paragraph_format.page_break_before = True
    h1.paragraph_format.space_before, h1.paragraph_format.space_after = Pt(0), Pt(14)
    h1.paragraph_format.keep_with_next = True
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h1.paragraph_format.left_indent = Mm(0)          # the template style is indented
    h1.paragraph_format.first_line_indent = Mm(0)

    h2 = styles["Heading 2"]
    set_run_font(h2.element, FONT)
    h2.font.size, h2.font.bold, h2.font.color.rgb = Pt(13), True, black
    h2.paragraph_format.space_before, h2.paragraph_format.space_after = Pt(12), Pt(5)
    h2.paragraph_format.keep_with_next = True
    h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h2.paragraph_format.left_indent = Mm(0)
    h2.paragraph_format.first_line_indent = Mm(0)

    title = para_style("Report Title")
    set_run_font(title.element, FONT)
    title.font.size, title.font.bold = Pt(16), True
    title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before, title.paragraph_format.space_after = Pt(0), Pt(16)

    bodys = para_style("Report Body")
    set_run_font(bodys.element, FONT)
    bodys.font.size = Pt(12)
    pf = bodys.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.2
    pf.space_before, pf.space_after = Pt(0), Pt(6)
    pf.first_line_indent = Mm(0)

    bul = para_style("Report Bullet", base="Report Body")
    bul.paragraph_format.space_after = Pt(3)
    bul.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    code = para_style("Report Code")
    # shading first: python-docx then slots spacing/indent in after it, in schema order
    shade(code.element.get_or_add_pPr(), "F2F2F2")
    set_run_font(code.element, MONO)
    code.font.size = Pt(9.5)
    cpf = code.paragraph_format
    cpf.space_before, cpf.space_after = Pt(0), Pt(0)
    cpf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    cpf.left_indent = Mm(4)
    cpf.keep_with_next = True

    cap = para_style("Report Caption")
    set_run_font(cap.element, FONT)
    cap.font.size, cap.font.italic = Pt(10.5), True
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before, cap.paragraph_format.space_after = Pt(0), Pt(10)

    tbl = para_style("Report Table")
    set_run_font(tbl.element, FONT)
    tbl.font.size = Pt(10.5)
    tbl.paragraph_format.space_before, tbl.paragraph_format.space_after = Pt(2), Pt(2)
    tbl.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    tbl.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT


# =============================================================================
# Content
# =============================================================================

def write_content(R: Report):
    """The report text lives in report_content.py; its numbers come from results/."""
    import report_content
    report_content.write_content(R, add_field)


# =============================================================================
# Word: update the table of contents and export a PDF
# =============================================================================

POWERSHELL_WORD = r"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open('{docx}')
    foreach ($toc in $doc.TablesOfContents) {
        $toc.Update()
        $toc.Range.Font.Name = 'Times New Roman'
        $toc.Range.Font.Size = 12
    }
    $doc.Save()
    $doc.ExportAsFixedFormat('{pdf}', 17)
    $doc.Close()
} finally {
    $word.Quit()
}
"""


def finish_with_word():
    script = POWERSHELL_WORD.replace("{docx}", str(OUT_DOCX)).replace("{pdf}", str(OUT_PDF))
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       check=True, capture_output=True, text=True, timeout=180)
        print(f"Word updated the table of contents and exported: {OUT_PDF}")
    except Exception as exc:  # Word not installed, etc.
        print("Could not run Microsoft Word to update the table of contents "
              f"({exc}). Open the .docx and press F9 on the contents to update it.")


def main():
    make_figures()
    document = docx.Document(str(TEMPLATE))
    prepare_template(document)
    define_styles(document)
    ensure_bullet_numbering(document)
    write_content(Report(document))
    document.save(str(OUT_DOCX))
    print(f"Report written: {OUT_DOCX}")
    finish_with_word()


if __name__ == "__main__":
    main()
