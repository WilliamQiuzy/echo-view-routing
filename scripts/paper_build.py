#!/usr/bin/env python
"""Build the numbered manuscript from docs/paper/manuscript.md (source with [R:key] citation keys).

Citations are numbered in order of first appearance over the whole document (PLOS ONE citation-sequence style); a
bracket with several keys becomes "[1, 3]" or a range "[4-6]". The "## References" section is replaced by the numbered
list from docs/paper/references.json. Unknown keys stop the build.

Default (internal review): writes build/manuscript.md and build/manuscript.docx with figures embedded (no captions
generated from image alt text) and the submission-form statements kept at the end.
--submission: drops the embedded figures, moves the statements to build/submission_form.md, and refuses to write while
any [TEAM...] placeholder or unresolved citation key remains (unknown information is never filled in automatically).

The Word file is post-processed: table rows do not break across pages and header rows repeat, body text is double
spaced, lines are numbered continuously and pages are numbered in the footer.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "docs" / "paper"
CITE = re.compile(r"\[(R:[^\]]+)\]")
STATEMENTS = "## Statements for the submission form (not part of the manuscript file)"
FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
          '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
          '<w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')


def group(nums: list[int]) -> str:
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return "[" + ", ".join(out) + "]"


def number_citations(text: str, refs: dict) -> tuple[str, list[str]]:
    order: list[str] = []
    for m in CITE.finditer(text):
        for key in (k.strip()[2:] for k in m.group(1).split(",")):
            if key not in refs:
                raise SystemExit(f"no reference entry for key {key!r}")
            if key not in order:
                order.append(key)
    num = {k: i + 1 for i, k in enumerate(order)}
    return CITE.sub(lambda m: group([num[k.strip()[2:]] for k in m.group(1).split(",")]), text), order


def postprocess_docx(path: Path) -> None:
    """Rows kept on one page, repeated header rows, double spacing, continuous line numbers, page numbers."""
    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf8")
                xml = re.sub(r"<w:tr>(?!<w:trPr>)", "<w:tr><w:trPr><w:cantSplit/></w:trPr>", xml)
                xml = re.sub(r"<w:tr><w:trPr>(?!<w:cantSplit/>)", "<w:tr><w:trPr><w:cantSplit/>", xml)
                xml = re.sub(r"(<w:tbl>.*?<w:tr><w:trPr><w:cantSplit/>)", r"\1<w:tblHeader/>", xml, flags=re.S)
                sect = '<w:footerReference w:type="default" r:id="rIdPaperFooter"/>'
                xml = re.sub(r"<w:sectPr( [^>]*)?>", lambda m: m.group(0) + sect, xml, count=1)
                xml = xml.replace("</w:sectPr>", '<w:lnNumType w:countBy="1" w:restart="continuous"/></w:sectPr>', 1) \
                    if "<w:lnNumType" not in xml else xml
                if 'xmlns:r="' not in xml[:2000]:
                    xml = xml.replace("<w:document ", '<w:document xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" ', 1)
                data = xml.encode("utf8")
            elif item.filename == "word/styles.xml":
                xml = data.decode("utf8")
                xml = re.sub(r'(<w:spacing [^>]*?)w:line="\d+"', r'\1', xml)
                xml = re.sub(r'(<w:spacing [^>]*?)w:lineRule="\w+"', r'\1', xml)
                xml = re.sub(r"<w:spacing ", '<w:spacing w:line="480" w:lineRule="auto" ', xml)
                if "<w:pPrDefault>" in xml and "<w:spacing" not in xml.split("<w:pPrDefault>")[1].split("</w:pPrDefault>")[0]:
                    xml = xml.replace("<w:pPrDefault><w:pPr>", '<w:pPrDefault><w:pPr><w:spacing w:line="480" w:lineRule="auto"/>', 1)
                data = xml.encode("utf8")
            elif item.filename == "word/_rels/document.xml.rels":
                xml = data.decode("utf8")
                xml = xml.replace("</Relationships>", '<Relationship Id="rIdPaperFooter" Type="http://schemas.openxmlformats.org/'
                                  'officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>')
                data = xml.encode("utf8")
            elif item.filename == "[Content_Types].xml":
                xml = data.decode("utf8")
                xml = xml.replace("</Types>", '<Override PartName="/word/footer1.xml" ContentType="application/'
                                  'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
                data = xml.encode("utf8")
            zout.writestr(item, data)
        zout.writestr("word/footer1.xml", FOOTER)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the numbered manuscript")
    ap.add_argument("--submission", action="store_true", help="strict build for submission (see module docstring)")
    args = ap.parse_args()
    src = (PAPER / "manuscript.md").read_text(); refs = json.loads((PAPER / "references.json").read_text())
    head, _, tail = src.partition("## References\n")
    rest = tail[tail.index("\n---"):] if "\n---" in tail else ""
    text, order = number_citations(head + "\x00" + rest, refs)
    head, rest = text.split("\x00")
    ref_list = "\n".join(f"{i + 1}. {refs[k]}" for i, k in enumerate(order))
    out = head + "## References\n\n" + ref_list + "\n" + rest
    out = re.sub(r"\*Internal review draft.*?\*\n", "*Internal review draft, numbered build. Edit `docs/paper/manuscript.md`, not this file.*\n",
                 out, count=1, flags=re.S)
    build = PAPER / "build"; build.mkdir(exist_ok=True)
    name = "manuscript"
    if args.submission:
        out, _, statements = out.partition(STATEMENTS)
        out = re.sub(r"!\[[^\]]*\]\([^)]*\)\n\n", "", out).rstrip().rstrip("-").rstrip() + "\n"
        out = re.sub(r"\*Internal review draft, numbered build.*?\*\n\n", "", out)
        leftovers = sorted(set(re.findall(r"\[TEAM[^\]]*\]|\[R:[^\]]*\]", out + statements)))
        if leftovers:
            raise SystemExit("submission build refused; unresolved items:\n- " + "\n- ".join(x[:100] for x in leftovers))
        (build / "submission_form.md").write_text("# Statements for the submission form\n" + statements)
        name = "manuscript_submission"
    (build / f"{name}.md").write_text(out)
    unused = sorted(set(refs) - set(order))
    print(f"{len(order)} references cited; unused entries: {unused}")
    if shutil.which("pandoc"):
        docx = build / f"{name}.docx"
        subprocess.run(["pandoc", str(build / f"{name}.md"), "-f", "markdown-implicit_figures", "-o", str(docx),
                        f"--resource-path={PAPER}"], check=True)
        postprocess_docx(docx)
        print(f"wrote {docx}")


if __name__ == "__main__":
    main()
