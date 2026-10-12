"""Render the public Markdown documents (problem statement and rules) to TeX and PDF.

Requires pandoc (or pypandoc with its bundled pandoc), XeLaTeX, and the named
system fonts. This script reads no examiner data or configuration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def render(source: Path, *, cjk_font: str = "AR PL SungtiL GB") -> Path:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        try:
            import pypandoc
            pandoc = pypandoc.get_pandoc_path()
        except (ImportError, OSError) as exc:
            raise RuntimeError("Install pandoc or pypandoc_binary to render the statement") from exc
    xelatex = shutil.which("xelatex")
    if not xelatex:
        raise RuntimeError("XeLaTeX is required")
    if not re.fullmatch(r"[\w .-]+", cjk_font):
        raise ValueError("Use a plain system font family name")
    source = source.resolve()
    markdown = source.read_text(encoding="utf-8")
    title, body = markdown.split("\n", 1)
    if not title.startswith("# ") or not re.search(r"^## [0-9]+\. ", body, flags=re.M):
        raise ValueError("Expected a title line and numbered sections")
    latex_body = subprocess.run([pandoc, "--from", "markdown+tex_math_single_backslash", "--to", "latex",
                                "--shift-heading-level-by=-1"], input=body, text=True,
                               check=True, capture_output=True).stdout
    latex_body = latex_body.replace(r"{\def\LTcaptype", r"{\small\def\LTcaptype")
    latex_title = subprocess.run([pandoc, "--from", "markdown", "--to", "latex"],
                                input=title[2:], text=True, check=True, capture_output=True).stdout.strip()
    preamble = r'''% Generated from SOURCE_NAME by scripts/render_statement.py.
% Public statement only; edit the Markdown source, then render again.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{fontspec,xeCJK}
\setmainfont{Liberation Serif}
\setsansfont{DejaVu Sans}
\setmonofont[Scale=0.83]{DejaVu Sans Mono}
\setCJKmainfont[AutoFakeBold=2]{CJK_FONT}
\setCJKsansfont[AutoFakeBold=2]{CJK_FONT}
\setCJKmonofont{CJK_FONT}
\usepackage{amsmath,amssymb}
\usepackage{longtable,booktabs,array,calc}
\usepackage{graphicx,xcolor}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage[font=small,labelsep=quad]{caption}
\renewcommand{\figurename}{图}
\newcounter{none} % Pandoc tables set \LTcaptype{none}; caption needs that counter.
\makeatletter
\@ifundefined{KV@Gin@alt}{\define@key{Gin}{alt}{}}{}
\newsavebox\pandoc@box
\newcommand*\pandocbounded[1]{\sbox\pandoc@box{#1}%
  \Gscale@div\@tempa{\textheight}{\dimexpr\ht\pandoc@box+\dp\pandoc@box\relax}%
  \Gscale@div\@tempb{\linewidth}{\wd\pandoc@box}%
  \ifdim\@tempb\p@<\@tempa\p@\let\@tempa\@tempb\fi
  \ifdim\@tempa\p@<\p@\scalebox{\@tempa}{\usebox\pandoc@box}\else\usebox{\pandoc@box}\fi}
\makeatother
\usepackage[unicode,hidelinks]{hyperref}
\usepackage{titlesec,enumitem}
\setcounter{secnumdepth}{0}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{11pt}{5pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.14}
\setlength{\LTpre}{4pt}
\setlength{\LTpost}{4pt}
\linespread{1.08}
\clubpenalty=10000
\widowpenalty=10000
\setlist{leftmargin=1.6em,itemsep=4pt,topsep=3pt}
\providecommand{\tightlist}{\setlength{\itemsep}{3pt}\setlength{\parskip}{0pt}}
\hypersetup{pdftitle={TITLE_TEXT},pdfauthor={SOP-UQ},pdfsubject={SOP-UQ public document}}
\begin{document}
\begin{center}\fontsize{15}{20}\selectfont\bfseries TITLE_TEXT\end{center}
\vspace{2pt}
'''.replace("CJK_FONT", cjk_font).replace("TITLE_TEXT", latex_title).replace(
        "SOURCE_NAME", source.name
    )
    tex = source.with_suffix(".tex")
    tex.write_text(preamble + latex_body + "\n\\end{document}\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="sop-statement-") as tmp:
        for _ in range(2):
            result = subprocess.run([xelatex, "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                                     "-output-directory", tmp, str(tex)], capture_output=True, text=True,
                                    cwd=source.parent)  # Figure paths are relative to the Markdown file.
            if result.returncode:
                raise RuntimeError("XeLaTeX failed:\n" + result.stdout[-6000:])
        log = (Path(tmp) / (source.stem + ".log")).read_text(errors="replace")
        if "Missing character:" in log:
            raise RuntimeError("The selected fonts cannot render all statement characters")
        overfull = re.findall(r"Overfull \\[hv]box[^\n]*", log)
        if overfull:
            raise RuntimeError("Fix overflowing text before publishing: " + "; ".join(overfull))
        pdf = source.with_suffix(".pdf")
        shutil.copyfile(Path(tmp) / pdf.name, pdf)
    return pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    docs = Path(__file__).resolve().parents[1] / "docs"
    parser.add_argument(
        "--source",
        type=Path,
        nargs="*",
        default=[docs / "problem_statement.md", docs / "rules.md"],
    )
    parser.add_argument("--cjk-font", default="AR PL SungtiL GB")
    args = parser.parse_args()
    for source in args.source:
        print(render(source, cjk_font=args.cjk_font))


if __name__ == "__main__":
    main()
