"""Document loaders + text cleaning.

Supports Markdown (.md), plain text (.txt), Word (.docx) and PDF (.pdf).
Each loader also parses the lightweight metadata header used in the sample corpus.
"""

from __future__ import annotations

import os
import re

from .schema import Document, DocMetadata, stable_id

_META_KEYS = {"source", "author", "date", "access"}

# zero-width, bidi-control, BOM and non-breaking-space characters, built from
# codepoints so this source file stays pure-ASCII and easy to edit.
_INVISIBLE = "".join(chr(c) for c in (0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF))
_INVISIBLE_RE = re.compile("[" + re.escape(_INVISIBLE) + "]")
_NBSP_RE = re.compile(chr(0x00A0))

# normalise common unicode punctuation to ASCII (keeps terminals/UI mojibake-free)
_PUNCT_MAP = {
    0x2014: "-", 0x2013: "-", 0x2012: "-", 0x2212: "-",       # dashes
    0x2018: "'", 0x2019: "'", 0x201A: "'",                     # single quotes
    0x201C: '"', 0x201D: '"', 0x201E: '"',                     # double quotes
    0x2026: "...", 0x00A0: " ", 0x2022: "-",                   # ellipsis, nbsp, bullet
}


def _parse_header_metadata(text: str, filename: str, doc_type: str) -> tuple[DocMetadata, str]:
    """Extract 'Key: value' metadata lines from the top of a document.

    The sample corpus stores source/author/date/access near the top (as markdown
    bold or plain lines). We scan the first ~15 lines and pull those out so they
    become real vector-store metadata rather than noise inside the body.
    """
    found: dict[str, str] = {}
    title = None
    lines = text.splitlines()
    for raw in lines[:15]:
        line = raw.strip().lstrip("#").strip()
        line = line.replace("**", "").lstrip("-* ").strip()  # strip md bold/list markers
        if title is None and line and ":" not in line:
            title = line
        m = re.match(r"^([A-Za-z/ ]+)\s*:\s*(.+)$", line)
        if m:
            key = m.group(1).strip().lower().split("/")[0]
            if key in _META_KEYS:
                found[key] = m.group(2).strip()
    meta = DocMetadata(
        source=filename,  # always key on the real filename
        title=title or filename,
        author=found.get("author", "Unknown"),
        date=found.get("date", "unknown"),
        access=found.get("access", "public").lower(),
        doc_type=doc_type,
    )
    return meta, text


_META_LINE_RE = re.compile(r"^\s*\*{0,2}(source|author|date|access)\*{0,2}\s*:.*$",
                           re.IGNORECASE | re.MULTILINE)


def _heading_to_sentence(m: re.Match) -> str:
    """Turn '## Data Encryption' into 'Data Encryption.' so it splits cleanly."""
    h = m.group(1).strip()
    return h if h.endswith((".", "!", "?", ":")) else h + "."


def _strip_markup(text: str) -> str:
    """Remove markdown heading/emphasis markers so chunks hold clean prose."""
    text = re.sub(r"^#{1,6}\s*(.+?)\s*$", _heading_to_sentence, text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "")              # bold
    text = re.sub(r"(?<!\w)[*_](?=\w)", "", text)                # stray emphasis
    return text


def clean_text(text: str, strip_metadata: bool = True) -> str:
    """Normalise whitespace/encoding and strip boilerplate + markdown markup."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE_RE.sub("", text)
    text = text.translate(_PUNCT_MAP)
    if strip_metadata:
        text = _META_LINE_RE.sub("", text)      # drop Source/Author/Date/Access lines
    text = _strip_markup(text)
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse 3+ blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)      # collapse runs of spaces/tabs
    return text.strip()


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_docx(path: str) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


_LOADERS = {
    ".md": (_read_txt, "md"),
    ".markdown": (_read_txt, "md"),
    ".txt": (_read_txt, "txt"),
    ".docx": (_read_docx, "docx"),
    ".pdf": (_read_pdf, "pdf"),
}


def load_document(path: str) -> Document:
    """Load a single file into a cleaned Document with parsed metadata."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    reader, doc_type = _LOADERS[ext]
    raw = reader(path)
    filename = os.path.basename(path)
    # parse metadata from the raw header first, then strip those lines from the body
    meta, _ = _parse_header_metadata(raw, filename, doc_type)
    body = clean_text(raw, strip_metadata=True)
    doc_id = stable_id(filename)
    return Document(doc_id=doc_id, text=body, metadata=meta)


def load_corpus(corpus_dir: str) -> list[Document]:
    """Load every supported file in a directory, sorted for deterministic order."""
    docs: list[Document] = []
    for name in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in _LOADERS:
            docs.append(load_document(path))
    return docs
