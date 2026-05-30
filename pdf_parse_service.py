import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


try:
    import fitz  # type: ignore

    HAS_PYMUPDF = True
except Exception:
    fitz = None
    HAS_PYMUPDF = False

try:
    from pypdf import PdfReader  # type: ignore

    HAS_PYPDF = True
except Exception:
    PdfReader = None
    HAS_PYPDF = False


@dataclass
class PDFParseConfig:
    output_dir: str = "./data/pdf_markdown"
    layout_dir: str = "./data/pdf_layout"
    heading_size_delta: float = 1.5
    min_heading_chars: int = 3
    preserve_page_markers: bool = False
    parser_backend: str = "pdfplumber"


def clean_line(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def starts_with_lowercase(text: str) -> bool:
    match = re.search(r"[A-Za-zА-Яа-яЁё]", text)
    return bool(match and match.group(0).islower())


def looks_like_sentence_continuation(prev_text: str, text: str) -> bool:
    if not prev_text or not text:
        return False

    prev_text = prev_text.rstrip()
    text = text.lstrip()

    if starts_with_lowercase(text):
        return True
    if text.startswith((")", "]", ",", ";", ":", ".", "!", "?", "-", "–", "—")):
        return True
    if prev_text.endswith(("(", "[", "«", "-", "–", "—", "/", "№")):
        return True
    if not prev_text.endswith((".", "!", "?", ":", ";")):
        return True
    return False


def merge_wrapped_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []

    for line in lines:
        line = clean_line(line)
        if not line:
            if merged and merged[-1] != "":
                merged.append("")
            continue

        if not merged:
            merged.append(line)
            continue

        prev = merged[-1]
        if prev == "":
            prev_nonblank = ""
            for item in reversed(merged[:-1]):
                if item != "":
                    prev_nonblank = item
                    break

            if prev_nonblank and looks_like_sentence_continuation(prev_nonblank, line):
                merged.pop()
                merged[-1] = f"{prev_nonblank} {line}"
                continue

            merged.append(line)
            continue

        looks_like_new_block = bool(
            re.match(r"^(\*{1,2}.+\*{1,2}|#{1,6}\s|\d+\.\s|[-–—•]\s|\|)", line)
        )
        prev_ends_hard = prev.endswith((".", ":", ";", "?", "!", "|"))

        if looks_like_new_block or prev_ends_hard:
            merged.append(line)
        else:
            merged[-1] = f"{prev} {line}"

    while merged and merged[-1] == "":
        merged.pop()

    return merged


def lines_to_markdown(lines: List[str]) -> str:
    out: List[str] = []
    prev_blank = True

    for raw_line in lines:
        line = clean_line(raw_line)
        if not line:
            if not prev_blank:
                out.append("")
            prev_blank = True
            continue

        if line.startswith("|") and line.endswith("|"):
            out.append(line)
        elif re.match(r"^\d+\.\s", line):
            out.append(line)
        elif re.match(r"^[-–—•]\s", line):
            out.append(f"- {line[1:].strip()}")
        else:
            out.append(line)

        prev_blank = False

    return "\n".join(out).strip() + "\n"


class PDFParseService:
    def __init__(self, config: Optional[PDFParseConfig] = None):
        self.config = config or PDFParseConfig()

    def parse_pdf_assets(
        self,
        pdf_path: str,
        markdown_output_path: Optional[str] = None,
        layout_output_path: Optional[str] = None,
    ) -> Dict[str, str]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.layout_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        if markdown_output_path is None:
            markdown_output_path = os.path.join(self.config.output_dir, f"{base_name}.md")
        if layout_output_path is None:
            layout_output_path = os.path.join(self.config.layout_dir, f"{base_name}.json")

        markdown_text = self.extract_markdown(pdf_path)
        layout_payload = self.extract_layout(pdf_path)

        with open(markdown_output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        with open(layout_output_path, "w", encoding="utf-8") as f:
            json.dump(layout_payload, f, ensure_ascii=False, indent=2)

        return {
            "markdown_path": markdown_output_path,
            "layout_path": layout_output_path,
        }

    def parse_pdf_to_markdown(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        return self.parse_pdf_assets(
            pdf_path=pdf_path,
            markdown_output_path=output_path,
        )["markdown_path"]

    def extract_markdown(self, pdf_path: str) -> str:
        if HAS_PYMUPDF:
            return self._extract_markdown_with_pymupdf(pdf_path)
        if HAS_PYPDF:
            return self._extract_markdown_with_pypdf(pdf_path)
        raise RuntimeError("Для парсинга PDF нужен установленный PyMuPDF (`fitz`) или pypdf.")

    def extract_layout(self, pdf_path: str) -> Dict[str, Any]:
        if HAS_PYMUPDF:
            return self._extract_layout_with_pymupdf(pdf_path)
        if HAS_PYPDF:
            return self._extract_layout_with_pypdf(pdf_path)
        raise RuntimeError("Для извлечения layout PDF нужен установленный PyMuPDF (`fitz`) или pypdf.")

    def _extract_markdown_with_pymupdf(self, pdf_path: str) -> str:
        assert fitz is not None
        doc = fitz.open(pdf_path)
        try:
            all_lines: List[str] = []
            body_font_size = self._detect_body_font_size(doc)

            for page_index, page in enumerate(doc):
                page_lines = self._extract_page_lines_with_styles(page, body_font_size)
                if self.config.preserve_page_markers and page_index > 0:
                    all_lines.append("")
                    all_lines.append(f"<!-- page {page_index + 1} -->")
                    all_lines.append("")
                all_lines.extend(page_lines)
                all_lines.append("")

            merged = merge_wrapped_lines(all_lines)
            return lines_to_markdown(merged)
        finally:
            doc.close()

    def _detect_body_font_size(self, doc) -> float:
        sizes: List[float] = []

        for page in doc:
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = clean_line(str(span.get("text", "")))
                        size = float(span.get("size", 0.0))
                        if text and size > 0:
                            sizes.append(size)

        if not sizes:
            return 12.0

        rounded = [round(size, 1) for size in sizes]
        return max(set(rounded), key=rounded.count)

    def _extract_page_lines_with_styles(self, page, body_font_size: float) -> List[str]:
        data = page.get_text("dict")
        lines_out: List[str] = []

        for block in data.get("blocks", []):
            block_lines: List[str] = []

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                line_payload = self._build_line_payload(spans, body_font_size)
                text = line_payload["text"]
                if not text:
                    continue

                prev_text = ""
                for prev_line in reversed(block_lines):
                    if prev_line:
                        prev_text = prev_line
                        break

                if prev_text and looks_like_sentence_continuation(prev_text, text):
                    block_lines[-1] = f"{prev_text} {text}"
                    continue

                if line_payload["type"] == "heading":
                    if block_lines and block_lines[-1] != "":
                        block_lines.append("")
                    level = int(line_payload["heading_level"] or 3)
                    block_lines.append(f"{'#' * level} {text}")
                    block_lines.append("")
                else:
                    block_lines.append(text)

            if block_lines:
                lines_out.extend(block_lines)
                if lines_out and lines_out[-1] != "":
                    lines_out.append("")

        return lines_out

    def _build_line_payload(self, spans, body_font_size: float) -> Dict[str, Any]:
        span_payloads: List[Dict[str, Any]] = []
        line_text_parts: List[str] = []
        max_size = 0.0
        is_bold = False
        line_bbox = None

        for span in spans:
            span_text = str(span.get("text", ""))
            clean_span_text = clean_line(span_text)
            if not clean_span_text:
                continue

            font_name = str(span.get("font", ""))
            font_size = float(span.get("size", 0.0))
            bold_flag = "bold" in font_name.lower() or int(span.get("flags", 0)) & 16
            span_bbox = [float(v) for v in span.get("bbox", [])]

            span_payloads.append(
                {
                    "text": clean_span_text,
                    "bbox": span_bbox,
                    "font": font_name,
                    "size": font_size,
                    "bold": bool(bold_flag),
                }
            )
            line_text_parts.append(clean_span_text)
            max_size = max(max_size, font_size)
            is_bold = is_bold or bool(bold_flag)
            line_bbox = span_bbox if line_bbox is None else line_bbox

        line_text = clean_line(" ".join(line_text_parts))
        line_type = "paragraph"
        heading_level = None

        if line_text:
            if self._looks_like_heading(line_text, max_size, body_font_size, is_bold):
                line_type = "heading"
                heading_level = self._heading_level(max_size, body_font_size)
            elif re.match(r"^\d+\.\s", line_text):
                line_type = "numbered_item"
            elif re.match(r"^[-–—•]\s", line_text):
                line_type = "bullet_item"

        return {
            "text": line_text,
            "bbox": line_bbox or [],
            "type": line_type,
            "heading_level": heading_level,
            "font_size": max_size if max_size > 0 else None,
            "bold": is_bold,
            "spans": span_payloads,
        }

    def _looks_like_heading(
        self,
        text: str,
        max_size: float,
        body_font_size: float,
        is_bold: bool,
    ) -> bool:
        if len(text) < self.config.min_heading_chars:
            return False
        if re.match(r"^\d+\.\s", text):
            return False
        if re.match(r"^\d+\s+[А-ЯЁA-Z]", text):
            return False
        if starts_with_lowercase(text):
            return False
        if text.endswith(";"):
            return False
        if text.endswith((",", ":", "(", "-", "–", "—")):
            return False
        if max_size >= body_font_size + self.config.heading_size_delta:
            return True
        if is_bold and len(text) <= 140:
            return True
        return False

    def _heading_level(self, max_size: float, body_font_size: float) -> int:
        if max_size >= body_font_size + 6:
            return 1
        if max_size >= body_font_size + 3:
            return 2
        return 3

    def _extract_markdown_with_pypdf(self, pdf_path: str) -> str:
        assert PdfReader is not None
        reader = PdfReader(pdf_path)
        all_lines: List[str] = []

        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            page_lines = text.splitlines()
            if self.config.preserve_page_markers and page_index > 0:
                all_lines.append("")
                all_lines.append(f"<!-- page {page_index + 1} -->")
                all_lines.append("")
            all_lines.extend(page_lines)
            all_lines.append("")

        merged = merge_wrapped_lines(all_lines)
        return lines_to_markdown(merged)

    def _extract_layout_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        assert fitz is not None
        doc = fitz.open(pdf_path)
        try:
            body_font_size = self._detect_body_font_size(doc)
            pages: List[Dict[str, Any]] = []

            for page_index, page in enumerate(doc):
                page_dict = page.get_text("dict")
                page_payload: Dict[str, Any] = {
                    "page_no": page_index + 1,
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "blocks": [],
                }

                for block_index, block in enumerate(page_dict.get("blocks", [])):
                    block_lines: List[Dict[str, Any]] = []
                    block_text_parts: List[str] = []
                    block_bbox = [float(v) for v in block.get("bbox", [])]

                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue

                        line_payload = self._build_line_payload(spans, body_font_size)
                        if not line_payload["text"]:
                            continue

                        block_lines.append(line_payload)
                        block_text_parts.append(str(line_payload["text"]))

                    block_text = "\n".join(block_text_parts).strip()
                    if not block_text:
                        continue

                    first_type = str(block_lines[0]["type"]) if block_lines else "paragraph"
                    page_payload["blocks"].append(
                        {
                            "block_no": block_index,
                            "type": first_type,
                            "bbox": block_bbox,
                            "text": block_text,
                            "lines": block_lines,
                        }
                    )

                pages.append(page_payload)

            return {
                "source_pdf": pdf_path,
                "pages": pages,
            }
        finally:
            doc.close()

    def _extract_layout_with_pypdf(self, pdf_path: str) -> Dict[str, Any]:
        assert PdfReader is not None
        reader = PdfReader(pdf_path)
        pages: List[Dict[str, Any]] = []

        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            lines = [clean_line(line) for line in text.splitlines()]
            lines = [line for line in lines if line]
            pages.append(
                {
                    "page_no": page_index + 1,
                    "width": None,
                    "height": None,
                    "blocks": [
                        {
                            "block_no": 0,
                            "type": "paragraph",
                            "bbox": [],
                            "text": "\n".join(lines),
                            "lines": [
                                {
                                    "text": line,
                                    "bbox": [],
                                    "type": "paragraph",
                                    "heading_level": None,
                                    "font_size": None,
                                    "bold": False,
                                    "spans": [],
                                }
                                for line in lines
                            ],
                        }
                    ],
                }
            )

        return {
            "source_pdf": pdf_path,
            "pages": pages,
        }
