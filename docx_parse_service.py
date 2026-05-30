import os
from dataclasses import dataclass
from typing import Iterator, Optional, Union

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


@dataclass
class DOCXParseConfig:
    output_dir: str = "./data/docx_markdown"


def clean_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split()).strip()


def iter_block_items(parent: DocumentObject) -> Iterator[Union[Paragraph, Table]]:
    parent_element = parent.element.body
    for child in parent_element.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def style_to_heading_level(style_name: str) -> Optional[int]:
    normalized = style_name.strip().lower()
    if normalized in {"title"}:
        return 1
    if normalized in {"subtitle"}:
        return 2
    if normalized.startswith("heading "):
        suffix = normalized.split("heading ", 1)[1].strip()
        if suffix.isdigit():
            return max(1, min(int(suffix), 6))
    return None


def table_to_markdown(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [clean_text(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]

    header = normalized_rows[0]
    separator = ["---"] * max_cols
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in normalized_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class DOCXParseService:
    def __init__(self, config: Optional[DOCXParseConfig] = None):
        self.config = config or DOCXParseConfig()

    def extract_markdown(self, docx_path: str) -> str:
        if not os.path.exists(docx_path):
            raise FileNotFoundError(docx_path)

        document = Document(docx_path)
        output = []

        for block in iter_block_items(document):
            if isinstance(block, Paragraph):
                text = clean_text(block.text)
                if not text:
                    continue

                style_name = block.style.name if block.style is not None else ""
                heading_level = style_to_heading_level(style_name)
                if heading_level is not None:
                    output.append(f"{'#' * heading_level} {text}")
                    output.append("")
                    continue

                output.append(text)
                output.append("")
                continue

            if isinstance(block, Table):
                table_md = table_to_markdown(block)
                if table_md:
                    output.append(table_md)
                    output.append("")

        markdown_text = "\n".join(output).strip()
        return markdown_text + "\n" if markdown_text else ""

    def parse_docx_to_markdown(self, docx_path: str, output_path: Optional[str] = None) -> str:
        os.makedirs(self.config.output_dir, exist_ok=True)
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(docx_path))[0]
            output_path = os.path.join(self.config.output_dir, f"{base_name}.md")

        markdown_text = self.extract_markdown(docx_path)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(markdown_text)
        return output_path


def main() -> None:
    docx_path = input("Путь к DOCX: ").strip().strip('"')
    service = DOCXParseService()
    result = service.parse_docx_to_markdown(docx_path)
    print(f"markdown_path: {result}")


if __name__ == "__main__":
    main()
