import html
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from bundle_store import DEFAULT_BUNDLE_DB_PATH, load_bundle


@dataclass
class CompareConfig:
    bundle_db_path: str = DEFAULT_BUNDLE_DB_PATH
    compare_dir: str = "./data/compare"


@dataclass
class SectionBlock:
    section_id: int
    section_title: str
    text: str
    display_text: str


@dataclass
class NumberedPoint:
    point_no: str
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", text)
    text = re.sub(r"\(\s*", "(", text)
    text = re.sub(r"\s*\)", ")", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_display_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", text)
    text = re.sub(r"\(\s*", "(", text)
    text = re.sub(r"\s*\)", ")", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_with_separators(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)


def preserve_paragraphs_html(text: str) -> str:
    paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not paragraphs:
        return ""

    return "".join(
        f"<div class='paragraph'>{html.escape(paragraph)}</div>"
        for paragraph in paragraphs
    )


def render_diff_html(left_text: str, right_text: str) -> Tuple[str, str, bool]:
    left_tokens = tokenize_with_separators(left_text)
    right_tokens = tokenize_with_separators(right_text)
    matcher = SequenceMatcher(a=left_tokens, b=right_tokens, autojunk=False)

    left_parts: List[str] = []
    right_parts: List[str] = []
    changed = False

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        left_chunk = "".join(left_tokens[a0:a1])
        right_chunk = "".join(right_tokens[b0:b1])

        if opcode == "equal":
            left_parts.append(html.escape(left_chunk))
            right_parts.append(html.escape(right_chunk))
            continue

        changed = True
        if opcode in ("replace", "delete"):
            left_parts.append(
                f"<span class='diff-removed'>{html.escape(left_chunk)}</span>"
            )
        if opcode in ("replace", "insert"):
            right_parts.append(
                f"<span class='diff-added'>{html.escape(right_chunk)}</span>"
            )

    return "".join(left_parts), "".join(right_parts), changed


def split_numbered_points(text: str) -> List[NumberedPoint]:
    pattern = re.compile(r"(?m)^\s*(\d+)\.\s")
    matches = list(pattern.finditer(text))

    if len(matches) < 2:
        return []

    points: List[NumberedPoint] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        points.append(NumberedPoint(point_no=match.group(1), text=chunk))

    return points


def render_block_diff_html(
    left_text: str,
    right_text: str,
    label: str | None = None,
) -> Tuple[str, str, bool]:
    left_html, right_html, changed = render_diff_html(left_text, right_text)

    if label:
        label_html = f"<div class='point-label'>{html.escape(label)}</div>"
        left_html = label_html + f"<div class='point-body'>{left_html}</div>"
        right_html = label_html + f"<div class='point-body'>{right_html}</div>"

    return left_html, right_html, changed


def render_structured_section_diff(
    left_text: str,
    right_text: str,
    left_display_text: str,
    right_display_text: str,
) -> Tuple[str, str, bool]:
    left_points = split_numbered_points(left_text)
    right_points = split_numbered_points(right_text)

    if not left_points and not right_points:
        left_html, right_html, changed = render_diff_html(left_text, right_text)
        if not changed:
            return (
                preserve_paragraphs_html(left_display_text),
                preserve_paragraphs_html(right_display_text),
                False,
            )
        return left_html, right_html, changed

    left_display_points = split_numbered_points(left_display_text)
    right_display_points = split_numbered_points(right_display_text)

    left_by_no = {point.point_no: point for point in left_points}
    right_by_no = {point.point_no: point for point in right_points}
    left_display_by_no = {point.point_no: point for point in left_display_points}
    right_display_by_no = {point.point_no: point for point in right_display_points}
    ordered_numbers: List[str] = []

    for point in left_points:
        if point.point_no not in ordered_numbers:
            ordered_numbers.append(point.point_no)
    for point in right_points:
        if point.point_no not in ordered_numbers:
            ordered_numbers.append(point.point_no)

    left_parts: List[str] = []
    right_parts: List[str] = []
    changed = False

    for number in ordered_numbers:
        left_point = left_by_no.get(number)
        right_point = right_by_no.get(number)
        left_display_point = left_display_by_no.get(number)
        right_display_point = right_display_by_no.get(number)

        left_chunk = left_point.text if left_point else ""
        right_chunk = right_point.text if right_point else ""
        left_display_chunk = left_display_point.text if left_display_point else left_chunk
        right_display_chunk = right_display_point.text if right_display_point else right_chunk

        left_html, right_html, block_changed = render_block_diff_html(
            left_chunk,
            right_chunk,
            label=f"Пункт {number}",
        )

        if not block_changed:
            label_html = f"<div class='point-label'>{html.escape(f'Пункт {number}')}</div>"
            left_html = (
                label_html
                + f"<div class='point-body'>{preserve_paragraphs_html(left_display_chunk)}</div>"
            )
            right_html = (
                label_html
                + f"<div class='point-body'>{preserve_paragraphs_html(right_display_chunk)}</div>"
            )

        left_parts.append(f"<div class='point'>{left_html}</div>")
        right_parts.append(f"<div class='point'>{right_html}</div>")
        changed = changed or block_changed

    return "".join(left_parts), "".join(right_parts), changed


def bundle_to_sections(bundle: Dict[str, object]) -> List[SectionBlock]:
    sections_raw = bundle.get("sections", [])
    sections: List[SectionBlock] = []

    if not isinstance(sections_raw, list):
        return sections

    for item in sections_raw:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        page_content = str(item.get("page_content", ""))
        sections.append(
            SectionBlock(
                section_id=int(metadata.get("section_id", 0)),
                section_title=str(metadata.get("section_title", "Документ")).strip(),
                text=normalize_text(page_content),
                display_text=normalize_display_text(page_content),
            )
        )

    sections.sort(key=lambda x: x.section_id)
    return sections


def pair_sections(
    left_sections: List[SectionBlock],
    right_sections: List[SectionBlock],
) -> List[Tuple[SectionBlock | None, SectionBlock | None]]:
    right_by_title: Dict[str, List[SectionBlock]] = {}
    for section in right_sections:
        right_by_title.setdefault(section.section_title, []).append(section)

    pairs: List[Tuple[SectionBlock | None, SectionBlock | None]] = []
    used_right_ids = set()

    for left in left_sections:
        matched = None
        candidates = right_by_title.get(left.section_title, [])
        for candidate in candidates:
            if candidate.section_id not in used_right_ids:
                matched = candidate
                used_right_ids.add(candidate.section_id)
                break
        pairs.append((left, matched))

    for right in right_sections:
        if right.section_id not in used_right_ids:
            pairs.append((None, right))

    return pairs


class CompareService:
    def __init__(self, config: CompareConfig | None = None):
        self.config = config or CompareConfig()

    def load_document_bundle(self, doc_id: str) -> Dict[str, object]:
        bundle = load_bundle(doc_id, db_path=self.config.bundle_db_path)
        if bundle is None:
            raise RuntimeError(f"Не удалось загрузить bundle для doc_id={doc_id}")
        return bundle

    def compare_documents(
        self,
        left_bundle: Dict[str, object],
        right_bundle: Dict[str, object],
    ) -> Dict[str, object]:
        left_sections = bundle_to_sections(left_bundle)
        right_sections = bundle_to_sections(right_bundle)
        section_pairs = pair_sections(left_sections, right_sections)

        comparisons: List[Dict[str, object]] = []
        changed_sections = 0

        for index, (left, right) in enumerate(section_pairs, start=1):
            left_text = left.text if left else ""
            right_text = right.text if right else ""
            left_display_text = left.display_text if left else ""
            right_display_text = right.display_text if right else ""

            left_html, right_html, changed = render_structured_section_diff(
                left_text,
                right_text,
                left_display_text,
                right_display_text,
            )

            title = (
                left.section_title
                if left and left.section_title
                else right.section_title if right else f"Section {index}"
            )

            if left and not right:
                change_type = "removed"
            elif right and not left:
                change_type = "added"
            elif changed:
                change_type = "changed"
            else:
                change_type = "same"

            comparisons.append(
                {
                    "position": index,
                    "section_title": title,
                    "left_section_id": left.section_id if left else None,
                    "right_section_id": right.section_id if right else None,
                    "changed": changed,
                    "change_type": change_type,
                    "left_text": left_display_text,
                    "right_text": right_display_text,
                    "left_html": left_html,
                    "right_html": right_html,
                }
            )

            if changed:
                changed_sections += 1

        return {
            "left_doc": {
                "doc_id": left_bundle.get("doc_id", ""),
                "document_name": left_bundle.get("document_name", ""),
                "version": left_bundle.get("version", ""),
                "source": left_bundle.get("source", ""),
            },
            "right_doc": {
                "doc_id": right_bundle.get("doc_id", ""),
                "document_name": right_bundle.get("document_name", ""),
                "version": right_bundle.get("version", ""),
                "source": right_bundle.get("source", ""),
            },
            "summary": {
                "left_sections": len(left_sections),
                "right_sections": len(right_sections),
                "compared_sections": len(comparisons),
                "changed_sections": changed_sections,
            },
            "sections": comparisons,
        }

    def ensure_compare_dir(self) -> None:
        os.makedirs(self.config.compare_dir, exist_ok=True)

    def save_compare_json(self, compare_result: Dict[str, object]) -> str:
        self.ensure_compare_dir()
        left_doc = compare_result["left_doc"]
        right_doc = compare_result["right_doc"]
        filename = f"{left_doc['doc_id']}__{right_doc['doc_id']}.json"
        path = os.path.join(self.config.compare_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(compare_result, f, ensure_ascii=False, indent=2)
        return path

    def save_compare_html(self, compare_result: Dict[str, object]) -> str:
        self.ensure_compare_dir()
        left_doc = compare_result["left_doc"]
        right_doc = compare_result["right_doc"]
        filename = f"{left_doc['doc_id']}__{right_doc['doc_id']}.html"
        path = os.path.join(self.config.compare_dir, filename)

        sections_html = []
        for section in compare_result["sections"]:
            badge = "changed" if section["changed"] else "same"
            sections_html.append(
                f"""
                <section class="compare-section">
                  <div class="section-head">
                    <h2>{html.escape(str(section["section_title"]))}</h2>
                    <span class="badge {badge}">{badge}</span>
                  </div>
                  <div class="compare-grid">
                    <div class="pane">
                      <div class="pane-title">Левый документ</div>
                      <div class="pane-body">{section["left_html"]}</div>
                    </div>
                    <div class="pane">
                      <div class="pane-title">Правый документ</div>
                      <div class="pane-body">{section["right_html"]}</div>
                    </div>
                  </div>
                </section>
                """
            )

        html_text = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Document Compare</title>
  <style>
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: #f4f1ea;
      color: #1f1a17;
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
      box-sizing: border-box;
    }}
    .hero {{
      background: linear-gradient(135deg, #fdf8ef, #e8dfcf);
      border: 1px solid #d5c7ad;
      padding: 20px 24px;
      margin-bottom: 24px;
      box-sizing: border-box;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      font-size: 14px;
      overflow-wrap: anywhere;
    }}
    .compare-section {{
      margin-bottom: 22px;
      border: 1px solid #d8ccb7;
      background: #fffdf8;
      box-sizing: border-box;
      overflow: hidden;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid #e6ddcf;
      gap: 12px;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 18px;
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    .badge {{
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.08em;
      flex: 0 0 auto;
    }}
    .badge.changed {{
      color: #a33a24;
    }}
    .badge.same {{
      color: #3d6e48;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 0;
      align-items: start;
    }}
    .pane {{
      min-height: 180px;
      min-width: 0;
      overflow: hidden;
      box-sizing: border-box;
    }}
    .pane + .pane {{
      border-left: 1px solid #e6ddcf;
    }}
    .pane-title {{
      padding: 10px 14px;
      font-size: 13px;
      background: #f7f1e5;
      border-bottom: 1px solid #e6ddcf;
    }}
    .pane-body {{
      padding: 16px;
      white-space: normal;
      line-height: 1.55;
      overflow-wrap: anywhere;
      word-break: break-word;
      box-sizing: border-box;
    }}
    .point {{
      margin-bottom: 18px;
    }}
    .point:last-child {{
      margin-bottom: 0;
    }}
    .point-label {{
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #6d5a43;
      margin-bottom: 8px;
    }}
    .point-body {{
      white-space: normal;
    }}
    .paragraph {{
      margin-bottom: 10px;
      white-space: pre-wrap;
    }}
    .paragraph:last-child {{
      margin-bottom: 0;
    }}
    .diff-added {{
      text-decoration: underline;
      text-decoration-color: #18794e;
      text-decoration-thickness: 3px;
      background: rgba(24, 121, 78, 0.1);
    }}
    .diff-removed {{
      text-decoration: underline;
      text-decoration-color: #b42318;
      text-decoration-thickness: 3px;
      background: rgba(180, 35, 24, 0.1);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Сравнение документов</h1>
      <div class="meta">
        <div>
          <strong>Левый:</strong><br>
          {html.escape(str(left_doc["document_name"]))} | version={html.escape(str(left_doc["version"]))}<br>
          {html.escape(str(left_doc["source"]))}
        </div>
        <div>
          <strong>Правый:</strong><br>
          {html.escape(str(right_doc["document_name"]))} | version={html.escape(str(right_doc["version"]))}<br>
          {html.escape(str(right_doc["source"]))}
        </div>
      </div>
    </div>
    {''.join(sections_html)}
  </div>
</body>
</html>
"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_text)
        return path
