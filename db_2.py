import os
import re
import hashlib
import json
import shutil
import requests
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from bundle_store import DEFAULT_BUNDLE_DB_PATH, save_bundle
from docx_parse_service import DOCXParseConfig, DOCXParseService
from model_utils import default_hf_cache_dir, resolve_hf_model_source
from pdf_parse_service import PDFParseConfig, PDFParseService

load_dotenv()


# =========================================================
# CONFIG
# =========================================================

@dataclass
class IngestConfig:
    db_path: str = "./my_chroma_db"
    bundle_db_path: str = DEFAULT_BUNDLE_DB_PATH
    processed_dir: str = "./data/processed"
    pdf_markdown_dir: str = "./data/pdf_markdown"
    pdf_layout_dir: str = "./data/pdf_layout"
    docx_markdown_dir: str = "./data/docx_markdown"
    source_files_dir: str = "./data/source_files"
    docs_collection_name: str = "docs"
    summary_collection_name: str = "section_summaries"

    embedding_model: str = "ai-forever/sbert_large_nlu_ru"
    embedding_model_path: str | None = os.getenv("EMBEDDING_MODEL_PATH")
    hf_cache_dir: str = os.getenv("HF_HOME", default_hf_cache_dir())
    local_files_only: bool = os.getenv("HF_LOCAL_FILES_ONLY", "1") == "1"
    device: str = "cpu"

    chunk_size: int = 1800
    chunk_overlap: int = 200

    yandex_api_key: str | None = os.getenv("YANDEX_API_KEY")
    yandex_cloud_folder: str | None = os.getenv("YANDEX_CLOUD_FOLDER")


# =========================================================
# LLM CLIENT
# =========================================================

class YandexLLMClient:
    def __init__(self, config: IngestConfig):
        self.config = config
        if not self.config.yandex_api_key:
            raise RuntimeError("Не задан YANDEX_API_KEY в .env")
        if not self.config.yandex_cloud_folder:
            raise RuntimeError("Не задан YANDEX_CLOUD_FOLDER в .env")

        self.model_uri = f"gpt://{self.config.yandex_cloud_folder}/yandexgpt/latest"
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def completion(
        self,
        system_text: str,
        user_text: str,
        temperature: float = 0.1,
        max_tokens: int = 350,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.config.yandex_api_key}",
        }
        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
            "messages": [
                {"role": "system", "text": system_text},
                {"role": "user", "text": user_text},
            ],
        }

        r = requests.post(self.url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Yandex API error {r.status_code}: {r.text}")

        return r.json()["result"]["alternatives"][0]["message"]["text"]


# =========================================================
# HELPERS
# =========================================================

def preprocess_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace(" – ", "\n- ")
    text = text.replace("– ", "\n- ")
    return text


def make_splitter(config: IngestConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", ". ", " "],
    )


def stable_doc_id(document_name: str, version: str, document_folder: str = "") -> str:
    key = f"{document_folder}::{document_name}::{version}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:16]


# =========================================================
# SECTIONS
# =========================================================

def split_into_sections(
    raw_text: str,
    source_file: str,
    document_folder: str,
    document_name: str,
    version: str,
    doc_id: str,
) -> List[Document]:
    lines = raw_text.splitlines()
    sections: List[Document] = []

    current_title = "Документ"
    current_lines: List[str] = []
    section_id = 0

    heading_re = re.compile(r"^(#{1,3})\s+(.+?)\s*$")

    def flush_section():
        nonlocal section_id, current_lines, current_title, sections
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(
                Document(
                    page_content=body,
                    metadata={
                        "source": source_file,
                        "document_folder": document_folder,
                        "document_name": document_name,
                        "version": version,
                        "doc_id": doc_id,
                        "section_id": section_id,
                        "section_title": current_title,
                    },
                )
            )
            section_id += 1

    found_heading = False

    for line in lines:
        m = heading_re.match(line)
        if m:
            found_heading = True
            flush_section()
            current_title = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    flush_section()

    if not found_heading or not sections:
        sections = [
            Document(
                page_content=raw_text.strip(),
                metadata={
                    "source": source_file,
                    "document_folder": document_folder,
                    "document_name": document_name,
                    "version": version,
                    "doc_id": doc_id,
                    "section_id": 0,
                    "section_title": "Документ",
                },
            )
        ]

    return sections


def chunk_sections(sections: List[Document], config: IngestConfig) -> List[Document]:
    splitter = make_splitter(config)
    all_chunks: List[Document] = []
    global_chunk_id = 0

    for sec in sections:
        sec_source = str(sec.metadata["source"])
        sec_document_folder = str(sec.metadata.get("document_folder", "Без папки"))
        sec_document_name = str(sec.metadata["document_name"])
        sec_version = str(sec.metadata["version"])
        sec_doc_id = str(sec.metadata["doc_id"])
        sec_id = int(sec.metadata["section_id"])
        sec_title = str(sec.metadata["section_title"])

        parts = splitter.split_documents([
            Document(
                page_content=sec.page_content,
                metadata={
                    "source": sec_source,
                    "document_folder": sec_document_folder,
                    "document_name": sec_document_name,
                    "version": sec_version,
                    "doc_id": sec_doc_id,
                    "section_id": sec_id,
                    "section_title": sec_title,
                },
            )
        ])

        for d in parts:
            d.metadata["chunk_id"] = global_chunk_id
            all_chunks.append(d)
            global_chunk_id += 1

    return all_chunks


# =========================================================
# SUMMARIES
# =========================================================

def summarize_section(llm_client: YandexLLMClient, section_doc: Document) -> str:
    text = section_doc.page_content.strip()
    title = section_doc.metadata.get("section_title", "Раздел")

    system_text = (
        "Сделай краткий summary раздела документа для поискового индекса.\n"
        "Нужно 3-5 предложений.\n"
        "Сохрани ключевые темы, сущности, условия, сроки, ограничения, номера пунктов, если они важны.\n"
        "Не добавляй новую информацию."
    )
    user_text = f"ЗАГОЛОВОК РАЗДЕЛА: {title}\n\nТЕКСТ РАЗДЕЛА:\n{text[:6000]}"

    return llm_client.completion(
        system_text,
        user_text,
        temperature=0.1,
        max_tokens=350,
    ).strip()


def build_summary_docs(llm_client: YandexLLMClient, sections: List[Document]) -> List[Document]:
    summary_docs: List[Document] = []

    for sec in sections:
        try:
            summary = summarize_section(llm_client, sec)
        except Exception:
            summary = sec.page_content[:800]

        summary_docs.append(
            Document(
                page_content=summary,
                metadata={
                    "source": sec.metadata["source"],
                    "document_folder": sec.metadata.get("document_folder", "Без папки"),
                    "document_name": sec.metadata["document_name"],
                    "version": sec.metadata["version"],
                    "doc_id": sec.metadata["doc_id"],
                    "section_id": sec.metadata["section_id"],
                    "section_title": sec.metadata["section_title"],
                },
            )
        )

    return summary_docs


# =========================================================
# DB HELPERS
# =========================================================

def init_embeddings(config: IngestConfig) -> HuggingFaceEmbeddings:
    model_source = resolve_hf_model_source(
        repo_id=config.embedding_model,
        explicit_path=config.embedding_model_path,
        cache_dir=config.hf_cache_dir,
    )
    return HuggingFaceEmbeddings(
        model_name=model_source,
        cache_folder=config.hf_cache_dir,
        model_kwargs={
            "device": config.device,
            "local_files_only": config.local_files_only,
        },
        encode_kwargs={"normalize_embeddings": True},
    )


def init_chroma(config: IngestConfig, collection_name: str, emb: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        persist_directory=config.db_path,
        collection_name=collection_name,
        embedding_function=emb,
    )


def delete_existing_doc_records(db: Chroma, doc_id: str) -> None:
    try:
        db.delete(where={"doc_id": doc_id})
    except Exception as e:
        print(f"WARNING: не удалось удалить старые записи doc_id={doc_id}: {e}")


def serialize_document(doc: Document) -> Dict[str, Any]:
    return {
        "page_content": doc.page_content,
        "metadata": dict(doc.metadata),
    }


def get_bundle_path(processed_dir: str, doc_id: str) -> str:
    return os.path.join(processed_dir, f"{doc_id}.json")


def store_source_file(source_path: str, source_files_dir: str, doc_id: str) -> str:
    os.makedirs(source_files_dir, exist_ok=True)
    ext = os.path.splitext(source_path)[1].lower() or ".bin"
    stored_path = os.path.join(source_files_dir, f"{doc_id}{ext}")
    shutil.copy2(source_path, stored_path)
    return stored_path


def save_document_bundle(
    bundle_db_path: str,
    processed_dir: str,
    doc_id: str,
    source: str,
    source_type: str,
    original_filename: str,
    stored_source_path: str,
    text_source_path: str,
    layout_source_path: str,
    document_folder: str,
    document_name: str,
    version: str,
    raw_text: str,
    sections: List[Document],
    summary_docs: List[Document],
    chunk_docs: List[Document],
) -> str:
    os.makedirs(processed_dir, exist_ok=True)
    bundle_path = get_bundle_path(processed_dir, doc_id)
    payload = {
        "doc_id": doc_id,
        "source": source,
        "source_type": source_type,
        "original_filename": original_filename,
        "stored_source_path": stored_source_path,
        "text_source_path": text_source_path,
        "layout_source_path": layout_source_path,
        "document_folder": document_folder,
        "document_name": document_name,
        "version": version,
        "raw_text": raw_text,
        "sections": [serialize_document(d) for d in sections],
        "summary_docs": [serialize_document(d) for d in summary_docs],
        "chunk_docs": [serialize_document(d) for d in chunk_docs],
    }
    save_bundle(payload, db_path=bundle_db_path)
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return bundle_path


# =========================================================
# INGEST SERVICE
# =========================================================

class IngestService:
    def __init__(self, config: IngestConfig | None = None):
        self.config = config or IngestConfig()
        self.llm_client = YandexLLMClient(self.config)
        self.pdf_parser = PDFParseService(
            PDFParseConfig(
                output_dir=self.config.pdf_markdown_dir,
                layout_dir=self.config.pdf_layout_dir,
            )
        )
        self.docx_parser = DOCXParseService(
            DOCXParseConfig(output_dir=self.config.docx_markdown_dir)
        )
        self.embeddings = init_embeddings(self.config)
        self.docs_db = init_chroma(self.config, self.config.docs_collection_name, self.embeddings)
        self.summary_db = init_chroma(self.config, self.config.summary_collection_name, self.embeddings)

    def _prepare_source_text(
        self,
        source_path: str,
        document_name: str,
        version: str,
    ) -> tuple[str, str, str]:
        if not os.path.exists(source_path):
            raise FileNotFoundError(source_path)

        source_ext = os.path.splitext(source_path)[1].lower()
        if source_ext == ".pdf":
            parsed_assets = self.pdf_parser.parse_pdf_assets(
                source_path,
                markdown_output_path=os.path.join(
                    self.config.pdf_markdown_dir,
                    f"{document_name}__{version}.md",
                ),
                layout_output_path=os.path.join(
                    self.config.pdf_layout_dir,
                    f"{document_name}__{version}.json",
                ),
            )
            parsed_md_path = parsed_assets["markdown_path"]
            parsed_layout_path = parsed_assets["layout_path"]
            print(f"A) pdf parsed: {parsed_md_path}")
            print(f"A) layout saved: {parsed_layout_path}")
            return source_path, parsed_md_path, parsed_layout_path

        if source_ext == ".docx":
            parsed_md_path = self.docx_parser.parse_docx_to_markdown(
                source_path,
                output_path=os.path.join(
                    self.config.docx_markdown_dir,
                    f"{document_name}__{version}.md",
                ),
            )
            print(f"A) docx parsed: {parsed_md_path}")
            return source_path, parsed_md_path, ""

        return source_path, source_path, ""

    def add_file_to_db(
        self,
        source_path: str,
        document_name: str,
        version: str,
        document_folder: str = "Без папки",
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Dict[str, Any]:
        def progress(value: float, message: str) -> None:
            if progress_callback is not None:
                progress_callback(value, message)

        progress(0.05, "Подготавливаю исходный файл")
        original_source_path, text_source_path, layout_source_path = self._prepare_source_text(
            source_path=source_path,
            document_name=document_name,
            version=version,
        )

        progress(0.18, "Читаю распознанный текст")
        with open(text_source_path, "r", encoding="utf-8") as f:
            raw_text = preprocess_text(f.read())

        document_folder = document_folder.strip() or "Без папки"
        doc_id = stable_doc_id(document_name, version, document_folder)
        source_type = os.path.splitext(original_source_path)[1].lower().lstrip(".") or "unknown"

        progress(0.24, "Сохраняю исходный файл")
        print("B) storing source artifact...")
        stored_source_path = store_source_file(
            source_path=original_source_path,
            source_files_dir=self.config.source_files_dir,
            doc_id=doc_id,
        )
        print(f"B) source stored: {stored_source_path}")

        progress(0.32, "Разбиваю документ на разделы")
        print("C) building sections...")
        sections = split_into_sections(
            raw_text=raw_text,
            source_file=original_source_path,
            document_folder=document_folder,
            document_name=document_name,
            version=version,
            doc_id=doc_id,
        )
        print(f"C) sections built: {len(sections)}")

        progress(0.45, "Строю summary разделов")
        print("D) building summaries...")
        summary_docs = build_summary_docs(self.llm_client, sections)
        print(f"D) summaries built: {len(summary_docs)}")

        progress(0.62, "Нарезаю документ на чанки")
        print("E) building chunks...")
        chunk_docs = chunk_sections(sections, self.config)
        print(f"E) chunks built: {len(chunk_docs)}")

        summary_ids = [
            f"{doc_id}:summary:{int(d.metadata['section_id'])}"
            for d in summary_docs
        ]

        chunk_ids = [
            f"{doc_id}:chunk:{int(d.metadata['chunk_id'])}"
            for d in chunk_docs
        ]

        progress(0.72, "Удаляю старую версию, если она есть")
        print("F) deleting previous version data if exists...")
        delete_existing_doc_records(self.docs_db, doc_id)
        delete_existing_doc_records(self.summary_db, doc_id)
        print("F) old records removed")

        progress(0.80, "Добавляю summary в векторную базу")
        print("G) adding summaries...")
        self.summary_db.add_documents(summary_docs, ids=summary_ids)
        print("G) summaries added")

        progress(0.88, "Добавляю чанки в векторную базу")
        print("H) adding chunks...")
        self.docs_db.add_documents(chunk_docs, ids=chunk_ids)
        print("H) chunks added")

        progress(0.96, "Сохраняю карточку документа")
        print("I) saving document bundle...")
        bundle_path = save_document_bundle(
            bundle_db_path=self.config.bundle_db_path,
            processed_dir=self.config.processed_dir,
            doc_id=doc_id,
            source=original_source_path,
            source_type=source_type,
            original_filename=os.path.basename(original_source_path),
            stored_source_path=stored_source_path,
            text_source_path=text_source_path,
            layout_source_path=layout_source_path,
            document_folder=document_folder,
            document_name=document_name,
            version=version,
            raw_text=raw_text,
            sections=sections,
            summary_docs=summary_docs,
            chunk_docs=chunk_docs,
        )
        print("I) bundle saved")

        result = {
            "source": original_source_path,
            "source_type": source_type,
            "stored_source_path": stored_source_path,
            "text_source": text_source_path,
            "layout_source": layout_source_path,
            "document_folder": document_folder,
            "document_name": document_name,
            "version": version,
            "doc_id": doc_id,
            "bundle_path": bundle_path,
            "sections": len(sections),
            "summaries": len(summary_docs),
            "chunks": len(chunk_docs),
        }

        print(
            "OK: indexed document\n"
            f"  source={result['source']}\n"
            f"  source_type={result['source_type']}\n"
            f"  stored_source_path={result['stored_source_path']}\n"
            f"  text_source={result['text_source']}\n"
            f"  layout_source={result['layout_source']}\n"
            f"  document_folder={result['document_folder']}\n"
            f"  document_name={result['document_name']}\n"
            f"  version={result['version']}\n"
            f"  doc_id={result['doc_id']}\n"
            f"  bundle_path={result['bundle_path']}\n"
            f"  sections={result['sections']}\n"
            f"  summaries={result['summaries']}\n"
            f"  chunks={result['chunks']}"
        )

        progress(1.0, "Готово")
        return result
