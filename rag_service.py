import os
import re
import json
import hashlib
import logging
import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from bundle_store import DEFAULT_BUNDLE_DB_PATH, load_bundle
from compare_service import CompareService, CompareConfig
from model_utils import default_hf_cache_dir, resolve_hf_model_source

load_dotenv()

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("rag_service")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(LOG_DIR, "rag_service.log"), encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except Exception:
    HAS_CROSS_ENCODER = False


# =========================================================
# CONFIG
# =========================================================

@dataclass
class RAGConfig:
    db_path: str = "./my_chroma_db"
    bundle_db_path: str = DEFAULT_BUNDLE_DB_PATH
    processed_dir: str = "./data/processed"
    collection_name: str = "docs"
    summary_collection_name: str = "section_summaries"

    embedding_model: str = "ai-forever/sbert_large_nlu_ru"
    embedding_model_path: Optional[str] = os.getenv("EMBEDDING_MODEL_PATH")
    hf_cache_dir: str = os.getenv("HF_HOME", default_hf_cache_dir())
    local_files_only: bool = os.getenv("HF_LOCAL_FILES_ONLY", "1") == "1"
    device: str = "cpu"

    chunk_size: int = 1800
    chunk_overlap: int = 200

    chroma_k: int = 10
    bm25_k: int = 8
    summary_chroma_k: int = 4
    summary_bm25_k: int = 4

    top_docs_after_fusion: int = 10
    rerank_top_n: int = 6
    max_context_chars: int = 16000
    neighbor_window: int = 1

    max_query_variants: int = 4
    max_subquestions: int = 3
    max_sections_to_boost: int = 3

    enable_query_correction: bool = True
    enable_query_expansion: bool = True
    enable_decomposition: bool = True
    enable_section_summaries: bool = True
    enable_rerank: bool = True

    summary_cache_file: str = "./section_summary_cache.json"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_model_path: Optional[str] = os.getenv("RERANKER_MODEL_PATH")

    debug: bool = False

    yandex_cloud_folder: Optional[str] = os.getenv("YANDEX_CLOUD_FOLDER")
    yandex_api_key: Optional[str] = os.getenv("YANDEX_API_KEY")


# =========================================================
# HELPERS
# =========================================================

def preprocess_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace(" – ", "\n- ")
    text = text.replace("– ", "\n- ")
    return text


def normalize_query(q: str) -> str:
    q = q.replace("\r\n", "\n").strip()
    q = re.sub(r"[ \t]+", " ", q)
    return q.strip()


def uniq_keep_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for x in items:
        x = x.strip()
        if not x:
            continue
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def make_splitter(config: RAGConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", ". ", " "],
    )


def file_fingerprint(path: str) -> str:
    stat = os.stat(path)
    raw = f"{os.path.abspath(path)}::{stat.st_mtime_ns}::{stat.st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_summary_cache(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_summary_cache(path: str, cache: Dict[str, str]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# =========================================================
# LLM CLIENT
# =========================================================

class YandexLLMClient:
    def __init__(self, config: RAGConfig):
        self.config = config
        if not self.config.yandex_cloud_folder:
            raise RuntimeError("Не задан YANDEX_CLOUD_FOLDER")
        if not self.config.yandex_api_key:
            raise RuntimeError("Не задан YANDEX_API_KEY")

        self.model_uri = f"gpt://{self.config.yandex_cloud_folder}/yandexgpt/latest"
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def completion(
        self,
        system_text: str,
        user_text: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
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
# QUERY ROUTING
# =========================================================

def looks_like_quote(user_query: str) -> bool:
    if len(user_query) >= 220:
        return True
    if user_query.count("\n") >= 2:
        return True
    if re.match(r"^\s*\d+\.\s", user_query):
        return True
    if user_query.strip().endswith(":"):
        return True
    return False


def looks_like_complex_question(user_query: str) -> bool:
    q = user_query.lower()
    score = 0
    if len(q) > 130:
        score += 1
    if q.count("?") > 1:
        score += 1
    if " и " in q:
        score += 1
    if any(x in q for x in ["перечисли", "сравни", "условия", "основания", "сроки", "порядок"]):
        score += 1
    return score >= 2


def route_query(user_query: str) -> str:
    if looks_like_quote(user_query):
        return "quote"
    if looks_like_complex_question(user_query):
        return "complex"
    return "simple"


DATE_PATTERN = re.compile(
    r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"
    r"|\b\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"\s+\d{4}\s+г(?:ода|\\.)?"
    r"|\b\d{4}\s+г(?:ода|\\.)?"
    r"|\b\d{4}-\d{4}\b",
    flags=re.IGNORECASE,
)


def looks_like_date_question(user_query: str) -> bool:
    q = user_query.lower()
    markers = [
        "дата",
        "даты",
        "срок",
        "сроки",
        "когда",
        "период",
        "году",
        "год",
        "дедлайн",
        "deadline",
        "до какого",
        "с какого",
        "по какое",
    ]
    return any(marker in q for marker in markers)


def normalize_date_candidate(value: str) -> Optional[str]:
    item = re.sub(r"\s+", " ", value).strip(" .,;:()[]")
    if not item:
        return None

    range_match = re.fullmatch(r"(\d{4})-(\d{4})", item)
    if range_match:
        left = int(range_match.group(1))
        right = int(range_match.group(2))
        if 1900 <= left <= 2100 and 1900 <= right <= 2100:
            return item
        return None

    year_match = re.search(r"(\d{4})", item)
    if not year_match:
        return None

    year = int(year_match.group(1))
    if year < 1900 or year > 2100:
        return None

    return item


# =========================================================
# DIRECT QUOTE EXTRACTION
# =========================================================

def try_extract_point_by_quote(raw_text: str, user_query: str) -> Optional[str]:
    q = user_query.strip()
    if not q:
        return None

    anchor = re.sub(r"\s+", " ", q)[:140].strip()
    if len(anchor) < 40:
        return None

    pattern = re.escape(anchor)
    pattern = re.sub(r"\\ ", r"\\s+", pattern)

    m = re.search(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    start = m.start()

    mnum = re.match(r"^\s*(\d+)\.\s", q)
    if mnum:
        n = int(mnum.group(1))
        next_pat = rf"\n\s*{n+1}\.\s"
        mnext = re.search(next_pat, raw_text[start + 1:], flags=re.DOTALL)
        end = start + 1 + mnext.start() if mnext else min(len(raw_text), start + 4500)
        return raw_text[start:end].strip()

    mnext = re.search(r"\n\s*\d+\.\s", raw_text[start + 1:], flags=re.DOTALL)
    end = start + 1 + mnext.start() if mnext else min(len(raw_text), start + 4500)
    return raw_text[start:end].strip()


# =========================================================
# DOCUMENT STRUCTURE
# =========================================================

def split_into_sections(raw_text: str, source_file: str) -> List[Document]:
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
                    "section_id": 0,
                    "section_title": "Документ",
                },
            )
        ]

    return sections


def chunk_sections(
    sections: List[Document],
    config: RAGConfig,
) -> Tuple[List[Document], Dict[int, List[Document]]]:
    splitter = make_splitter(config)
    all_chunks: List[Document] = []
    section_to_chunks: Dict[int, List[Document]] = {}
    global_chunk_id = 0

    for sec in sections:
        sec_id = int(sec.metadata["section_id"])
        sec_title = str(sec.metadata["section_title"])
        sec_source = str(sec.metadata["source"])

        parts = splitter.split_documents([
            Document(
                page_content=sec.page_content,
                metadata={
                    "source": sec_source,
                    "section_id": sec_id,
                    "section_title": sec_title,
                },
            )
        ])

        bucket: List[Document] = []
        for d in parts:
            d.metadata["chunk_id"] = global_chunk_id
            bucket.append(d)
            all_chunks.append(d)
            global_chunk_id += 1

        section_to_chunks[sec_id] = bucket

    return all_chunks, section_to_chunks


# =========================================================
# RERANKER
# =========================================================

class SimpleReranker:
    def __init__(self, config: RAGConfig, llm_client: Optional[YandexLLMClient] = None):
        self.config = config
        self.llm_client = llm_client
        self.cross_encoder = None

        if self.config.enable_rerank and HAS_CROSS_ENCODER:
            try:
                reranker_source = resolve_hf_model_source(
                    repo_id=self.config.reranker_model,
                    explicit_path=self.config.reranker_model_path,
                    cache_dir=self.config.hf_cache_dir,
                )
                self.cross_encoder = CrossEncoder(
                    reranker_source,
                    local_files_only=self.config.local_files_only,
                )
            except Exception:
                self.cross_encoder = None

    def rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        if not self.config.enable_rerank or not docs:
            return docs[:top_n]

        if self.cross_encoder is not None:
            try:
                pairs = [(query, d.page_content[:1500]) for d in docs]
                scores = self.cross_encoder.predict(pairs)
                ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
                return [d for d, _ in ranked[:top_n]]
            except Exception:
                pass

        if self.llm_client is not None:
            return self._llm_rerank(query, docs, top_n)

        return docs[:top_n]

    def _llm_rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        short_blocks = []
        for i, d in enumerate(docs, start=1):
            txt = d.page_content[:700].replace("\n", " ")
            short_blocks.append(f"{i}. {txt}")

        system_text = (
            "Ты реранкер для RAG.\n"
            "Выбери фрагменты, которые лучше всего отвечают на вопрос.\n"
            f"Верни только номера лучших фрагментов, не больше {top_n}, через запятую."
        )
        user_text = f"ВОПРОС:\n{query}\n\nФРАГМЕНТЫ:\n" + "\n".join(short_blocks)

        try:
            out = self.llm_client.completion(system_text, user_text, temperature=0.0, max_tokens=120)
            nums = [int(x) for x in re.findall(r"\d+", out)]
            chosen = []
            seen = set()
            for n in nums:
                if 1 <= n <= len(docs) and n not in seen:
                    seen.add(n)
                    chosen.append(docs[n - 1])
                if len(chosen) >= top_n:
                    break
            return chosen if chosen else docs[:top_n]
        except Exception:
            return docs[:top_n]


# =========================================================
# DOCUMENT INDEX
# =========================================================

class DocumentIndex:
    def __init__(
        self,
        source_file: str,
        raw_text: str,
        file_fp: str,
        sections: List[Document],
        summary_docs: List[Document],
        summary_bm25: BM25Retriever,
        summary_db: Chroma,
        chunk_docs: List[Document],
        section_to_chunks: Dict[int, List[Document]],
        chunk_index: Dict[Tuple[str, int], Document],
        bm25: BM25Retriever,
    ):
        self.source_file = source_file
        self.raw_text = raw_text
        self.file_fp = file_fp

        self.sections = sections
        self.summary_docs = summary_docs
        self.summary_bm25 = summary_bm25
        self.summary_db = summary_db

        self.chunk_docs = chunk_docs
        self.section_to_chunks = section_to_chunks
        self.chunk_index = chunk_index
        self.bm25 = bm25


# =========================================================
# MAIN SERVICE
# =========================================================

class RAGService:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        embedding_model_source = resolve_hf_model_source(
            repo_id=self.config.embedding_model,
            explicit_path=self.config.embedding_model_path,
            cache_dir=self.config.hf_cache_dir,
        )
        self.embedding_func = HuggingFaceEmbeddings(
            model_name=embedding_model_source,
            cache_folder=self.config.hf_cache_dir,
            model_kwargs={
                "device": self.config.device,
                "local_files_only": self.config.local_files_only,
            },
            encode_kwargs={"normalize_embeddings": True},
        )

        self.db = Chroma(
            persist_directory=self.config.db_path,
            collection_name=self.config.collection_name,
            embedding_function=self.embedding_func,
        )
        self.summary_store = Chroma(
            persist_directory=self.config.db_path,
            collection_name=self.config.summary_collection_name,
            embedding_function=self.embedding_func,
        )

        self.llm_client = YandexLLMClient(self.config)
        self.reranker = SimpleReranker(self.config, self.llm_client)
        self.compare_service = CompareService(
            CompareConfig(bundle_db_path=self.config.bundle_db_path)
        )

    def _log_llm_failure(
        self,
        stage: str,
        error: Exception,
        user_query: str = "",
        context_preview: str = "",
    ) -> None:
        logger.warning(
            "llm_failure stage=%s query=%r error=%s context_preview=%r",
            stage,
            user_query[:300],
            str(error),
            context_preview[:1200],
        )

    def _extract_date_snippets_from_text(
        self,
        text: str,
        max_items: int = 24,
    ) -> List[str]:
        snippets: List[str] = []
        seen = set()

        normalized = text.replace("\r\n", "\n")
        blocks = re.split(r"\n{2,}", normalized)
        for block in blocks:
            clean = re.sub(r"\s+", " ", block).strip()
            if len(clean) < 20:
                continue
            if not DATE_PATTERN.search(clean):
                continue

            key = clean.lower()
            if key in seen:
                continue

            seen.add(key)
            snippets.append(clean[:700])
            if len(snippets) >= max_items:
                break

        return snippets

    def _build_date_context_from_index(
        self,
        doc_index: "DocumentIndex",
        max_chars: int = 12000,
    ) -> str:
        parts: List[str] = []
        total = 0

        for i, snippet in enumerate(self._extract_date_snippets_from_text(doc_index.raw_text), start=1):
            block = f"[{i}] source={doc_index.source_file} section=date-scan\n{snippet}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block) + 2

        return "\n\n---\n\n".join(parts)

    def _collect_date_evidence(
        self,
        doc_index: "DocumentIndex",
        max_items: int = 12,
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        seen = set()

        for snippet in self._extract_date_snippets_from_text(doc_index.raw_text, max_items=max_items * 3):
            raw_dates = DATE_PATTERN.findall(snippet)
            dates = uniq_keep_order(
                normalized
                for normalized in (normalize_date_candidate(item) for item in raw_dates)
                if normalized
            )
            if not dates:
                continue

            key = snippet.lower()
            if key in seen:
                continue
            seen.add(key)

            evidence.append(
                {
                    "index": len(evidence) + 1,
                    "source": doc_index.source_file,
                    "section_title": "date-scan",
                    "section_id": None,
                    "chunk_id": None,
                    "start_index": None,
                    "text": snippet,
                    "dates": dates,
                }
            )
            if len(evidence) >= max_items:
                break

        return evidence

    def _answer_date_question(
        self,
        doc_index: "DocumentIndex",
        user_query: str,
        answer_mode: str,
    ) -> Dict[str, Any]:
        evidence = self._collect_date_evidence(doc_index)
        all_dates: List[str] = []
        for item in evidence:
            all_dates.extend(item.get("dates", []))
        all_dates = uniq_keep_order(all_dates)

        if not all_dates:
            return {
                "answer": "В документе не удалось уверенно найти даты.",
                "context": "",
                "sources": [],
                "route": "date",
                "answer_source": "deterministic",
                "llm_error": "",
                "debug": {
                    "route": "date",
                    "is_date_query": True,
                    "answer_source": "deterministic",
                    "dates_found": 0,
                },
            }

        limit = 8 if answer_mode == "РљСЂР°С‚РєРѕ" else 20
        visible_dates = all_dates[:limit]
        lines = [f"- {date}" for date in visible_dates]
        answer = "Найдённые даты в документе:\n" + "\n".join(lines)
        if len(all_dates) > len(visible_dates):
            answer += f"\n\nПоказаны первые {len(visible_dates)} дат из {len(all_dates)} найденных."

        context_parts = []
        for item in evidence[:6]:
            context_parts.append(
                f"[{item['index']}] source={item['source']} section=date-scan\n{item['text']}"
            )

        return {
            "answer": answer,
            "context": "\n\n---\n\n".join(context_parts),
            "sources": [
                {
                    "index": item["index"],
                    "source": item["source"],
                    "section_title": item["section_title"],
                    "section_id": item["section_id"],
                    "chunk_id": item["chunk_id"],
                    "start_index": item["start_index"],
                    "text": item["text"],
                }
                for item in evidence
            ],
            "route": "date",
            "answer_source": "deterministic",
            "llm_error": "",
            "debug": {
                "route": "date",
                "is_date_query": True,
                "answer_source": "deterministic",
                "dates_found": len(all_dates),
                "dates_preview": visible_dates,
            },
        }

    def _fallback_answer_from_context(
        self,
        context: str,
        user_query: str,
    ) -> str:
        if looks_like_date_question(user_query):
            matches = DATE_PATTERN.findall(context)
            dates = uniq_keep_order(matches)
            if dates:
                visible = dates[:20]
                bullets = "\n".join(f"- {item}" for item in visible)
                suffix = ""
                if len(dates) > len(visible):
                    suffix = f"\n\nПоказаны первые {len(visible)} дат из {len(dates)} найденных в контексте."
                return f"LLM недоступен, поэтому показываю даты, найденные в контексте:\n{bullets}{suffix}"

        if context.strip():
            return (
                "LLM недоступен. Поиск нашёл релевантные фрагменты, "
                "но автоматически сформировать надёжный ответ без модели не удалось."
            )

        return "LLM недоступен, и поиск не собрал достаточный контекст для ответа."

    def _generate_answer_result(
        self,
        context: str,
        user_query: str,
        answer_mode: str = "Подробно",
    ) -> Dict[str, Any]:
        try:
            return {
                "answer": self.generate_answer(context, user_query, answer_mode=answer_mode),
                "answer_source": "llm",
                "llm_error": "",
            }
        except Exception as exc:
            self._log_llm_failure(
                stage="generate_answer",
                error=exc,
                user_query=user_query,
                context_preview=context,
            )
            return {
                "answer": self._fallback_answer_from_context(context, user_query),
                "answer_source": "fallback",
                "llm_error": str(exc),
            }

    def _build_planner_outline(
        self,
        doc_index: "DocumentIndex",
        max_sections: int = 18,
        max_chars_per_section: int = 220,
    ) -> str:
        lines: List[str] = []
        for sec in doc_index.summary_docs[:max_sections]:
            section_id = sec.metadata.get("section_id")
            title = str(sec.metadata.get("section_title", "section")).strip()
            text = re.sub(r"\s+", " ", sec.page_content).strip()[:max_chars_per_section]
            lines.append(f"{section_id}. {title}: {text}")
        return "\n".join(lines)

    def _resolve_section_title_hints(
        self,
        doc_index: "DocumentIndex",
        title_hints: List[str],
    ) -> List[int]:
        if not title_hints:
            return []

        out: List[int] = []
        lowered_hints = [hint.lower().strip() for hint in title_hints if hint.strip()]
        for sec in doc_index.sections:
            title = str(sec.metadata.get("section_title", "")).lower().strip()
            if title and any(hint in title or title in hint for hint in lowered_hints):
                out.append(int(sec.metadata.get("section_id", 0)))

        seen = set()
        result: List[int] = []
        for sid in out:
            if sid not in seen:
                seen.add(sid)
                result.append(sid)
        return result

    def plan_retrieval(
        self,
        user_query: str,
        doc_index: "DocumentIndex",
    ) -> Dict[str, Any]:
        outline = self._build_planner_outline(doc_index)
        if not outline.strip():
            return {"dense_queries": [], "bm25_queries": [], "section_titles": []}

        system_text = (
            "Ты помогаешь спланировать поиск ответа по документу.\n"
            "Не отвечай на вопрос. Верни только JSON вида:\n"
            '{"dense_queries":["..."],"bm25_queries":["..."],"section_titles":["..."]}\n'
            "Правила:\n"
            "1) dense_queries: до 4 переформулировок вопроса языком документа.\n"
            "2) bm25_queries: до 4 коротких лексических запросов, как это могло быть написано в документе.\n"
            "3) section_titles: до 5 названий разделов из списка, где вероятнее всего искать ответ.\n"
            "4) Не придумывай факты.\n"
            "5) Если вопрос общий, делай запросы более приземленными и ориентированными на фактические фрагменты."
        )
        user_text = f"ВОПРОС:\n{user_query}\n\nРАЗДЕЛЫ ДОКУМЕНТА:\n{outline}"

        try:
            raw = self.llm_client.completion(
                system_text,
                user_text,
                temperature=0.0,
                max_tokens=500,
            )
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not m:
                raise ValueError("planner JSON not found")

            data = json.loads(m.group(0))
            return {
                "dense_queries": uniq_keep_order([str(x) for x in data.get("dense_queries", [])])[:4],
                "bm25_queries": uniq_keep_order([str(x) for x in data.get("bm25_queries", [])])[:4],
                "section_titles": uniq_keep_order([str(x) for x in data.get("section_titles", [])])[:5],
            }
        except Exception as exc:
            self._log_llm_failure(
                stage="plan_retrieval",
                error=exc,
                user_query=user_query,
                context_preview=outline,
            )
            return {"dense_queries": [], "bm25_queries": [], "section_titles": []}

    def _scan_section_hints(
        self,
        doc_index: "DocumentIndex",
        queries: List[str],
        top_n: int = 5,
    ) -> List[int]:
        stopwords = {
            "что", "какие", "какой", "какая", "какое", "каких", "все", "всё", "в", "на",
            "по", "из", "для", "и", "или", "о", "об", "про", "как", "перечисли", "найди",
            "документе", "документ", "есть",
        }
        terms: List[str] = []
        for query in queries:
            for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{3,}", query.lower()):
                if token not in stopwords:
                    terms.append(token)

        if not terms:
            return []

        scores: Dict[int, float] = {}
        for sec in doc_index.sections:
            section_id = int(sec.metadata.get("section_id", 0))
            haystack = f"{sec.metadata.get('section_title', '')} {sec.page_content}".lower()
            score = 0.0
            for term in terms:
                hits = haystack.count(term)
                if hits:
                    score += min(hits, 4)
                    if term in str(sec.metadata.get("section_title", "")).lower():
                        score += 1.5
            if score > 0:
                scores[section_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in ranked[:top_n]]

    def _filter_low_information_docs(
        self,
        docs: List[Document],
        min_content_chars: int = 80,
    ) -> List[Document]:
        filtered: List[Document] = []
        for doc in docs:
            text = re.sub(r"\s+", " ", doc.page_content).strip()
            alnum = re.sub(r"[^A-Za-zА-Яа-яЁё0-9]+", "", text)
            if len(text) >= min_content_chars or len(alnum) >= 50:
                filtered.append(doc)

        return filtered if filtered else docs

    def _inject_section_hint_docs(
        self,
        docs: List[Document],
        section_hints: List[int],
        section_to_chunks: Dict[int, List[Document]],
        per_section: int = 2,
    ) -> List[Document]:
        out: List[Document] = []
        seen = set()

        for doc in docs:
            key = self.doc_key(doc)
            if key not in seen:
                seen.add(key)
                out.append(doc)

        for section_id in section_hints[:5]:
            injected = 0
            for chunk in self._filter_low_information_docs(section_to_chunks.get(section_id, [])):
                key = self.doc_key(chunk)
                if key in seen:
                    continue
                seen.add(key)
                out.append(chunk)
                injected += 1
                if injected >= per_section:
                    break

        return out

    # ----------------------------
    # PUBLIC API
    # ----------------------------

    def list_available_documents(self) -> List[Dict[str, str]]:
        try:
            data = self.db.get(include=["metadatas"])
        except Exception:
            return []

        documents: List[Dict[str, str]] = []
        seen = set()

        for meta in data.get("metadatas", []):
            if not meta:
                continue

            doc_id = str(meta.get("doc_id", "")).strip()
            source = str(meta.get("source", "")).strip()
            document_name = str(meta.get("document_name", "")).strip()
            version = str(meta.get("version", "")).strip()

            key = (doc_id, source, document_name, version)
            if not doc_id or key in seen:
                continue

            seen.add(key)
            documents.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "document_name": document_name,
                    "version": version,
                }
            )

        documents.sort(key=lambda x: (x["document_name"], x["version"], x["source"]))
        return documents

    @staticmethod
    def _deserialize_documents(items: List[Dict[str, Any]]) -> List[Document]:
        docs: List[Document] = []
        for item in items:
            docs.append(
                Document(
                    page_content=str(item.get("page_content", "")),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return docs

    def _bundle_path(self, doc_id: str) -> str:
        return os.path.join(self.config.processed_dir, f"{doc_id}.json")

    def _load_document_bundle(self, doc_id: str) -> Optional[Dict[str, Any]]:
        bundle = load_bundle(doc_id, db_path=self.config.bundle_db_path)
        if bundle is not None:
            return bundle

        bundle_path = self._bundle_path(doc_id)
        if not os.path.exists(bundle_path):
            return None

        try:
            with open(bundle_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _build_document_index_from_parts(
        self,
        source_file: str,
        raw_text: str,
        file_fp: str,
        sections: List[Document],
        summary_docs: List[Document],
        chunk_docs: List[Document],
    ) -> DocumentIndex:
        sections = sorted(sections, key=lambda d: int(d.metadata.get("section_id", 0)))
        summary_docs = sorted(summary_docs, key=lambda d: int(d.metadata.get("section_id", 0)))
        chunk_docs = sorted(chunk_docs, key=lambda d: int(d.metadata.get("chunk_id", 0)))

        if summary_docs:
            summary_bm25 = BM25Retriever.from_documents(summary_docs)
            summary_bm25.k = self.config.summary_bm25_k
            summary_db = Chroma(
                collection_name=f"section_summaries_{file_fp}",
                embedding_function=self.embedding_func,
            )
            summary_ids = [f"summary:{d.metadata['section_id']}" for d in summary_docs]
            summary_db.add_documents(summary_docs, ids=summary_ids)
        else:
            summary_docs, summary_bm25, summary_db = self._build_section_summary_index(
                sections=sections,
                file_fp=file_fp,
            )

        section_to_chunks: Dict[int, List[Document]] = {}
        for d in chunk_docs:
            sid = int(d.metadata.get("section_id", 0))
            section_to_chunks.setdefault(sid, []).append(d)

        chunk_index = {
            (str(d.metadata["source"]), int(d.metadata["chunk_id"])): d
            for d in chunk_docs
        }

        bm25 = BM25Retriever.from_documents(chunk_docs)
        bm25.k = self.config.bm25_k

        return DocumentIndex(
            source_file=source_file,
            raw_text=raw_text,
            file_fp=file_fp,
            sections=sections,
            summary_docs=summary_docs,
            summary_bm25=summary_bm25,
            summary_db=summary_db,
            chunk_docs=chunk_docs,
            section_to_chunks=section_to_chunks,
            chunk_index=chunk_index,
            bm25=bm25,
        )

    def build_document_index_from_db(self, doc_id: str) -> DocumentIndex:
        bundle = self._load_document_bundle(doc_id)
        if bundle is not None:
            source_file = str(bundle.get("source", ""))
            raw_text = preprocess_text(str(bundle.get("raw_text", "")))
            sections = self._deserialize_documents(bundle.get("sections", []))
            summary_docs = self._deserialize_documents(bundle.get("summary_docs", []))
            chunk_docs = self._deserialize_documents(bundle.get("chunk_docs", []))

            if chunk_docs:
                return self._build_document_index_from_parts(
                    source_file=source_file,
                    raw_text=raw_text,
                    file_fp=doc_id,
                    sections=sections,
                    summary_docs=summary_docs,
                    chunk_docs=chunk_docs,
                )

        data = self.db.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        if not documents or not metadatas:
            raise ValueError(f"Документ с doc_id={doc_id} не найден в БД")

        chunk_docs: List[Document] = []
        for page_content, metadata in zip(documents, metadatas):
            if not metadata:
                continue
            chunk_docs.append(
                Document(
                    page_content=str(page_content),
                    metadata=dict(metadata),
                )
            )

        if not chunk_docs:
            raise ValueError(f"В БД нет чанков для doc_id={doc_id}")

        source_file = str(chunk_docs[0].metadata.get("source", ""))
        raw_text = self._load_raw_text_for_doc(source_file, chunk_docs)

        summary_data = self.summary_store.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )

        summary_docs: List[Document] = []
        for page_content, metadata in zip(
            summary_data.get("documents", []),
            summary_data.get("metadatas", []),
        ):
            if not metadata:
                continue
            summary_docs.append(
                Document(
                    page_content=str(page_content),
                    metadata=dict(metadata),
                )
            )

        if summary_docs:
            sections = [
                Document(
                    page_content=d.page_content,
                    metadata=dict(d.metadata),
                )
                for d in summary_docs
            ]
        else:
            sections = split_into_sections(raw_text, source_file)

        return self._build_document_index_from_parts(
            source_file=source_file,
            raw_text=raw_text,
            file_fp=doc_id,
            sections=sections,
            summary_docs=summary_docs,
            chunk_docs=chunk_docs,
        )

    def build_document_index(self, source_file: str) -> DocumentIndex:
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Файл не найден: {source_file}")

        with open(source_file, "r", encoding="utf-8") as f:
            raw_text = preprocess_text(f.read())

        file_fp = file_fingerprint(source_file)
        sections = split_into_sections(raw_text, source_file)

        summary_docs, summary_bm25, summary_db = self._build_section_summary_index(
            sections=sections,
            file_fp=file_fp,
        )

        chunk_docs, section_to_chunks = chunk_sections(sections, self.config)

        chunk_index = {
            (str(d.metadata["source"]), int(d.metadata["chunk_id"])): d
            for d in chunk_docs
        }

        bm25 = BM25Retriever.from_documents(chunk_docs)
        bm25.k = self.config.bm25_k

        return DocumentIndex(
            source_file=source_file,
            raw_text=raw_text,
            file_fp=file_fp,
            sections=sections,
            summary_docs=summary_docs,
            summary_bm25=summary_bm25,
            summary_db=summary_db,
            chunk_docs=chunk_docs,
            section_to_chunks=section_to_chunks,
            chunk_index=chunk_index,
            bm25=bm25,
        )

    def _load_raw_text_for_doc(self, source_file: str, chunk_docs: List[Document]) -> str:
        if source_file and os.path.exists(source_file):
            with open(source_file, "r", encoding="utf-8") as f:
                return preprocess_text(f.read())

        ordered_chunks = sorted(chunk_docs, key=lambda d: int(d.metadata.get("chunk_id", 0)))
        return "\n\n".join(d.page_content for d in ordered_chunks)

    @staticmethod
    def _serialize_source_docs(docs: List[Document]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for index, doc in enumerate(docs, start=1):
            sources.append(
                {
                    "index": index,
                    "source": str(doc.metadata.get("source", "")),
                    "section_title": str(doc.metadata.get("section_title", "")),
                    "section_id": doc.metadata.get("section_id"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "start_index": doc.metadata.get("start_index"),
                    "text": doc.page_content.strip(),
                }
            )
        return sources

    def ask_question(
        self,
        doc_index: DocumentIndex,
        user_query: str,
        answer_mode: str = "Подробно",
    ) -> Dict[str, Any]:
        user_query = user_query.strip()
        if not user_query:
            raise ValueError("Пустой вопрос")

        route = route_query(user_query)
        debug_info: Dict[str, Any] = {"route": route}

        working_query = user_query
        if route != "quote":
            working_query = normalize_query(working_query)

        direct = None
        corrected_quote = None

        # 1. quote-route
        if route == "quote":
            direct = try_extract_point_by_quote(doc_index.raw_text, working_query)

            if not direct and self.config.enable_query_correction:
                corrected_quote = self.correct_query_typos(working_query)
                if corrected_quote and corrected_quote != working_query:
                    direct = try_extract_point_by_quote(doc_index.raw_text, corrected_quote)

            if direct:
                context = f"[1] source={doc_index.source_file} (direct extract)\n{direct}"
                answer_result = self._generate_answer_result(
                    context,
                    user_query,
                    answer_mode=answer_mode,
                )

                debug_info["direct_extract"] = True
                debug_info["context_preview"] = context[:2500]
                debug_info["answer_source"] = answer_result["answer_source"]
                debug_info["llm_error"] = answer_result["llm_error"]

                return {
                    "answer": answer_result["answer"],
                    "context": context,
                    "sources": [
                        {
                            "index": 1,
                            "source": doc_index.source_file,
                            "section_title": "direct extract",
                            "section_id": None,
                            "chunk_id": None,
                            "start_index": None,
                            "text": direct,
                        }
                    ],
                    "route": route,
                    "answer_source": answer_result["answer_source"],
                    "llm_error": answer_result["llm_error"],
                    "debug": debug_info,
                }

        # 2. query improvement
        corrected_query = working_query
        if self.config.enable_query_correction:
            corrected_query = self.correct_query_typos(working_query)

        dense_queries, bm25_queries = self.generate_query_variants(corrected_query)

        dense_queries = uniq_keep_order(
            [working_query, corrected_query] + dense_queries
        )[:self.config.max_query_variants]

        bm25_queries = uniq_keep_order(
            [working_query, corrected_query] + bm25_queries
        )[:self.config.max_query_variants]

        subquestions: List[str] = []
        if route == "complex":
            subquestions = self.decompose_question(corrected_query)

        planner = self.plan_retrieval(user_query, doc_index)
        planner_dense = planner.get("dense_queries", [])
        planner_bm25 = planner.get("bm25_queries", [])
        planner_section_titles = planner.get("section_titles", [])

        dense_queries = uniq_keep_order(dense_queries + planner_dense)[:self.config.max_query_variants]
        bm25_queries = uniq_keep_order(bm25_queries + planner_bm25)[:self.config.max_query_variants]
        planner_section_hints = self._resolve_section_title_hints(doc_index, planner_section_titles)
        scanned_section_hints = self._scan_section_hints(
            doc_index,
            dense_queries + bm25_queries,
            top_n=5,
        )

        debug_info["dense_queries"] = dense_queries
        debug_info["bm25_queries"] = bm25_queries
        debug_info["subquestions"] = subquestions
        debug_info["planner"] = planner
        debug_info["scanned_section_hints"] = scanned_section_hints

        # 3. section hints
        section_hints = self.retrieve_section_hints(
            dense_queries=dense_queries,
            bm25_queries=bm25_queries,
            summary_bm25=doc_index.summary_bm25,
            summary_db=doc_index.summary_db,
            source_file=doc_index.source_file,
        )
        section_hints = uniq_keep_order(
            [str(x) for x in (planner_section_hints + scanned_section_hints + section_hints)]
        )[: self.config.max_sections_to_boost + len(planner_section_hints) + len(scanned_section_hints)]
        section_hints = [int(x) for x in section_hints]

        # 4. main retrieval
        fused = self.retrieve_main_docs(
            dense_queries=dense_queries,
            bm25_queries=bm25_queries,
            source_file=doc_index.source_file,
            bm25=doc_index.bm25,
            section_hints=section_hints,
            section_to_chunks=doc_index.section_to_chunks,
        )

        # 5. subquestions for complex route
        if subquestions:
            extra_ranked = []
            for sq in subquestions:
                sq_dense, sq_bm25 = self.generate_query_variants(sq)
                sq_dense = uniq_keep_order([sq] + sq_dense)[:3]
                sq_bm25 = uniq_keep_order([sq] + sq_bm25)[:2]

                sq_sections = self.retrieve_section_hints(
                    dense_queries=sq_dense,
                    bm25_queries=sq_bm25,
                    summary_bm25=doc_index.summary_bm25,
                    summary_db=doc_index.summary_db,
                    source_file=doc_index.source_file,
                )

                sq_docs = self.retrieve_main_docs(
                    dense_queries=sq_dense,
                    bm25_queries=sq_bm25,
                    source_file=doc_index.source_file,
                    bm25=doc_index.bm25,
                    section_hints=sq_sections,
                    section_to_chunks=doc_index.section_to_chunks,
                )
                extra_ranked.append(sq_docs)

            if extra_ranked:
                fused = self.rrf_fuse([fused] + extra_ranked)[:self.config.top_docs_after_fusion]

        fused = self._filter_low_information_docs(fused)
        reranked = self.reranker.rerank(corrected_query, fused, top_n=self.config.rerank_top_n)
        reranked = self._filter_low_information_docs(reranked)
        expanded = self.add_neighbors(reranked, doc_index.chunk_index, window=self.config.neighbor_window)
        expanded = self._filter_low_information_docs(expanded)
        expanded = self._inject_section_hint_docs(
            expanded,
            section_hints,
            doc_index.section_to_chunks,
        )
        context = self.build_context(expanded)

        answer_result = self._generate_answer_result(
            context,
            user_query,
            answer_mode=answer_mode,
        )

        debug_info["section_hints"] = section_hints
        debug_info["fused_count"] = len(fused)
        debug_info["reranked_count"] = len(reranked)
        debug_info["expanded_count"] = len(expanded)
        debug_info["context_preview"] = context[:2500]
        debug_info["answer_source"] = answer_result["answer_source"]
        debug_info["llm_error"] = answer_result["llm_error"]

        return {
            "answer": answer_result["answer"],
            "context": context,
            "sources": self._serialize_source_docs(expanded),
            "route": route,
            "answer_source": answer_result["answer_source"],
            "llm_error": answer_result["llm_error"],
            "debug": debug_info,
        }

    def ask_question_across_indexes(
        self,
        doc_indexes: List[DocumentIndex],
        user_query: str,
        answer_mode: str = "Подробно",
    ) -> Dict[str, Any]:
        user_query = user_query.strip()
        if not user_query:
            raise ValueError("Пустой вопрос")
        if not doc_indexes:
            raise ValueError("Нет документов для поиска")

        route = route_query(user_query)
        if is_date_query:
            merged_sources: List[Dict[str, Any]] = []
            merged_dates: List[str] = []
            context_parts: List[str] = []

            for doc_index in doc_indexes:
                single = self._answer_date_question(doc_index, user_query, answer_mode)
                merged_sources.extend(single.get("sources", []))
                merged_dates.extend(re.findall(DATE_PATTERN, single.get("answer", "")))
                if single.get("context"):
                    context_parts.append(single["context"])

            merged_dates = uniq_keep_order(merged_dates)
            limit = 8 if answer_mode == "РљСЂР°С‚РєРѕ" else 20
            visible_dates = merged_dates[:limit]
            answer = "Найдённые даты в выбранных документах:\n" + "\n".join(
                f"- {date}" for date in visible_dates
            )
            if len(merged_dates) > len(visible_dates):
                answer += f"\n\nПоказаны первые {len(visible_dates)} дат из {len(merged_dates)} найденных."

            return {
                "answer": answer,
                "context": "\n\n---\n\n".join(context_parts),
                "sources": merged_sources[:12],
                "route": "date",
                "answer_source": "deterministic",
                "llm_error": "",
                "debug": {
                    "route": "date",
                    "is_date_query": True,
                    "answer_source": "deterministic",
                    "documents_searched": len(doc_indexes),
                    "dates_found": len(merged_dates),
                    "dates_preview": visible_dates,
                },
            }

        working_query = normalize_query(user_query) if route != "quote" else user_query

        corrected_query = working_query
        if self.config.enable_query_correction and route != "quote":
            corrected_query = self.correct_query_typos(working_query)

        dense_queries, bm25_queries = self.generate_query_variants(corrected_query)
        dense_queries = uniq_keep_order(
            [working_query, corrected_query] + dense_queries
        )[:self.config.max_query_variants]
        bm25_queries = uniq_keep_order(
            [working_query, corrected_query] + bm25_queries
        )[:self.config.max_query_variants]

        subquestions: List[str] = []
        if route == "complex":
            subquestions = self.decompose_question(corrected_query)

        ranked_lists: List[List[Document]] = []
        combined_chunk_index: Dict[Tuple[str, int], Document] = {}
        section_hints_by_doc: Dict[str, List[int]] = {}

        for doc_index in doc_indexes:
            combined_chunk_index.update(doc_index.chunk_index)
            section_hints = self.retrieve_section_hints(
                dense_queries=dense_queries,
                bm25_queries=bm25_queries,
                summary_bm25=doc_index.summary_bm25,
                summary_db=doc_index.summary_db,
                source_file=doc_index.source_file,
            )
            section_hints_by_doc[doc_index.source_file] = section_hints
            docs = self.retrieve_main_docs(
                dense_queries=dense_queries,
                bm25_queries=bm25_queries,
                source_file=doc_index.source_file,
                bm25=doc_index.bm25,
                section_hints=section_hints,
                section_to_chunks=doc_index.section_to_chunks,
            )
            ranked_lists.append(docs)

            for sq in subquestions:
                sq_dense, sq_bm25 = self.generate_query_variants(sq)
                sq_dense = uniq_keep_order([sq] + sq_dense)[:3]
                sq_bm25 = uniq_keep_order([sq] + sq_bm25)[:2]
                sq_sections = self.retrieve_section_hints(
                    dense_queries=sq_dense,
                    bm25_queries=sq_bm25,
                    summary_bm25=doc_index.summary_bm25,
                    summary_db=doc_index.summary_db,
                    source_file=doc_index.source_file,
                )
                sq_docs = self.retrieve_main_docs(
                    dense_queries=sq_dense,
                    bm25_queries=sq_bm25,
                    source_file=doc_index.source_file,
                    bm25=doc_index.bm25,
                    section_hints=sq_sections,
                    section_to_chunks=doc_index.section_to_chunks,
                )
                ranked_lists.append(sq_docs)

        fused = self.rrf_fuse(ranked_lists)[:self.config.top_docs_after_fusion]
        reranked = self.reranker.rerank(corrected_query, fused, top_n=self.config.rerank_top_n)
        expanded = self.add_neighbors(
            reranked,
            combined_chunk_index,
            window=self.config.neighbor_window,
        )
        context = self.build_context(expanded)
        if is_date_query:
            date_context_parts = []
            for doc_index in doc_indexes:
                date_context = self._build_date_context_from_index(doc_index, max_chars=4000)
                if date_context:
                    date_context_parts.append(date_context)
            if date_context_parts:
                context = (
                    f"{context}\n\n---\n\n" + "\n\n---\n\n".join(date_context_parts)
                    if context
                    else "\n\n---\n\n".join(date_context_parts)
                )

        answer_result = self._generate_answer_result(
            context,
            user_query,
            answer_mode=answer_mode,
        )

        debug_info: Dict[str, Any] = {
            "route": route,
            "is_date_query": is_date_query,
            "dense_queries": dense_queries,
            "bm25_queries": bm25_queries,
            "subquestions": subquestions,
            "documents_searched": len(doc_indexes),
            "section_hints_by_doc": section_hints_by_doc,
            "fused_count": len(fused),
            "reranked_count": len(reranked),
            "expanded_count": len(expanded),
            "context_preview": context[:2500],
            "answer_source": answer_result["answer_source"],
            "llm_error": answer_result["llm_error"],
        }

        return {
            "answer": answer_result["answer"],
            "context": context,
            "sources": self._serialize_source_docs(expanded),
            "route": route,
            "answer_source": answer_result["answer_source"],
            "llm_error": answer_result["llm_error"],
            "debug": debug_info,
        }

    # ----------------------------
    # QUERY IMPROVEMENT
    # ----------------------------

    def correct_query_typos(self, original: str) -> str:
        if not self.config.enable_query_correction:
            return original

        system_text = (
            "Исправь опечатки и орфографические ошибки в запросе пользователя.\n"
            "Не меняй смысл, не перефразируй, сохрани номера пунктов, даты и термины.\n"
            "Верни только исправленный текст одной строкой."
        )

        try:
            fixed = self.llm_client.completion(
                system_text,
                original,
                temperature=0.0,
                max_tokens=180,
            ).strip()
            if not fixed:
                return original
            fixed = re.sub(r"\s+", " ", fixed).strip()
            return fixed
        except Exception:
            return original

    def generate_query_variants(self, original: str) -> Tuple[List[str], List[str]]:
        base = normalize_query(original)

        if not self.config.enable_query_expansion:
            return [base], [base]

        system_text = (
            "Ты помогаешь строить варианты поискового запроса по документу.\n"
            "Нужно вернуть JSON вида:\n"
            '{"dense": ["..."], "bm25": ["..."]}\n'
            "Правила:\n"
            "1) dense — до 3 вариантов естественного вопроса без изменения смысла.\n"
            "2) bm25 — до 2 коротких keyword-запросов: ключевые термины, сущности, номера пунктов.\n"
            "3) Не добавляй новую информацию.\n"
            "4) Верни только JSON."
        )

        try:
            raw = self.llm_client.completion(
                system_text,
                base,
                temperature=0.15,
                max_tokens=300,
            )
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not m:
                raise ValueError("JSON not found")

            data = json.loads(m.group(0))
            dense = [base] + [str(x) for x in data.get("dense", [])]
            bm25 = [base] + [str(x) for x in data.get("bm25", [])]

            dense = uniq_keep_order(dense)[:self.config.max_query_variants]
            bm25 = uniq_keep_order(bm25)[:min(3, self.config.max_query_variants)]

            return dense, bm25
        except Exception:
            return [base], [base]

    def decompose_question(self, original: str) -> List[str]:
        if not self.config.enable_decomposition:
            return []

        system_text = (
            "Разбей сложный вопрос пользователя на 2-3 самостоятельных подвопроса для поиска по документу.\n"
            "Не добавляй новую информацию.\n"
            "Каждый подвопрос должен быть коротким и конкретным.\n"
            "Верни только список, каждый пункт с новой строки."
        )

        try:
            text = self.llm_client.completion(
                system_text,
                original,
                temperature=0.1,
                max_tokens=220,
            )
            items = [x.strip(" \t-•") for x in text.splitlines() if x.strip()]
            return uniq_keep_order(items)[:self.config.max_subquestions]
        except Exception:
            return []

    # ----------------------------
    # SUMMARIES
    # ----------------------------

    def summarize_section(
        self,
        section_doc: Document,
        file_fp: str,
        cache: Dict[str, str],
    ) -> str:
        text = section_doc.page_content.strip()
        title = section_doc.metadata.get("section_title", "Раздел")
        key_raw = (
            f"{file_fp}::{section_doc.metadata.get('section_id')}::"
            f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
        )
        cache_key = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()

        if cache_key in cache:
            return cache[cache_key]

        system_text = (
            "Сделай краткий summary раздела документа для поискового индекса.\n"
            "Нужно 3-5 предложений.\n"
            "Сохрани ключевые темы, сущности, условия, сроки, ограничения, номера пунктов, если они важны.\n"
            "Не добавляй новую информацию."
        )
        user_text = f"ЗАГОЛОВОК РАЗДЕЛА: {title}\n\nТЕКСТ РАЗДЕЛА:\n{text[:6000]}"

        try:
            summary = self.llm_client.completion(
                system_text,
                user_text,
                temperature=0.1,
                max_tokens=350,
            ).strip()
        except Exception:
            summary = text[:800]

        cache[cache_key] = summary
        return summary

    def _build_section_summary_index(
        self,
        sections: List[Document],
        file_fp: str,
    ) -> Tuple[List[Document], BM25Retriever, Chroma]:
        cache = load_summary_cache(self.config.summary_cache_file)
        summary_docs = []

        for sec in sections:
            if self.config.enable_section_summaries:
                summary = self.summarize_section(sec, file_fp, cache)
            else:
                summary = sec.page_content[:800]

            summary_docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        "source": sec.metadata["source"],
                        "section_id": sec.metadata["section_id"],
                        "section_title": sec.metadata["section_title"],
                    },
                )
            )

        save_summary_cache(self.config.summary_cache_file, cache)

        summary_bm25 = BM25Retriever.from_documents(summary_docs)
        summary_bm25.k = self.config.summary_bm25_k

        summary_db = Chroma(
            collection_name=f"section_summaries_{abs(hash(file_fp))}",
            embedding_function=self.embedding_func,
        )
        summary_ids = [f"summary:{d.metadata['section_id']}" for d in summary_docs]
        summary_db.add_documents(summary_docs, ids=summary_ids)

        return summary_docs, summary_bm25, summary_db

    # ----------------------------
    # RETRIEVAL
    # ----------------------------

    @staticmethod
    def doc_key(doc: Document) -> Tuple[str, int]:
        return (
            str(doc.metadata.get("source", "unknown")),
            int(doc.metadata.get("chunk_id", -1)),
        )

    def rrf_fuse(self, lists: List[List[Document]], k: int = 60) -> List[Document]:
        scores: Dict[Tuple[str, int], float] = {}
        by_key: Dict[Tuple[str, int], Document] = {}

        for docs in lists:
            for rank, d in enumerate(docs, start=1):
                key = self.doc_key(d)
                by_key[key] = d
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [by_key[key] for key, _ in fused]

    def retrieve_section_hints(
        self,
        dense_queries: List[str],
        bm25_queries: List[str],
        summary_bm25: BM25Retriever,
        summary_db: Chroma,
        source_file: str,
    ) -> List[int]:
        section_scores: Dict[int, float] = {}

        for q in dense_queries:
            try:
                docs = summary_db.similarity_search(
                    q,
                    k=self.config.summary_chroma_k,
                    filter={"source": source_file},
                )
                for rank, d in enumerate(docs, start=1):
                    sid = int(d.metadata["section_id"])
                    section_scores[sid] = section_scores.get(sid, 0.0) + 1.0 / (50 + rank)
            except Exception:
                pass

        for q in bm25_queries:
            try:
                docs = summary_bm25.invoke(q)
                for rank, d in enumerate(docs[:self.config.summary_bm25_k], start=1):
                    sid = int(d.metadata["section_id"])
                    section_scores[sid] = section_scores.get(sid, 0.0) + 1.0 / (50 + rank)
            except Exception:
                pass

        ranked = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in ranked[:self.config.max_sections_to_boost]]

    def retrieve_main_docs(
        self,
        dense_queries: List[str],
        bm25_queries: List[str],
        source_file: str,
        bm25: BM25Retriever,
        section_hints: List[int],
        section_to_chunks: Dict[int, List[Document]],
    ) -> List[Document]:
        ranked_lists: List[List[Document]] = []

        for q in dense_queries:
            try:
                chroma_docs = self.db.similarity_search(
                    q,
                    k=self.config.chroma_k,
                    filter={"source": source_file},
                )
                ranked_lists.append(chroma_docs)
            except Exception:
                pass

        for q in bm25_queries:
            try:
                ranked_lists.append(bm25.invoke(q)[:self.config.bm25_k])
            except Exception:
                pass

        for sid in section_hints:
            chunk_list = section_to_chunks.get(sid, [])
            if chunk_list:
                ranked_lists.append(chunk_list)

        fused = self.rrf_fuse(ranked_lists)
        return fused[:self.config.top_docs_after_fusion]

    def add_neighbors(
        self,
        docs: List[Document],
        index: Dict[Tuple[str, int], Document],
        window: int = 1,
    ) -> List[Document]:
        out = []
        seen = set()

        for d in docs:
            src = str(d.metadata["source"])
            cid = int(d.metadata["chunk_id"])

            for delta in range(-window, window + 1):
                key = (src, cid + delta)
                if key in index and key not in seen:
                    seen.add(key)
                    out.append(index[key])

        return out

    def build_context(self, docs: List[Document]) -> str:
        parts = []
        total = 0

        for i, d in enumerate(docs, 1):
            header = (
                f"[{i}] source={d.metadata.get('source')} "
                f"section={d.metadata.get('section_title')} "
                f"chunk_id={d.metadata.get('chunk_id')} "
                f"start={d.metadata.get('start_index')}\n"
            )
            block = header + d.page_content.strip()

            if total + len(block) > self.config.max_context_chars:
                break

            parts.append(block)
            total += len(block) + 2

        return "\n\n---\n\n".join(parts)

    # ----------------------------
    # ANSWERING
    # ----------------------------

    def generate_answer(
        self,
        context: str,
        user_query: str,
        answer_mode: str = "Подробно",
    ) -> str:
        mode_instructions = {
            "Кратко": (
                "Формат ответа: краткий.\n"
                "Дай только прямой ответ: максимум 4 предложения или максимум 5 коротких пунктов.\n"
                "Не добавляй вводных фраз, подробных объяснений и пересказа контекста.\n"
                "Ссылки [номер] всё равно обязательны после утверждений."
            ),
            "Подробно": (
                "Формат ответа: подробный.\n"
                "Дай развёрнутый ответ с сохранением структуры документа: условия, исключения, сроки, "
                "перечни и последовательность действий выноси отдельными пунктами.\n"
                "Если в контексте есть несколько релевантных фрагментов, сведи их в единый ответ.\n"
                "Не цитируй дословно большие куски без необходимости, но ссылки [номер] обязательны."
            ),
            "Строго с цитатами": (
                "Формат ответа: строго с цитатами.\n"
                "Не делай обобщений без дословной опоры на контекст.\n"
                "Каждый ключевой пункт оформи так: Цитата: \"короткая дословная выдержка\" [номер]. Вывод: ...\n"
                "Цитаты должны быть короткими, обычно 5-25 слов.\n"
                "Если для ответа нет точной цитаты, напиши: 'В контексте нет точной цитаты для ответа'.\n"
                "Не используй знания вне контекста и не смягчай отсутствие цитаты пересказом."
            ),
        }
        token_limits = {
            "Кратко": 650,
            "Подробно": 1800,
            "Строго с цитатами": 1400,
        }
        mode_text = mode_instructions.get(answer_mode, mode_instructions["Подробно"])
        system_text = (
            "Ты отвечаешь только по предоставленному контексту.\n"
            "Нельзя добавлять факты от себя.\n"
            "Если ответ состоит из перечня, выпиши все релевантные пункты полностью и по порядку.\n"
            "Если ответа нет, напиши: 'В контексте нет ответа'.\n"
            "После утверждений ставь ссылки на источник [номер].\n\n"
            f"Режим ответа: {answer_mode}.\n"
            f"{mode_text}"
        )

        user_text = (
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_query}"
        )

        return self.llm_client.completion(
            system_text,
            user_text,
            temperature=0.1,
            max_tokens=token_limits.get(answer_mode, 1800),
        )

    # ----------------------------
    # DOCUMENT COMPARE QA
    # ----------------------------

    def _build_compare_context(
        self,
        compare_result: Dict[str, Any],
        mode: str = "all",
        max_changed_sections: int = 12,
        max_chars_per_side: int = 2200,
    ) -> str:
        left_doc = compare_result["left_doc"]
        right_doc = compare_result["right_doc"]
        summary = compare_result["summary"]
        changed_sections = [s for s in compare_result["sections"] if s.get("changed")]

        if mode != "all":
            changed_sections = [
                s for s in changed_sections if str(s.get("change_type", "")) == mode
            ]

        parts = [
            "META:",
            f"left_document_name={left_doc.get('document_name', '')}",
            f"left_version={left_doc.get('version', '')}",
            f"right_document_name={right_doc.get('document_name', '')}",
            f"right_version={right_doc.get('version', '')}",
            f"left_sections={summary.get('left_sections', 0)}",
            f"right_sections={summary.get('right_sections', 0)}",
            f"changed_sections={summary.get('changed_sections', 0)}",
            f"compare_mode={mode}",
            "",
            "CHANGED_SECTIONS:",
        ]

        for index, section in enumerate(changed_sections[:max_changed_sections], start=1):
            title = str(section.get("section_title", f"Section {index}"))
            left_text = str(section.get("left_text", ""))[:max_chars_per_side].strip()
            right_text = str(section.get("right_text", ""))[:max_chars_per_side].strip()
            parts.extend(
                [
                    f"SECTION {index}: {title}",
                    "LEFT:",
                    left_text or "[absent]",
                    "RIGHT:",
                    right_text or "[absent]",
                    "",
                ]
            )

        if len(changed_sections) > max_changed_sections:
            parts.append(
                f"... trimmed {len(changed_sections) - max_changed_sections} more changed sections ..."
            )

        return "\n".join(parts).strip()

    def _fallback_compare_answer(self, compare_result: Dict[str, Any], mode: str = "all") -> str:
        changed_sections = [s for s in compare_result["sections"] if s.get("changed")]
        if mode != "all":
            changed_sections = [
                s for s in changed_sections if str(s.get("change_type", "")) == mode
            ]

        mode_label = {
            "all": "Измененных разделов",
            "added": "Добавленных разделов",
            "removed": "Удаленных разделов",
            "changed": "Измененных разделов",
        }.get(mode, "Разделов")
        lines = [
            f"{mode_label}: {len(changed_sections)}."
        ]

        for section in changed_sections[:8]:
            title = str(section.get("section_title", "Раздел"))
            change_type = str(section.get("change_type", "changed"))
            if change_type == "added":
                lines.append(f"- Добавлено: {title}")
            elif change_type == "removed":
                lines.append(f"- Удалено: {title}")
            else:
                lines.append(f"- Изменено: {title}")

        return "\n".join(lines)

    def detect_compare_mode(self, user_query: str) -> str:
        q = user_query.strip().lower()
        if not q:
            return "all"

        added_markers = [
            "только добавлен",
            "только новое",
            "что добав",
            "что новое",
            "РґРѕР±Р°РІР»РµРЅРѕ",
            "РґРѕР±Р°РІР»РµРЅРЅРѕРµ",
            "РЅРѕРІРѕРіРѕ",
            "\u0434\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u043e",
            "\u0434\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u043d\u043e\u0435",
            "\u043d\u043e\u0432\u043e\u0433\u043e",
            "only added",
            "just added",
        ]
        removed_markers = [
            "только удал",
            "что удал",
            "что убра",
            "СѓРґР°Р»РµРЅРѕ",
            "СѓРґР°Р»РµРЅРЅРѕРµ",
            "СѓР±СЂР°РЅРѕ",
            "\u0443\u0434\u0430\u043b\u0435\u043d\u043e",
            "\u0443\u0434\u0430\u043b\u0435\u043d\u043d\u043e\u0435",
            "\u0443\u0431\u0440\u0430\u043d\u043e",
            "only removed",
            "just removed",
        ]
        changed_markers = [
            "только измен",
            "что измен",
            "РёР·РјРµРЅРµРЅРѕ",
            "РёР·РјРµРЅРµРЅРЅРѕРµ",
            "\u0438\u0437\u043c\u0435\u043d\u0435\u043d\u043e",
            "\u0438\u0437\u043c\u0435\u043d\u0435\u043d\u043d\u043e\u0435",
            "only changed",
            "just changed",
        ]

        if any(marker in q for marker in added_markers):
            return "added"
        if any(marker in q for marker in removed_markers):
            return "removed"
        if any(marker in q for marker in changed_markers):
            return "changed"
        return "all"

    def generate_compare_answer(
        self,
        compare_result: Dict[str, Any],
        user_query: str = "",
    ) -> str:
        mode = self.detect_compare_mode(user_query)
        compare_context = self._build_compare_context(compare_result, mode=mode)
        question = user_query.strip() or "Сравни документы и кратко перечисли, что было добавлено, удалено и изменено."

        system_text = (
            "Ты сравниваешь две версии документа по уже подготовленному diff-контексту.\n"
            "Нужно ответить по-русски, кратко и структурно.\n"
            "Выдели три категории, если они есть: добавлено, удалено, изменено.\n"
            "Если compare_mode не all, отвечай только по этому типу изменений.\n"
            "Не придумывай факты, опирайся только на compare-контекст.\n"
            "Если формулировки неочевидны, пиши осторожно: 'вероятно', 'в этом разделе изменено'.\n"
            "Если пользователь не задал уточняющий вопрос, дай общий summary изменений."
        )
        user_text = f"ВОПРОС:\n{question}\n\nCOMPARE_CONTEXT:\n{compare_context}"

        try:
            return self.llm_client.completion(
                system_text,
                user_text,
                temperature=0.1,
                max_tokens=1400,
            ).strip()
        except Exception:
            return self._fallback_compare_answer(compare_result, mode=mode)

    def compare_documents_qa(
        self,
        left_doc_id: str,
        right_doc_id: str,
        user_query: str = "",
    ) -> Dict[str, Any]:
        left_bundle = self.compare_service.load_document_bundle(left_doc_id)
        right_bundle = self.compare_service.load_document_bundle(right_doc_id)
        compare_result = self.compare_service.compare_documents(left_bundle, right_bundle)
        answer = self.generate_compare_answer(compare_result, user_query)

        return {
            "answer": answer,
            "compare_result": compare_result,
            "left_doc": compare_result["left_doc"],
            "right_doc": compare_result["right_doc"],
            "summary": compare_result["summary"],
        }
