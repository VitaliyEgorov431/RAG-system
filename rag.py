# rag.py
import os
import re
import requests
from typing import List, Dict, Tuple, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from dotenv import load_dotenv
load_dotenv()


YANDEX_CLOUD_FOLDER = "b1gr2b050rg9ph80aob2"
YANDEX_MODEL_URI = f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt/latest"
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")


DB_PATH = "./my_chroma_db"
COLLECTION_NAME = "docs"
SOURCE_FILE = "parsed_result.md"

EMBEDDING_MODEL = "ai-forever/sbert_large_nlu_ru"
DEVICE = "cpu"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200

# retrieval
CHROMA_K = 8
CHROMA_FETCH_K = 40
BM25_K = 6

ENABLE_QUERY_CORRECTION = True
ENABLE_QUERY_EXPANSION = True
MAX_QUERY_VARIANTS = 3

# context
TOP_DOCS_AFTER_FUSION = 8
NEIGHBOR_WINDOW = 1        # добавим соседние чанки +/-1
MAX_CONTEXT_CHARS = 16000

DEBUG = True
# -----------------------------------------


def preprocess_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace(" – ", "\n- ")
    text = text.replace("– ", "\n- ")
    return text


def normalize_query(q: str) -> str:
    q = q.replace("\r\n", "\n").strip()
    # Для "обычных" вопросов склеим лишние переносы в пробелы
    # (для цитат это не делаем, см. ниже)
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


def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", ". ", " "],
    )


def yandex_completion(system_text: str, user_text: str, temperature=0.1, max_tokens=1200) -> str:
    if not YANDEX_API_KEY:
        raise RuntimeError("Не задан YANDEX_CLOUD_API_KEY в переменных окружения")

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {YANDEX_API_KEY}"}
    payload = {
        "modelUri": YANDEX_MODEL_URI,
        "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": max_tokens},
        "messages": [
            {"role": "system", "text": system_text},
            {"role": "user", "text": user_text},
        ],
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Yandex API error {r.status_code}: {r.text}")
    return r.json()["result"]["alternatives"][0]["message"]["text"]


def looks_like_quote(user_query: str) -> bool:
    # эвристика: длинная вставка/пункт/цитата
    if len(user_query) >= 220:
        return True
    if user_query.count("\n") >= 2:
        return True
    if re.match(r"^\s*\d+\.\s", user_query):
        return True
    if user_query.strip().endswith(":"):
        return True
    return False


def try_extract_point_by_quote(raw_text: str, user_query: str) -> Optional[str]:
    """
    Если пользователь вставил кусок текста (цитату), пытаемся найти её в raw_text
    и вытащить пункт целиком до следующего "NN.".
    """
    q = user_query.strip()
    if not q:
        return None

    # Берем “якорь” - первые ~140 символов без лишних пробелов
    anchor = re.sub(r"\s+", " ", q)[:140].strip()
    if len(anchor) < 40:
        return None

    # regex: пробелы в якоре заменяем на \s+ чтобы матчилось через переносы строк
    pattern = re.escape(anchor)
    pattern = re.sub(r"\\ ", r"\\s+", pattern)

    m = re.search(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    start = m.start()

    # Если в начале запроса есть номер пункта "27.", ищем до "28."
    mnum = re.match(r"^\s*(\d+)\.\s", q)
    if mnum:
        n = int(mnum.group(1))
        next_pat = rf"\n\s*{n+1}\.\s"
        mnext = re.search(next_pat, raw_text[start + 1 :], flags=re.DOTALL)
        end = start + 1 + mnext.start() if mnext else min(len(raw_text), start + 4500)
        return raw_text[start:end].strip()

    # иначе — до следующего "NN."
    mnext = re.search(r"\n\s*\d+\.\s", raw_text[start + 1 :], flags=re.DOTALL)
    end = start + 1 + mnext.start() if mnext else min(len(raw_text), start + 4500)
    return raw_text[start:end].strip()


def correct_query_typos(original: str) -> str:
    """
    Исправляет опечатки/орфографию, не меняя смысл.
    Возвращает одну строку. Если что-то пошло не так — исходный текст.
    """
    if not ENABLE_QUERY_CORRECTION:
        return original

    system_text = (
        "Ты исправляешь опечатки и орфографические ошибки в запросе пользователя.\n"
        "Правила:\n"
        "1) НЕ перефразируй и НЕ меняй смысл.\n"
        "2) Сохраняй числа, даты, аббревиатуры, имена собственные, ссылки на пункты (например '27.').\n"
        "3) Сохраняй специальные термины как есть.\n"
        "Верни ТОЛЬКО исправленный текст одной строкой, без комментариев."
    )

    try:
        fixed = yandex_completion(system_text, original, temperature=0.0, max_tokens=200).strip()
        if not fixed or len(fixed) < 3:
            return original
        fixed = re.sub(r"\s+", " ", fixed).strip()
        return fixed
    except:
        return original


def expand_query(original: str) -> List[str]:
    system_text = (
        "Ты помощник для поиска по документам. "
        "Переформулируй запрос пользователя 2 разными способами, НЕ меняя смысл и фокус. "
        "Не смещай вопрос в другую тему (например, не превращай 'что указано' в 'какой срок'). "
        "Верни только варианты, разделяй переводом строки или запятыми. Без пояснений."
    )
    try:
        text = yandex_completion(system_text, original, temperature=0.2, max_tokens=200)
        parts = re.split(r"[\n,;]+", text)
        variants = [original] + [p.strip(" \t-•") for p in parts if p.strip()]
        variants = uniq_keep_order(variants)
        return variants[:MAX_QUERY_VARIANTS]
    except:
        return [original]


def doc_key(doc: Document) -> Tuple[str, int]:
    return (str(doc.metadata.get("source", "unknown")), int(doc.metadata.get("chunk_id", -1)))


def rrf_fuse(lists: List[List[Document]], k: int = 60) -> List[Document]:
    scores: Dict[Tuple[str, int], float] = {}
    by_key: Dict[Tuple[str, int], Document] = {}

    for docs in lists:
        for rank, d in enumerate(docs, start=1):
            key = doc_key(d)
            by_key[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_key[key] for key, _ in fused]


def add_neighbors(docs: List[Document], id_to_doc: Dict[Tuple[str, int], Document], window: int) -> List[Document]:
    out = []
    seen = set()
    for d in docs:
        src = str(d.metadata.get("source", "unknown"))
        cid = int(d.metadata.get("chunk_id", -1))
        for delta in range(-window, window + 1):
            key = (src, cid + delta)
            if key in id_to_doc and key not in seen:
                seen.add(key)
                out.append(id_to_doc[key])
    return out


def build_context(docs: List[Document]) -> str:
    parts = []
    total = 0
    for i, d in enumerate(docs, start=1):
        header = f"[{i}] source={d.metadata.get('source')} chunk_id={d.metadata.get('chunk_id')} start={d.metadata.get('start_index')}\n"
        block = header + d.page_content.strip()
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n---\n\n".join(parts)


def main():
    print("Init...")

    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(f"Не найден {SOURCE_FILE}")

    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        raw_text = preprocess_text(f.read())

    splitter = make_splitter()
    bm25_docs = splitter.split_documents([Document(page_content=raw_text, metadata={"source": SOURCE_FILE})])
    for i, d in enumerate(bm25_docs):
        d.metadata["chunk_id"] = i

    id_to_doc = {(d.metadata["source"], d.metadata["chunk_id"]): d for d in bm25_docs}

    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = BM25_K

    embedding_func = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_func,
    )

    chroma = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": CHROMA_K, "fetch_k": CHROMA_FETCH_K},
    )

    print("Ready. Type 'exit'.")

    while True:
        user_query = input("\nВопрос: ").strip()
        if user_query.lower() in ("exit", "выход"):
            break
        if not user_query:
            continue

        is_quote = looks_like_quote(user_query)

        # Для обычных вопросов нормализуем пробелы/переносы (цитату не трогаем)
        if not is_quote:
            user_query = normalize_query(user_query)

        # ----------------------------
        # 1) Попытка прямого извлечения для цитаты
        # ----------------------------
        direct = None
        corrected_quote = None

        if is_quote:
            direct = try_extract_point_by_quote(raw_text, user_query)
            if not direct and ENABLE_QUERY_CORRECTION:
                # если цитата с опечатками/шумом — попробуем “исправленную цитату”
                corrected_quote = correct_query_typos(user_query)
                if corrected_quote and corrected_quote != user_query:
                    direct = try_extract_point_by_quote(raw_text, corrected_quote)

            if direct and DEBUG:
                print("Direct extract by quote: SUCCESS")

        if direct:
            context = f"[1] source={SOURCE_FILE} (direct extract)\n{direct}"
        else:
            # ----------------------------
            # 2) Стандартный RAG
            # ----------------------------
            corrected_query = user_query
            if (not is_quote) and ENABLE_QUERY_CORRECTION:
                corrected_query = correct_query_typos(user_query)
                if DEBUG and corrected_query != user_query:
                    print(f"Corrected query: {corrected_query}")

            if is_quote:
                # для цитат — без expansion, но ищем и по исходному, и по исправленному (если есть)
                base = [user_query]
                if corrected_quote and corrected_quote != user_query:
                    base.append(corrected_quote)
                queries = uniq_keep_order(base)[:MAX_QUERY_VARIANTS]
            else:
                variants = expand_query(corrected_query) if ENABLE_QUERY_EXPANSION else [corrected_query]
                # добавим оригинал как fallback
                queries = uniq_keep_order([user_query, corrected_query] + variants)[:MAX_QUERY_VARIANTS]

            if DEBUG:
                print(f"Query variants: {queries}")

            ranked_lists = []
            for q in queries:
                ranked_lists.append(chroma.invoke(q))
                ranked_lists.append(bm25.invoke(q))

            fused = rrf_fuse(ranked_lists)
            fused = fused[:TOP_DOCS_AFTER_FUSION]

            expanded = add_neighbors(fused, id_to_doc, window=NEIGHBOR_WINDOW)

            if DEBUG:
                print(f"Fused top: {len(fused)}, after neighbors: {len(expanded)}")

            context = build_context(expanded)

        if DEBUG:
            print("\n=== CONTEXT (first 2500 chars) ===")
            print(context[:2500])
            print("=== END CONTEXT ===\n")

        system_text = (
            "Ты извлекаешь ответ ТОЛЬКО из предоставленного контекста.\n"
            "Нельзя добавлять факты от себя.\n"
            "Если в контексте есть перечисление — выпиши ВСЕ пункты полностью и в исходном порядке, "
            "без сокращений и без перефразирования.\n"
            "Если ответа нет — напиши: 'В контексте нет ответа'.\n"
            "После каждого пункта поставь ссылку на источник в формате [номер], например: '... [2]'."
        )

        user_text = (
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ЗАДАНИЕ:\n"
            f"1) Определи, что именно спрашивает пользователь.\n"
            f"2) Если это пункт с двоеточием — выпиши весь перечень после двоеточия.\n\n"
            f"ВВОД ПОЛЬЗОВАТЕЛЯ:\n{user_query}"
        )

        try:
            answer = yandex_completion(system_text, user_text, temperature=0.1, max_tokens=2000)
            print("\n>>> ОТВЕТ:\n" + answer)
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()