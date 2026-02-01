# compare_versions.py
import os
import re
import json
import time
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

import requests
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
load_dotenv()

YANDEX_CLOUD_FOLDER = "b1gr2b050rg9ph80aob2"
YANDEX_MODEL_URI = f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt/latest"
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")


def log(msg: str):
    print(msg, flush=True)


def yandex_completion(system_text: str, user_text: str, temperature=0.0, max_tokens=2000) -> str:
    if not YANDEX_API_KEY:
        raise RuntimeError("Не задан YANDEX_API_KEY в переменной окружения")

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {YANDEX_API_KEY}"}
    payload = {
        "modelUri": YANDEX_MODEL_URI,
        "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": max_tokens},
        "messages": [{"role": "system", "text": system_text}, {"role": "user", "text": user_text}],
    }

    t0 = time.perf_counter()
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    dt = time.perf_counter() - t0

    if r.status_code != 200:
        raise RuntimeError(f"Yandex API error {r.status_code}: {r.text}")

    log(f"    YandexGPT responded in {dt:.2f}s")
    return r.json()["result"]["alternatives"][0]["message"]["text"]


# ====================
# --- Chroma config ---
# ====================
DB_PATH = "./my_chroma_db"
COLLECTION_NAME = "docs"

EMBEDDING_MODEL = "ai-forever/sbert_large_nlu_ru"
DEVICE = "cpu"

# thresholds
MIN_SEM_SIM = 0.75
SKIP_IF_TEXT_SIM = 0.985
MAX_CHUNKS_TO_CHECK = 500
MAX_LLM_CALLS = 120

# progress
PROGRESS_EVERY = 5          # как часто печатать прогресс
PRINT_SKIPS = False         # если True — будет много логов почему пропустили


def norm_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def md_escape(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("|", r"\|")
    s = s.replace("\n", "<br>")
    return s


def get_all_docs_for_version(db: Chroma, version: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Достаём ВСЕ документы (чанки) из Chroma по where filter.
    Возвращаем список {id, text, meta}.
    """
    col = db._collection
    out: List[Dict[str, Any]] = []
    offset = 0
    page = 500

    where = {"version": version}
    if source:
        where["source"] = source

    log(f"Loading chunks for version={version}" + (f", source={source}" if source else "") + " ...")
    while True:
        t0 = time.perf_counter()
        batch = col.get(
            where=where,
            include=["documents", "metadatas"],
            limit=page,
            offset=offset,
        )
        dt = time.perf_counter() - t0

        ids = batch.get("ids", [])
        docs = batch.get("documents", [])
        metas = batch.get("metadatas", [])

        if not ids:
            break

        for i in range(len(ids)):
            out.append({"id": ids[i], "text": docs[i] or "", "meta": metas[i] or {}})

        offset += page
        log(f"  loaded {len(out)} chunks (last page {len(ids)}), get() took {dt:.2f}s")

    out.sort(key=lambda x: int(x["meta"].get("chunk_id", 10**9)))
    log(f"Done loading version={version}: {len(out)} chunks\n")
    return out


def similarity_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def extract_changes_llm(
    version_a: str,
    version_b: str,
    text_a: str,
    text_b: str,
    ref_a: str,
    ref_b: str,
) -> List[Dict[str, str]]:
    system_text = (
        "Ты извлекаешь ИЗМЕНЕНИЯ между двумя версиями документа.\n"
        "Правила:\n"
        "1) Нельзя ничего придумывать — только то, что явно есть в A и B.\n"
        "2) Изменения должны быть атомарными: одно изменение = один объект.\n"
        "3) Значения old/new бери как ТОЧНЫЕ фрагменты из текста (даты, числа, формулировки).\n"
        "Верни ТОЛЬКО JSON-массив объектов формата:\n"
        "[{\"change\":\"...\",\"old\":\"...\",\"new\":\"...\",\"ref_old\":\"...\",\"ref_new\":\"...\"}]\n"
        "Если отличий нет — верни []"
    )

    user_text = (
        f"ВЕРСИЯ A = {version_a} (ref_old={ref_a})\n{text_a}\n\n"
        f"ВЕРСИЯ B = {version_b} (ref_new={ref_b})\n{text_b}\n"
    )

    raw = yandex_completion(system_text, user_text, temperature=0.0, max_tokens=1400).strip()

    # попытка вытащить JSON даже если модель добавила мусор
    m = re.search(r"(\[\s*\{.*\}\s*\])", raw, flags=re.DOTALL)
    json_text = m.group(1) if m else raw

    try:
        data = json.loads(json_text)
        if not isinstance(data, list):
            return []
        clean = []
        for x in data:
            if not isinstance(x, dict):
                continue
            if not x.get("old") and not x.get("new"):
                continue
            clean.append(
                {
                    "change": str(x.get("change", "")).strip(),
                    "old": str(x.get("old", "")).strip(),
                    "new": str(x.get("new", "")).strip(),
                    "ref_old": str(x.get("ref_old", ref_a)).strip(),
                    "ref_new": str(x.get("ref_new", ref_b)).strip(),
                }
            )
        return clean
    except Exception as e:
        log(f"    JSON parse failed: {e}\nRAW (first 800 chars):\n{raw[:800]}\n")
        return []


def main(
    version_a: str,
    version_b: str,
    source_a: Optional[str] = None,
    source_b: Optional[str] = None,
    out_path: str = "changes_2cols.md",
):
    t_start = time.perf_counter()
    log("=== Compare versions ===")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(DB_PATH)

    log("Init embeddings model (first run can be slow)...")
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    log("Open Chroma DB...")
    db = Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
    )

    a_chunks = get_all_docs_for_version(db, version_a, source=source_a)
    b_chunks = get_all_docs_for_version(db, version_b, source=source_b)

    if not a_chunks or not b_chunks:
        raise RuntimeError(
            f"Не найдены чанки одной из версий. A={len(a_chunks)}, B={len(b_chunks)}. "
            f"Проверь metadata.version / source в БД."
        )

    log(f"Will check up to {min(len(a_chunks), MAX_CHUNKS_TO_CHECK)} chunks from version A\n")

    changes_all: List[Dict[str, str]] = []
    llm_calls = 0

    # stats
    skipped_empty = 0
    skipped_nohits = 0
    skipped_sem = 0
    skipped_textsim = 0
    candidates_for_llm = 0

    for i, a in enumerate(a_chunks[:MAX_CHUNKS_TO_CHECK], start=1):
        if i % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - t_start
            log(
                f"[{i}/{min(len(a_chunks), MAX_CHUNKS_TO_CHECK)}] "
                f"llm_calls={llm_calls}, raw_changes={len(changes_all)}, elapsed={elapsed:.1f}s"
            )

        a_text = norm_text(a["text"])
        if not a_text:
            skipped_empty += 1
            continue

        # Ищем матч в версии B через векторный поиск.
        # В разных версиях связки Chroma/LangChain параметр может называться filter или where.
        try:
            hits = db.similarity_search_with_score(
                a_text,
                k=1,
                filter={"version": version_b, **({"source": source_b} if source_b else {})},
            )
        except TypeError:
            hits = db.similarity_search_with_score(
                a_text,
                k=1,
                where={"version": version_b, **({"source": source_b} if source_b else {})},
            )

        if not hits:
            skipped_nohits += 1
            if PRINT_SKIPS:
                log("  skip: no hits")
            continue

        b_doc, dist = hits[0]
        sem_sim = 1.0 - float(dist)

        if sem_sim < MIN_SEM_SIM:
            skipped_sem += 1
            if PRINT_SKIPS:
                log(f"  skip: sem_sim={sem_sim:.3f} < {MIN_SEM_SIM}")
            continue

        b_text = norm_text(b_doc.page_content)

        if similarity_ratio(a_text, b_text) >= SKIP_IF_TEXT_SIM:
            skipped_textsim += 1
            if PRINT_SKIPS:
                log("  skip: almost identical text")
            continue

        candidates_for_llm += 1

        if llm_calls >= MAX_LLM_CALLS:
            log("Reached MAX_LLM_CALLS, stopping.")
            break

        ref_a = f"{a['meta'].get('source')}#v{version_a}:chunk={a['meta'].get('chunk_id')}"
        ref_b = f"{b_doc.metadata.get('source')}#v{version_b}:chunk={b_doc.metadata.get('chunk_id')}"

        llm_calls += 1
        log(
            f"  -> LLM call {llm_calls}: sem_sim={sem_sim:.3f} | "
            f"A_chunk={a['meta'].get('chunk_id')} vs B_chunk={b_doc.metadata.get('chunk_id')}"
        )

        extracted = extract_changes_llm(version_a, version_b, a_text, b_text, ref_a, ref_b)
        if extracted:
            log(f"    extracted {len(extracted)} change(s)")
        changes_all.extend(extracted)

    # дедупликация (с сохранением порядка первых вхождений)
    uniq: Dict[Any, Dict[str, str]] = {}
    for c in changes_all:
        key = (c.get("change", "").lower(), c.get("old", ""), c.get("new", ""))
        if key not in uniq:
            uniq[key] = c
    changes = list(uniq.values())

    log("\n=== STATS ===")
    log(f"checked_chunks={min(len(a_chunks), MAX_CHUNKS_TO_CHECK)}")
    log(f"candidates_for_llm={candidates_for_llm}")
    log(f"llm_calls={llm_calls}")
    log(
        f"skipped_empty={skipped_empty}, skipped_nohits={skipped_nohits}, "
        f"skipped_sem={skipped_sem}, skipped_textsim={skipped_textsim}"
    )
    log(f"total_changes_raw={len(changes_all)}, unique_changes={len(changes)}")
    log(f"total_time={time.perf_counter() - t_start:.1f}s\n")

    # ============
    # 2 столбца: Документ 1 / Документ 2
    # ============
    doc1_title = f"Документ 1 (версия {version_a})"
    doc2_title = f"Документ 2 (версия {version_b})"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Найдено изменений: {len(changes)}\n\n")
        f.write(f"| {md_escape(doc1_title)} | {md_escape(doc2_title)} |\n")
        f.write("|---|---|\n")
        for c in changes:
            label = md_escape(c.get("change", "изменение"))
            old = md_escape(c.get("old", ""))
            new = md_escape(c.get("new", ""))

            left_cell = f"**{label}**: {old}" if old else f"**{label}**: (нет)"
            right_cell = f"**{label}**: {new}" if new else f"**{label}**: (нет)"

            f.write(f"| {left_cell} | {right_cell} |\n")

    print(f"Найдено изменений: {len(changes)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # сравнение version=1 vs version=2
    # Если хочешь ещё жёстче зафиксировать источники, раскомментируй source_a/source_b:
    main(
        "1",
        "2",
        # source_a="K:/document.md",
        # source_b="K:/document1.md",
        out_path="changes_2cols.md",
    )