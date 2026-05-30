import os
import re
import uuid
from html import escape
from pathlib import Path
from typing import Dict, List

import chromadb
import streamlit as st

from bundle_store import backfill_bundles_from_dir, delete_bundle, list_bundles, load_bundle
from db_2 import IngestConfig, IngestService
from rag_service import RAGConfig, RAGService


APP_TITLE = "RAG Document Chat"
UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = "./data/processed"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
DEFAULT_FOLDER = "Без папки"
MAX_UPLOAD_MB = 100


def safe_document_name(name: str) -> str:
    value = re.sub(r"[^\w_. -]+", "_", name, flags=re.UNICODE).strip(" ._")
    return value or "document"


def document_folder(doc: Dict[str, str]) -> str:
    return doc.get("document_folder") or DEFAULT_FOLDER


def document_label(doc: Dict[str, str]) -> str:
    name = doc.get("document_name") or "document"
    version = doc.get("version") or "version"
    source_type = doc.get("source_type") or Path(doc.get("source", "")).suffix.lstrip(".")
    return f"{document_folder(doc)} / {name} | {version} | {source_type or 'file'}"


def document_matches(doc: Dict[str, str], query: str, folder: str) -> bool:
    if folder != "Все папки" and document_folder(doc) != folder:
        return False

    if not query:
        return True

    haystack = " ".join(
        [
            document_folder(doc),
            doc.get("document_name", ""),
            doc.get("version", ""),
            doc.get("source_type", ""),
            doc.get("source", ""),
            doc.get("doc_id", ""),
        ]
    ).lower()
    return query.lower() in haystack


def normalized_filename(name: str) -> str:
    return Path(str(name)).name.strip().lower()


def find_duplicate_uploads(uploaded_filename: str, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    target = normalized_filename(uploaded_filename)
    if not target:
        return []

    matches = []
    for doc in documents:
        candidates = [
            doc.get("original_filename", ""),
            doc.get("source", ""),
            doc.get("stored_source_path", ""),
        ]
        if any(normalized_filename(candidate) == target for candidate in candidates if candidate):
            matches.append(doc)

    return matches


def list_documents() -> List[Dict[str, str]]:
    documents = list_bundles()
    if documents:
        return documents

    backfill_bundles_from_dir(PROCESSED_DIR)
    return list_bundles()


@st.cache_resource(show_spinner="Загружаю RAG-сервис...")
def get_rag_service() -> RAGService:
    return RAGService(RAGConfig(debug=False))


@st.cache_resource(show_spinner="Готовлю сервис загрузки...")
def get_ingest_service() -> IngestService:
    return IngestService(IngestConfig())


@st.cache_resource(show_spinner="Собираю индекс документа...")
def get_document_index(doc_id: str):
    service = get_rag_service()
    return service.build_document_index_from_db(doc_id)


def persist_uploaded_file(uploaded_file) -> str:
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Неподдерживаемый формат файла. Разрешены: {allowed}.")

    size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(f"Файл слишком большой: {size_mb:.1f} МБ. Лимит: {MAX_UPLOAD_MB} МБ.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = safe_document_name(Path(uploaded_file.name).stem)
    path = UPLOAD_DIR / f"{safe_name}_{uuid.uuid4().hex[:8]}{ext}"

    with open(path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return str(path)


def is_inside_workspace_data(path_value: str) -> bool:
    if not path_value:
        return False

    try:
        root = Path.cwd().resolve()
        path = Path(path_value).resolve()
        data_root = (root / "data").resolve()
        return path == data_root or data_root in path.parents
    except Exception:
        return False


def remove_file_if_safe(path_value: str) -> None:
    if not is_inside_workspace_data(path_value):
        return

    path = Path(path_value)
    if path.exists() and path.is_file():
        path.unlink()


def delete_document(doc_id: str) -> None:
    bundle = load_bundle(doc_id)

    client = chromadb.PersistentClient(path="./my_chroma_db")
    delete_errors = []
    for collection_name in ("docs", "section_summaries"):
        try:
            collection = client.get_collection(collection_name)
            collection.delete(where={"doc_id": doc_id})
        except Exception as exc:
            message = str(exc)
            if "does not exist" not in message.lower() and "not found" not in message.lower():
                delete_errors.append(f"{collection_name}: {message}")

    if delete_errors:
        raise RuntimeError("; ".join(delete_errors))

    delete_bundle(doc_id)

    processed_path = Path(PROCESSED_DIR) / f"{doc_id}.json"
    if processed_path.exists():
        processed_path.unlink()

    if bundle:
        for key in ("stored_source_path", "text_source_path", "layout_source_path", "source"):
            remove_file_if_safe(str(bundle.get(key, "")))


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 980px;
            padding-top: 1.4rem;
            padding-bottom: 5rem;
        }
        [data-testid="stSidebar"] { min-width: 350px; }
        [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
        .app-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin: 0 0 .25rem 0;
        }
        .app-subtitle {
            color: #6b7280;
            font-size: .92rem;
            margin-bottom: 1rem;
        }
        .doc-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: .65rem .75rem;
            margin-bottom: .55rem;
            background: #ffffff;
        }
        .doc-title {
            font-weight: 650;
            margin-bottom: .15rem;
        }
        .doc-meta {
            color: #6b7280;
            font-size: .82rem;
            line-height: 1.35;
        }
        .mode-caption {
            color: #6b7280;
            margin-top: -.35rem;
            margin-bottom: .8rem;
        }
        .empty-chat {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            color: #4b5563;
            background: #fafafa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_upload_panel() -> None:
    with st.sidebar.expander("Загрузить документ", expanded=False):
        documents = list_documents()
        uploaded_file = st.file_uploader(
            "Файл",
            type=["pdf", "docx", "md", "txt"],
            key="sidebar_upload",
        )

        suggested_name = ""
        if uploaded_file is not None:
            suggested_name = safe_document_name(Path(uploaded_file.name).stem)

        folder = st.text_input("Папка", value=DEFAULT_FOLDER, key="upload_folder")
        document_name = st.text_input("Название", value=suggested_name, key="upload_name")
        version = st.text_input("Версия", value="v1", key="upload_version")

        duplicate_docs = []
        if uploaded_file is not None:
            duplicate_docs = find_duplicate_uploads(uploaded_file.name, documents)

        allow_duplicate = True
        if duplicate_docs:
            st.warning(
                "Файл с таким именем уже загружался: "
                + ", ".join(document_label(doc) for doc in duplicate_docs[:3])
            )
            allow_duplicate = st.checkbox(
                "Всё равно загрузить повторно",
                value=False,
                key="allow_duplicate_upload",
            )

        disabled = uploaded_file is None or (bool(duplicate_docs) and not allow_duplicate)
        if st.button("Индексировать", type="primary", disabled=disabled, use_container_width=True):
            if not document_name.strip() or not version.strip():
                st.error("Укажите название и версию.")
                return

            try:
                source_path = persist_uploaded_file(uploaded_file)
                progress_bar = st.progress(0, text="Старт")

                with st.status("Индексирую документ...", expanded=True) as status:
                    st.write("Файл сохранён.")

                    def on_progress(value: float, message: str) -> None:
                        progress_bar.progress(
                            min(max(float(value), 0.0), 1.0),
                            text=message,
                        )
                        st.write(message)

                    service = get_ingest_service()
                    result = service.add_file_to_db(
                        source_path=source_path,
                        document_folder=folder.strip() or DEFAULT_FOLDER,
                        document_name=safe_document_name(document_name),
                        version=version.strip(),
                        progress_callback=on_progress,
                    )
                    status.update(label="Документ готов", state="complete")

                get_document_index.clear()
                st.success(
                    f"{result['document_name']} | {result['version']} "
                    f"({result['chunks']} чанков)"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось загрузить документ: {exc}")


@st.dialog("Загрузить документы", width="large")
def upload_documents_dialog() -> None:
    documents = list_documents()
    uploaded_file = st.file_uploader(
        "Файл",
        type=["pdf", "docx", "md", "txt"],
        key="dialog_upload",
    )

    suggested_name = ""
    if uploaded_file is not None:
        suggested_name = safe_document_name(Path(uploaded_file.name).stem)

    col_folder, col_version = st.columns([1, 0.45])
    with col_folder:
        folder = st.text_input("Папка", value=DEFAULT_FOLDER, key="dialog_upload_folder")
    with col_version:
        version = st.text_input("Версия", value="v1", key="dialog_upload_version")

    document_name = st.text_input("Название", value=suggested_name, key="dialog_upload_name")

    duplicate_docs = []
    if uploaded_file is not None:
        duplicate_docs = find_duplicate_uploads(uploaded_file.name, documents)

    allow_duplicate = True
    if duplicate_docs:
        st.warning(
            "Файл с таким именем уже загружался: "
            + ", ".join(document_label(doc) for doc in duplicate_docs[:3])
        )
        allow_duplicate = st.checkbox(
            "Всё равно загрузить повторно",
            value=False,
            key="dialog_allow_duplicate_upload",
        )

    disabled = uploaded_file is None or (bool(duplicate_docs) and not allow_duplicate)
    if st.button("Индексировать", type="primary", disabled=disabled, use_container_width=True):
        if not document_name.strip() or not version.strip():
            st.error("Укажите название и версию.")
            return

        try:
            source_path = persist_uploaded_file(uploaded_file)
            progress_bar = st.progress(0, text="Старт")

            with st.status("Индексирую документ...", expanded=True) as status:
                st.write("Файл сохранён.")

                def on_progress(value: float, message: str) -> None:
                    progress_bar.progress(
                        min(max(float(value), 0.0), 1.0),
                        text=message,
                    )
                    st.write(message)

                service = get_ingest_service()
                result = service.add_file_to_db(
                    source_path=source_path,
                    document_folder=folder.strip() or DEFAULT_FOLDER,
                    document_name=safe_document_name(document_name),
                    version=version.strip(),
                    progress_callback=on_progress,
                )
                status.update(label="Документ готов", state="complete")

            get_document_index.clear()
            st.success(
                f"{result['document_name']} | {result['version']} "
                f"({result['chunks']} чанков)"
            )
        except Exception as exc:
            st.error(f"Не удалось загрузить документ: {exc}")


def render_document_list(documents: List[Dict[str, str]]) -> None:
    with st.sidebar.expander(f"Загруженные документы ({len(documents)})", expanded=True):
        if not documents:
            st.caption("Документов пока нет.")
            return

        grouped: Dict[str, List[Dict[str, str]]] = {}
        for doc in documents:
            grouped.setdefault(document_folder(doc), []).append(doc)

        for folder in sorted(grouped):
            st.caption(folder)
            for doc in grouped[folder]:
                title = escape(doc.get("document_name") or "document")
                version = escape(doc.get("version") or "-")
                source_type = escape(doc.get("source_type") or "-")
                doc_id = escape(doc.get("doc_id") or "-")
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <div class="doc-title">{title}</div>
                        <div class="doc-meta">
                            version={version} · type={source_type}<br>
                            id={doc_id}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


@st.dialog("Загруженные документы", width="large")
def documents_dialog(documents: List[Dict[str, str]]) -> None:
    query = st.text_input("Поиск", placeholder="Название, версия, папка, id...", key="docs_dialog_search")
    filtered = [
        doc for doc in documents if document_matches(doc, query.strip(), "Все папки")
    ]

    if not filtered:
        st.info("Документы не найдены.")
        return

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for doc in filtered:
        grouped.setdefault(document_folder(doc), []).append(doc)

    for folder in sorted(grouped):
        st.subheader(folder)
        for doc in grouped[folder]:
            left, right = st.columns([1, 0.22])
            with left:
                st.markdown(f"**{doc.get('document_name') or 'document'}**")
                st.caption(
                    f"Версия: {doc.get('version') or '-'} · "
                    f"Тип: {doc.get('source_type') or '-'} · "
                    f"id: {doc.get('doc_id') or '-'}"
                )
            with right:
                if st.button("Удалить", key=f"delete_doc_row:{doc['doc_id']}", use_container_width=True):
                    try:
                        delete_document(doc["doc_id"])
                        get_document_index.clear()
                        st.cache_resource.clear()
                        st.success("Документ удалён.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Не удалось удалить документ: {exc}")
            st.divider()


def render_delete_panel(documents: List[Dict[str, str]]) -> None:
    with st.sidebar.expander("Удалить документ", expanded=False):
        if not documents:
            st.caption("Нет документов для удаления.")
            return

        labels = [document_label(doc) for doc in documents]
        delete_index = st.selectbox(
            "Документ",
            range(len(documents)),
            format_func=lambda i: labels[i],
            key="delete_doc",
        )
        selected = documents[delete_index]
        st.caption(f"doc_id: {selected['doc_id']}")
        confirm = st.checkbox("Подтверждаю удаление", key="confirm_delete_doc")

        if st.button("Удалить", disabled=not confirm, use_container_width=True):
            try:
                delete_document(selected["doc_id"])
                get_document_index.clear()
                st.cache_resource.clear()
                st.success("Документ удалён.")
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось удалить документ: {exc}")


def render_environment_panel() -> None:
    with st.sidebar.expander("Окружение", expanded=False):
        st.write("Yandex folder:", "задан" if os.getenv("YANDEX_CLOUD_FOLDER") else "не задан")
        st.write("Yandex API key:", "задан" if os.getenv("YANDEX_API_KEY") else "не задан")
        st.write("HF local only:", os.getenv("HF_LOCAL_FILES_ONLY", "1"))


def compact_source_title(source: Dict[str, object]) -> str:
    title = str(source.get("section_title") or "").strip()
    if not title:
        return "Фрагмент документа"
    return title if len(title) <= 120 else title[:117].rstrip() + "..."


def compact_source_text(text: str, max_chars: int = 1400) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def top_sources(sources: List[Dict[str, object]], limit: int = 5) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    seen = set()

    for source in sources:
        key = (
            str(source.get("source", "")),
            str(source.get("section_title", "")),
            str(source.get("chunk_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(source)
        if len(selected) >= limit:
            break

    return selected


def render_sources(sources: List[Dict[str, object]], show_fragments: bool) -> None:
    if not sources:
        return

    visible_sources = top_sources(sources)
    label = f"Источники: топ-{len(visible_sources)} из {len(sources)}"

    with st.expander(label):
        for source in visible_sources:
            index = source.get("index", "?")
            section_title = compact_source_title(source)
            file_name = Path(str(source.get("source", ""))).name or str(source.get("source", ""))
            text = compact_source_text(str(source.get("text", "")).strip())

            st.markdown(f"**[{index}] {file_name}**")
            if section_title and section_title != "Фрагмент документа":
                st.caption(section_title)

            if show_fragments:
                st.markdown(f"> {text}")
            else:
                st.caption(text[:260] + ("..." if len(text) > 260 else ""))


def render_sidebar(documents: List[Dict[str, str]]) -> Dict[str, object]:
    st.sidebar.markdown('<div class="app-title">RAG Chat</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="app-subtitle">Документы, вопросы и сравнение версий.</div>',
        unsafe_allow_html=True,
    )

    mode = st.sidebar.radio(
        "Режим",
        ["Чат", "Сравнение"],
        label_visibility="collapsed",
        horizontal=False,
    )

    st.sidebar.divider()

    if st.sidebar.button("Загрузить документы", use_container_width=True):
        upload_documents_dialog()
    if st.sidebar.button("Загруженные документы", use_container_width=True):
        documents_dialog(documents)

    st.sidebar.divider()

    folders = sorted({document_folder(doc) for doc in documents})
    folder_filter = st.sidebar.selectbox("Папка", ["Все папки"] + folders)
    search_query = st.sidebar.text_input("Поиск", placeholder="Название, версия, id...")
    visible_documents = [
        doc for doc in documents if document_matches(doc, search_query.strip(), folder_filter)
    ]

    selected_doc_index = None
    left_doc_index = None
    right_doc_index = None
    labels = [document_label(doc) for doc in visible_documents]

    if mode == "Чат":
        st.sidebar.caption("Активный документ")
        if visible_documents:
            selected_doc_index = st.sidebar.selectbox(
                "Документ",
                range(len(visible_documents)),
                format_func=lambda i: labels[i],
                label_visibility="collapsed",
            )
        else:
            st.sidebar.info("Нет документов по текущему фильтру.")

    if mode == "Сравнение":
        st.sidebar.caption("Версии для сравнения")
        if len(visible_documents) >= 2:
            left_doc_index = st.sidebar.selectbox(
                "Левая версия",
                range(len(visible_documents)),
                format_func=lambda i: labels[i],
                key="sidebar_compare_left",
            )
            right_doc_index = st.sidebar.selectbox(
                "Правая версия",
                range(len(visible_documents)),
                index=1,
                format_func=lambda i: labels[i],
                key="sidebar_compare_right",
            )
        else:
            st.sidebar.info("Для сравнения нужны минимум два документа по текущему фильтру.")

    if st.sidebar.button("Обновить", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    render_environment_panel()

    return {
        "mode": mode,
        "selected_doc_index": selected_doc_index,
        "left_doc_index": left_doc_index,
        "right_doc_index": right_doc_index,
        "documents": visible_documents,
    }


def render_chat(
    documents: List[Dict[str, str]],
    selected_doc_index: int | None,
) -> None:
    st.title("Чат")
    st.markdown(
        '<div class="mode-caption">Задавайте вопросы по выбранному документу.</div>',
        unsafe_allow_html=True,
    )

    if selected_doc_index is None or not documents:
        st.markdown(
            '<div class="empty-chat">Слева загрузите документ или выберите уже загруженный.</div>',
            unsafe_allow_html=True,
        )
        return

    selected_doc = documents[selected_doc_index]
    doc_id = selected_doc["doc_id"]
    history_key = f"chat_history:{doc_id}"

    top_left, top_right = st.columns([1, 0.18])
    with top_left:
        st.caption(f"Документ: {document_label(selected_doc)}")
    with top_right:
        if st.button("Очистить", use_container_width=True):
            st.session_state[history_key] = []
            st.rerun()

    search_scope = st.segmented_control(
        "Область поиска",
        ["Выбранный документ", "Папка"],
        default="Выбранный документ",
    )
    answer_mode = st.segmented_control(
        "Режим ответа",
        ["Кратко", "Подробно", "Строго с цитатами"],
        default="Кратко",
    )
    show_fragments = st.checkbox("Показать найденные фрагменты", value=False)

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    messages = st.session_state[history_key]

    if not messages:
        with st.chat_message("assistant"):
            st.markdown("Готов отвечать по выбранному документу. Задайте вопрос внизу.")

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("answer_mode"):
                scope_label = message.get("search_scope", "Выбранный документ")
                st.caption(f"Режим ответа: {message['answer_mode']} · область: {scope_label}")
            if message.get("answer_source") == "fallback":
                st.warning("LLM не ответил, поэтому показан fallback-ответ.")
            st.markdown(message["content"])
            render_sources(message.get("sources", []), show_fragments)

    question = st.chat_input("Сообщение")
    if not question:
        return

    messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ в документе..."):
                service = get_rag_service()
                if search_scope == "Папка":
                    folder_docs = [
                        doc for doc in documents
                        if document_folder(doc) == document_folder(selected_doc)
                    ]
                    doc_indexes = [get_document_index(doc["doc_id"]) for doc in folder_docs]
                    result = service.ask_question_across_indexes(
                        doc_indexes,
                        question,
                        answer_mode=answer_mode,
                    )
                else:
                    doc_index = get_document_index(doc_id)
                    result = service.ask_question(
                        doc_index,
                        question,
                        answer_mode=answer_mode,
                    )
                answer = result["answer"]
                st.caption(f"Режим ответа: {answer_mode} · область: {search_scope}")
                if result.get("answer_source") == "fallback":
                    st.warning("LLM не ответил, поэтому показан fallback-ответ.")
                st.markdown(answer)
                render_sources(result.get("sources", []), show_fragments)

        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "answer_mode": answer_mode,
                "search_scope": search_scope,
                "answer_source": result.get("answer_source", "llm"),
                "llm_error": result.get("llm_error", ""),
                "context": result.get("context", ""),
                "sources": result.get("sources", []),
            }
        )
    except Exception as exc:
        error_text = f"Не удалось получить ответ: {exc}"
        st.error(error_text)
        messages.append({"role": "assistant", "content": error_text})


def render_compare(
    documents: List[Dict[str, str]],
    left_doc_index: int | None,
    right_doc_index: int | None,
) -> None:
    st.title("Сравнение")
    st.markdown(
        '<div class="mode-caption">Сравнение выводится как обычный ответ ассистента.</div>',
        unsafe_allow_html=True,
    )

    if left_doc_index is None or right_doc_index is None or len(documents) < 2:
        st.markdown(
            '<div class="empty-chat">Слева выберите две версии документов для сравнения.</div>',
            unsafe_allow_html=True,
        )
        return

    left_doc = documents[left_doc_index]
    right_doc = documents[right_doc_index]
    history_key = f"compare_history:{left_doc['doc_id']}:{right_doc['doc_id']}"

    left_col, right_col, action_col = st.columns([1, 1, 0.25])
    with left_col:
        st.caption(f"Левая: {document_label(left_doc)}")
    with right_col:
        st.caption(f"Правая: {document_label(right_doc)}")
    with action_col:
        if st.button("Очистить", use_container_width=True):
            st.session_state[history_key] = []
            st.rerun()

    if left_doc_index == right_doc_index:
        st.warning("Выберите две разные версии.")
        return

    if history_key not in st.session_state:
        st.session_state[history_key] = []

    if not st.session_state[history_key]:
        with st.chat_message("assistant"):
            st.markdown("Готов сравнить выбранные версии. Можно отправить вопрос внизу.")

    for message in st.session_state[history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("summary"):
                with st.expander("Сводка"):
                    st.json(message["summary"])

    query = st.chat_input("Например: что добавлено, удалено и изменено?")
    if not query:
        return

    st.session_state[history_key].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Сравниваю документы..."):
                service = get_rag_service()
                result = service.compare_documents_qa(
                    left_doc_id=left_doc["doc_id"],
                    right_doc_id=right_doc["doc_id"],
                    user_query=query,
                )
                answer = result["answer"]
                st.markdown(answer)
                with st.expander("Сводка"):
                    st.json(result["summary"])

        st.session_state[history_key].append(
            {
                "role": "assistant",
                "content": answer,
                "summary": result.get("summary", {}),
            }
        )
    except Exception as exc:
        error_text = f"Не удалось сравнить документы: {exc}"
        st.error(error_text)
        st.session_state[history_key].append({"role": "assistant", "content": error_text})


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_styles()

    documents = list_documents()
    state = render_sidebar(documents)
    visible_documents = state["documents"]

    if state["mode"] == "Чат":
        render_chat(visible_documents, state["selected_doc_index"])
    else:
        render_compare(visible_documents, state["left_doc_index"], state["right_doc_index"])


if __name__ == "__main__":
    main()
