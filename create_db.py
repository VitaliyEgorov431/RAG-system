import os
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "./my_chroma_db"
COLLECTION_NAME = "docs"

EMBEDDING_MODEL = "ai-forever/sbert_large_nlu_ru"
DEVICE = "cpu"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


def preprocess_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace(" – ", "\n- ")
    text = text.replace("– ", "\n- ")
    return text


def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", ". ", " "],
    )


def stable_doc_id(source_path: str, version: str) -> str:
    # стабильный id документа (чтобы удобно удалять/переиндексировать)
    key = f"{os.path.basename(source_path)}::{version}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:16]


def add_file_to_db(md_path: str, version: str):
    if not os.path.exists(md_path):
        raise FileNotFoundError(md_path)

    with open(md_path, "r", encoding="utf-8") as f:
        raw_text = preprocess_text(f.read())

    splitter = make_splitter()
    doc_id = stable_doc_id(md_path, version)

    chunks = splitter.split_documents([
        Document(page_content=raw_text, metadata={"source": md_path, "version": version, "doc_id": doc_id})
    ])

    # проставим chunk_id и сделаем уникальные ids
    ids = []
    for i, d in enumerate(chunks):
        d.metadata["chunk_id"] = i
        ids.append(f"{doc_id}:{i}")   # уникальный id чанка

    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
    )

    # ВАЖНО: это добавление в существующую коллекцию
    db.add_documents(chunks, ids=ids)
    db.persist()

    print(f"OK: added {len(chunks)} chunks from {md_path} version={version} doc_id={doc_id}")


if __name__ == "__main__":
    # пример
    add_file_to_db("K:/document.md", "1")
    add_file_to_db("K:/document1.md", "2")