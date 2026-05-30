import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


DEFAULT_BUNDLE_DB_PATH = "./data/document_bundles.db"


def ensure_bundle_db(db_path: str = DEFAULT_BUNDLE_DB_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_bundles (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                document_name TEXT NOT NULL,
                version TEXT NOT NULL,
                bundle_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_document_bundles_name_version
            ON document_bundles(document_name, version)
            """
        )
        conn.commit()


def save_bundle(
    bundle: Dict[str, Any],
    db_path: str = DEFAULT_BUNDLE_DB_PATH,
) -> None:
    ensure_bundle_db(db_path)
    payload = json.dumps(bundle, ensure_ascii=False)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO document_bundles (
                doc_id, source, document_name, version, bundle_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(doc_id) DO UPDATE SET
                source=excluded.source,
                document_name=excluded.document_name,
                version=excluded.version,
                bundle_json=excluded.bundle_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(bundle.get("doc_id", "")),
                str(bundle.get("source", "")),
                str(bundle.get("document_name", "")),
                str(bundle.get("version", "")),
                payload,
            ),
        )
        conn.commit()


def load_bundle(
    doc_id: str,
    db_path: str = DEFAULT_BUNDLE_DB_PATH,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT bundle_json FROM document_bundles WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()

    if row is None:
        return None

    try:
        return json.loads(str(row[0]))
    except Exception:
        return None


def delete_bundle(
    doc_id: str,
    db_path: str = DEFAULT_BUNDLE_DB_PATH,
) -> None:
    if not os.path.exists(db_path):
        return

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM document_bundles WHERE doc_id = ?",
            (doc_id,),
        )
        conn.commit()


def list_bundles(
    db_path: str = DEFAULT_BUNDLE_DB_PATH,
) -> List[Dict[str, str]]:
    if not os.path.exists(db_path):
        return []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT doc_id, source, document_name, version, bundle_json
            FROM document_bundles
            ORDER BY document_name, version, source
            """
        ).fetchall()

    bundles: List[Dict[str, str]] = []
    for doc_id, source, document_name, version, bundle_json in rows:
        payload: Dict[str, Any] = {}
        try:
            payload = json.loads(str(bundle_json))
        except Exception:
            payload = {}

        bundles.append(
            {
                "doc_id": str(doc_id),
                "source": str(source),
                "document_name": str(document_name),
                "version": str(version),
                "document_folder": str(payload.get("document_folder", "Без папки")),
                "source_type": str(payload.get("source_type", "")),
                "original_filename": str(payload.get("original_filename", "")),
                "stored_source_path": str(payload.get("stored_source_path", "")),
                "text_source_path": str(payload.get("text_source_path", "")),
            }
        )

    return bundles


def backfill_bundles_from_dir(
    processed_dir: str,
    db_path: str = DEFAULT_BUNDLE_DB_PATH,
) -> int:
    if not os.path.isdir(processed_dir):
        return 0

    imported = 0
    for name in os.listdir(processed_dir):
        if not name.endswith(".json"):
            continue

        path = os.path.join(processed_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
        except Exception:
            continue

        doc_id = str(bundle.get("doc_id", "")).strip()
        if not doc_id:
            continue

        save_bundle(bundle, db_path=db_path)
        imported += 1

    return imported
