import os
from typing import Optional


def default_hf_cache_dir() -> str:
    return os.getenv(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )


def repo_id_to_cache_dir(repo_id: str, cache_dir: Optional[str] = None) -> str:
    base_cache = cache_dir or default_hf_cache_dir()
    repo_folder = "models--" + repo_id.replace("/", "--")
    return os.path.join(base_cache, "hub", repo_folder)


def resolve_hf_model_source(
    repo_id: str,
    explicit_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    if explicit_path:
        explicit_path = os.path.abspath(explicit_path)
        if os.path.exists(explicit_path):
            return explicit_path

    repo_cache_dir = repo_id_to_cache_dir(repo_id, cache_dir)
    snapshots_dir = os.path.join(repo_cache_dir, "snapshots")

    if os.path.isdir(snapshots_dir):
        snapshot_names = sorted(os.listdir(snapshots_dir), reverse=True)
        for snapshot_name in snapshot_names:
            snapshot_path = os.path.join(snapshots_dir, snapshot_name)
            if os.path.isdir(snapshot_path):
                return snapshot_path

    return repo_id


def has_local_hf_model(
    repo_id: str,
    explicit_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> bool:
    resolved = resolve_hf_model_source(
        repo_id=repo_id,
        explicit_path=explicit_path,
        cache_dir=cache_dir,
    )
    return os.path.exists(resolved)
