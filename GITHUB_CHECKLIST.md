# GitHub Publish Checklist

Use this checklist before pushing the project to GitHub.

## 1. Secrets

- Make sure `.env` is not committed.
- Make sure real `YANDEX_API_KEY` and `YANDEX_CLOUD_FOLDER` values are not present in committed files.
- Keep only `.env.example` in the repository.

If `.env` was committed earlier, rotate the Yandex API key and remove the secret from git history before making the repository public.

## 2. Heavy Local Data

Do not commit:

- `.venv/`;
- `models/`;
- `data/`;
- `my_chroma_db/`;
- `logs/`;
- `Ultralytics/`;
- local SQLite databases;
- uploaded documents;
- parsed PDF/Markdown artifacts.

These files are ignored by `.gitignore`.

## 3. Expected Files to Commit

The repository should mainly contain:

- application source files: `app.py`, `rag_service.py`, `db_2.py`, parsers, helpers;
- `requirements.txt`;
- `README.md`;
- `.env.example`;
- `.gitignore`;
- optional documentation files.

## 4. Suggested Local Commands

Run these before the first public push:

```powershell
git status --short
git diff --stat
git diff -- .gitignore .env.example README.md
```

Check whether ignored files are being tracked:

```powershell
git ls-files .env data models my_chroma_db logs .venv
```

If this command prints anything sensitive or heavy, remove it from the git index without deleting local files:

```powershell
git rm --cached -r .env data models my_chroma_db logs .venv
```

Then commit:

```powershell
git add .
git commit -m "Prepare project for GitHub"
```

## 5. Important Note

This project depends on local model files and local indexed data. A fresh clone from GitHub will need:

- Python environment setup;
- dependencies from `requirements.txt`;
- a local `.env`;
- downloaded or locally provided embedding/reranker models;
- new document uploads and re-indexing.
