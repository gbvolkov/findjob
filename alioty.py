"""Utility helpers for extracting text content from user-supplied files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union
from loader import get_loader, TextLoader, JSONLoader, fallback_json, fallback_text
from json import JSONDecodeError

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log", ".csv"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
_DOC_EXTENSIONS = {".doc", ".docx"}
_PDF_EXTENSIONS = {".pdf"}


def alioty(path: Union[str, Path], *, mime_type: Optional[str] = None) -> str:
    """Extract textual content from *path* based on file type hints.

    This is a lightweight placeholder that currently supports utf-8 text files and
    logs unsupported types for future implementation (PDF, DOCX, images).
    """
    file_path = Path(path)
    mime = (mime_type or "").lower()
    suffix = file_path.suffix.lower()

    if suffix in _TEXT_EXTENSIONS or mime.startswith("text/"):
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("alioty: failed to read text file", exc_info=exc, extra={"path": str(file_path)})
            return ""

    loader = get_loader(file_path.parent, file_path.name)
    if loader is None:
        logger.info("alioty: unsupported file type", extra={"path": str(file_path), "suffix": suffix, "mime": mime})
        return ""

    try:
        try:
            docs = loader.load()
        except Exception as e:
            docs = None
            if isinstance(loader, JSONLoader) and isinstance(e, JSONDecodeError):
                docs = fallback_json(file_path)
            elif isinstance(loader, TextLoader) and isinstance(e, RuntimeError):
                docs = fallback_text(file_path)
            if docs is None:
                raise e
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("alioty: failed to load file", exc_info=exc, extra={"path": str(file_path)})
        return ""

