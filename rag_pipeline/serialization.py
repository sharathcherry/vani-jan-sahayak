"""Safe serialization and deserialization utilities for RAG index artifacts.

Eliminates unsafe pickle deserialization vulnerabilities (CWE-502) across
the RAG pipeline by:
1. Replacing parent_docs and corpus with pure JSON serialization.
2. Hardening binary BM25 deserialization with:
   - Cryptographic SHA-256 integrity checksum verification before unpickling.
   - Restricted SafeUnpickler that permits only whitelisted data types
     (rank_bm25 classes and standard primitives), blocking any executable bytecode.
3. Providing seamless backward-compatibility and migration from legacy .pkl files.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DeserializationSecurityError(Exception):
    """Raised when deserialization security verification fails."""
    pass


# Strict whitelist for safe unpickling of primitive data structures
SAFE_PRIMITIVE_CLASSES: Set[Tuple[str, str]] = {
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "str"),
    ("builtins", "bytes"),
    ("builtins", "bool"),
    ("builtins", "tuple"),
    ("builtins", "NoneType"),
    ("collections", "defaultdict"),
    ("collections", "Counter"),
    ("collections", "OrderedDict"),
}

# Whitelist for BM25 model deserialization
SAFE_BM25_CLASSES: Set[Tuple[str, str]] = SAFE_PRIMITIVE_CLASSES | {
    ("rank_bm25", "BM25Okapi"),
    ("rank_bm25", "BM25"),
    ("rank_bm25", "BM25Plus"),
    ("rank_bm25", "BM25L"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler that only resolves explicitly whitelisted classes.

    Prevents arbitrary code execution exploits via __reduce__, os.system,
    subprocess, builtins.eval/exec, etc.
    """

    def __init__(self, file, allowed_classes: Optional[Set[Tuple[str, str]]] = None):
        super().__init__(file)
        self._allowed = allowed_classes if allowed_classes is not None else SAFE_BM25_CLASSES

    def find_class(self, module: str, name: str):
        if (module, name) in self._allowed:
            return super().find_class(module, name)
        raise DeserializationSecurityError(
            f"Security Error: Unauthorized class '{module}.{name}' encountered during "
            f"unpickling. Deserialization blocked to prevent code execution."
        )


# ── Cryptographic Checksum Helpers ──────────────────────────────────────────

def compute_file_sha256(filepath: str | Path) -> str:
    """Compute the SHA-256 hash of a file on disk."""
    p = Path(filepath)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_sha256_checksum(target_path: str | Path) -> Path:
    """Write a companion .sha256 file for target_path."""
    p = Path(target_path)
    checksum_path = p.with_suffix(p.suffix + ".sha256")
    hash_hex = compute_file_sha256(p)
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write(f"{hash_hex}  {p.name}\n")
    return checksum_path


def verify_sha256_checksum(target_path: str | Path, strict: bool = False) -> bool:
    """Verify that target_path matches its companion .sha256 file.

    Raises DeserializationSecurityError on hash mismatch or if missing when strict=True.
    """
    p = Path(target_path)
    checksum_path = p.with_suffix(p.suffix + ".sha256")
    if not checksum_path.exists():
        if strict:
            raise DeserializationSecurityError(
                f"Security Error: Required checksum file '{checksum_path}' does not exist for '{p}'."
            )
        logger.warning(
            "[SECURITY WARNING] No SHA-256 checksum file found for '%s'. "
            "Proceeding with restricted unpickler verification only.",
            p,
        )
        return False

    with open(checksum_path, "r", encoding="utf-8") as f:
        expected_hash = f.read().strip().split()[0]

    actual_hash = compute_file_sha256(p)
    if not hmac.compare_digest(actual_hash.lower(), expected_hash.lower()):
        raise DeserializationSecurityError(
            f"Security Error: SHA-256 integrity check failed for '{p}'! "
            f"Expected {expected_hash}, calculated {actual_hash}. "
            f"File may be corrupted or tampered with."
        )
    return True


# ── Atomic File Write Helper ────────────────────────────────────────────────

def _atomic_write(target_path: str | Path, write_fn: Callable[[Any], None], binary: bool = False) -> Path:
    """Write to a temporary file in the same directory and atomically replace target."""
    p = Path(target_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"

    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=str(p.parent),
        prefix=f".{p.name}.tmp_",
    )
    tmp_path = Path(tmp_path_str)
    try:
        with open(tmp_fd, mode=mode, encoding=encoding) as f:
            write_fn(f)
        os.replace(tmp_path, p)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return p


# ── Safe JSON Read / Write ──────────────────────────────────────────────────

def safe_json_dump(data: Any, filepath: str | Path, indent: Optional[int] = 2) -> Path:
    """Atomically write JSON data with UTF-8 encoding and generate a companion SHA-256."""
    p = _atomic_write(filepath, lambda f: json.dump(data, f, ensure_ascii=False, indent=indent), binary=False)
    write_sha256_checksum(p)
    return p


def safe_json_load(filepath: str | Path, verify_checksum: bool = False) -> Any:
    """Safely load JSON data, optionally verifying the SHA-256 checksum."""
    p = Path(filepath)
    if verify_checksum:
        verify_sha256_checksum(p, strict=False)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Hardened Pickle Read / Write ────────────────────────────────────────────

def safe_pickle_dump(obj: Any, filepath: str | Path, protocol: int = pickle.HIGHEST_PROTOCOL) -> Path:
    """Atomically pickle an object and create a companion SHA-256 checksum file."""
    p = _atomic_write(filepath, lambda f: pickle.dump(obj, f, protocol=protocol), binary=True)
    write_sha256_checksum(p)
    return p


def safe_pickle_load(
    filepath: str | Path,
    allowed_classes: Optional[Set[Tuple[str, str]]] = None,
    verify_checksum: bool = True,
    strict_checksum: bool = False,
) -> Any:
    """Load pickled object using restricted SafeUnpickler and SHA-256 verification."""
    p = Path(filepath)
    if verify_checksum:
        verify_sha256_checksum(p, strict=strict_checksum)

    with open(p, "rb") as f:
        unpickler = SafeUnpickler(f, allowed_classes=allowed_classes)
        return unpickler.load()


# ── BM25 Serialization (JSON + Hardened Binary) ─────────────────────────────

def bm25_to_dict(bm25: Any) -> Dict[str, Any]:
    """Extract BM25Okapi state into a pure JSON-serializable dictionary."""
    return {
        "k1": float(bm25.k1),
        "b": float(bm25.b),
        "epsilon": float(bm25.epsilon),
        "corpus_size": int(bm25.corpus_size),
        "avgdl": float(bm25.avgdl),
        "doc_freqs": bm25.doc_freqs,
        "idf": {str(k): float(v) for k, v in bm25.idf.items()},
        "doc_len": [int(l) for l in bm25.doc_len],
        "average_idf": float(getattr(bm25, "average_idf", 0.0)),
    }


def bm25_from_dict(data: Dict[str, Any]) -> Any:
    """Reconstruct BM25Okapi instance from dictionary without arbitrary code execution."""
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi.__new__(BM25Okapi)
    bm25.k1 = float(data["k1"])
    bm25.b = float(data["b"])
    bm25.epsilon = float(data["epsilon"])
    bm25.corpus_size = int(data["corpus_size"])
    bm25.avgdl = float(data["avgdl"])
    bm25.doc_freqs = data["doc_freqs"]
    bm25.idf = data["idf"]
    bm25.doc_len = data["doc_len"]
    bm25.tokenizer = None
    bm25.average_idf = float(data.get("average_idf", 0.0))
    return bm25


def save_bm25_safe(
    bm25: Any,
    pkl_path: str | Path,
    json_path: Optional[str | Path] = None,
) -> None:
    """Save BM25 index with SHA-256 verified binary pickle and optional pure JSON sidecar."""
    safe_pickle_dump(bm25, pkl_path)
    if json_path is not None:
        try:
            bm25_data = bm25_to_dict(bm25)
            safe_json_dump(bm25_data, json_path, indent=None)
        except Exception as e:
            logger.warning("Failed to save BM25 JSON sidecar: %s", e)


def load_bm25_safe(
    pkl_path: str | Path,
    json_path: Optional[str | Path] = None,
) -> Any:
    """Load BM25 index safely.

    Tries SHA-256 verified binary pickle with SafeUnpickler first for maximum speed.
    Falls back to pure JSON if binary is missing, corrupted, or tampered.
    """
    p_pkl = Path(pkl_path)
    p_json = Path(json_path) if json_path else p_pkl.with_suffix(".json")

    # 1. Try binary load with checksum and restricted unpickler
    if p_pkl.exists():
        try:
            bm25 = safe_pickle_load(p_pkl, allowed_classes=SAFE_BM25_CLASSES, verify_checksum=True)
            # If JSON sidecar is missing, export it asynchronously/safely
            if not p_json.exists():
                try:
                    safe_json_dump(bm25_to_dict(bm25), p_json, indent=None)
                except Exception:
                    pass
            return bm25
        except DeserializationSecurityError:
            # Re-raise explicit security violations
            raise
        except Exception as exc:
            logger.warning("[serialization] Failed to load binary BM25 from %s: %s", p_pkl, exc)

    # 2. Try JSON fallback
    if p_json.exists():
        logger.info("[serialization] Loading BM25 from safe JSON: %s", p_json)
        data = safe_json_load(p_json)
        return bm25_from_dict(data)

    raise FileNotFoundError(
        f"BM25 index not found at '{pkl_path}' or '{p_json}'. "
        "Run indexer / ingest to build the index."
    )


# ── Parent Docs Safe Helpers ────────────────────────────────────────────────

def save_parent_docs_safe(
    parent_docs: Dict[str, Dict[str, Any]],
    json_path: str | Path,
    legacy_pkl_path: Optional[str | Path] = None,
) -> None:
    """Save parent documents dictionary to safe JSON and optional legacy verified pickle."""
    safe_json_dump(parent_docs, json_path, indent=2)
    if legacy_pkl_path is not None:
        safe_pickle_dump(parent_docs, legacy_pkl_path)


def load_parent_docs_safe(
    json_path: str | Path,
    legacy_pkl_path: Optional[str | Path] = None,
    default: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load parent documents, preferring safe JSON with safe restricted fallback from legacy .pkl."""
    p_json = Path(json_path)
    if p_json.exists():
        return safe_json_load(p_json)

    if legacy_pkl_path is not None:
        p_pkl = Path(legacy_pkl_path)
        if p_pkl.exists():
            logger.info("[serialization] Migrating legacy parent_docs pickle to safe JSON: %s", p_json)
            # Restrict unpickling strictly to primitive data types
            data = safe_pickle_load(p_pkl, allowed_classes=SAFE_PRIMITIVE_CLASSES, verify_checksum=False)
            if isinstance(data, dict):
                safe_json_dump(data, p_json, indent=2)
                return data

    if default is not None:
        return default
    raise FileNotFoundError(f"parent_docs not found at '{json_path}' or '{legacy_pkl_path}'.")


# ── Corpus Safe Helpers ─────────────────────────────────────────────────────

def save_corpus_safe(
    corpus: List[Dict[str, Any]],
    json_path: str | Path,
    legacy_pkl_path: Optional[str | Path] = None,
) -> None:
    """Save corpus chunks to safe JSON and optional legacy verified pickle."""
    safe_json_dump(corpus, json_path, indent=None)
    if legacy_pkl_path is not None:
        safe_pickle_dump(corpus, legacy_pkl_path)


def load_corpus_safe(
    json_path: str | Path,
    legacy_pkl_path: Optional[str | Path] = None,
    default: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Load corpus chunks, preferring safe JSON with safe restricted fallback from legacy .pkl."""
    p_json = Path(json_path)
    if p_json.exists():
        return safe_json_load(p_json)

    if legacy_pkl_path is not None:
        p_pkl = Path(legacy_pkl_path)
        if p_pkl.exists():
            logger.info("[serialization] Migrating legacy corpus pickle to safe JSON: %s", p_json)
            data = safe_pickle_load(p_pkl, allowed_classes=SAFE_PRIMITIVE_CLASSES, verify_checksum=False)
            if isinstance(data, list):
                safe_json_dump(data, p_json, indent=None)
                return data

    if default is not None:
        return default
    raise FileNotFoundError(f"Corpus not found at '{json_path}' or '{legacy_pkl_path}'.")
