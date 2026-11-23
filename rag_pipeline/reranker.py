"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Takes a user query + candidate chunks → returns chunks re-ordered by
relevance score.  This corrects ordering errors from approximate retrieval.

Model notes:
  - bge-reranker-v2-m3 is multilingual, works well for Hindi terms mixed in.
  - Runs on CPU; 20-chunk batch ≈ 200-500ms on a modern CPU.
  - For GPU, torch automatically detects CUDA if available.
"""
from __future__ import annotations

import os
from rag_pipeline.config import RERANKER_MODEL, FINAL_TOP_K
from rag_pipeline.retriever import RetrievedChunk


class Reranker:
    """Cross-encoder reranker with resilient fallbacks (FlashRank / RRF)."""

    def __init__(self) -> None:
        self._model = None
        self._mode = "rrf"

        # Try FlashRank first (lightweight ONNX, zero dependency conflicts)
        try:
            from flashrank import Ranker
            self._model = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
            self._mode = "flashrank"
            return
        except Exception:
            pass

        # Try CrossEncoder with bge-reranker-v2-m3
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                RERANKER_MODEL,
                max_length=512,
                automodel_args={"torch_dtype": "auto"},
            )
            self._mode = "cross_encoder"
        except Exception as e:
            # Fall back to preserving retrieval RRF order
            self._mode = "rrf"
            self._model = None

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = FINAL_TOP_K,
    ) -> list[RetrievedChunk]:
        """Re-score all chunks and return top_k sorted by reranker score."""
        if not chunks:
            return []

        if self._mode == "flashrank" and self._model:
            try:
                passages = [{"id": str(i), "text": c.text} for i, c in enumerate(chunks)]
                from flashrank import RerankRequest
                req = RerankRequest(query=query, passages=passages)
                results = self._model.rerank(req)
                reordered = []
                for res in results[:top_k]:
                    idx = int(res["id"])
                    c = chunks[idx]
                    c.score = float(res.get("score", c.score))
                    reordered.append(c)
                return reordered
            except Exception:
                pass

        if self._mode == "cross_encoder" and self._model:
            try:
                pairs = [(query, chunk.text) for chunk in chunks]
                scores = self._model.predict(pairs, show_progress_bar=False)
                for chunk, score in zip(chunks, scores):
                    chunk.score = float(score)
                chunks.sort(key=lambda c: c.score, reverse=True)
                return chunks[:top_k]
            except Exception:
                pass

        # Default: Return top_k based on existing retrieval score
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]
