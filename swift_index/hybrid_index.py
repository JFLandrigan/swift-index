from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from swift_index.base_index import BaseIndex
from swift_index.bm25_index import BM25Index
from swift_index.faiss_index import FAISSIndex
from swift_index.keyword import KeywordIndex
from swift_index.sparse_index import SparseIndex

SPARSE_TYPES: List[str] = ["bm25", "tfidf", "count"]


class HybridIndex(BaseIndex):
    def __init__(self, sentence_transformer_name: str,, keyword_transform: str) -> None:
        super().__init__()
        self.dense_index : FAISSIndex = None
        self.sentence_transformer_name : str = sentence_transformer_name

        self.keyword_index : KeywordIndex = None

        if keyword_transform in SPARSE_TYPES:
            self.keyword_transform : str = keyword_transform
        else:
            raise ValueError(f"keyword_transform expected one of {SPARSE_TYPES}, got {keyword_transform}")

        self.num_items: int = None

    def build(
        self,
        doc_ids: List[str],
        docs: List[str],
        embeddings: Optional[np.ndarray] = None,
    ):
        self.num_items = len(doc_ids)

        self.dense_index.build(
            doc_ids=doc_ids,
            docs=docs,
            transformer=self.sentence_transformer_name,
            embeddings=embeddings,
        )

        self.keyword_index.build(
            doc_ids=doc_ids, 
            docs=docs, 
            transform=self.keyword_transform
        )

        return

    def _min_max_normalize(self, column):
        return (column - column.min()) / (column.max() - column.min())

    def _rrf_scoring(self, data: pd.DataFrame, k: int = 60) -> pd.DataFrame:
        """Perform reciprocal rank fusion scoring combination"""
        data["rank1"] = data["dense_score"].rank(ascending=False, method="min")
        data["rank2"] = data["sparse_score"].rank(ascending=False, method="min")

        # Reciprocal Rank Fusion
        data["score"] = 1 / (k + data["rank1"]) + 1 / (k + data["rank2"])

        return data

    def search(
        self,
        query: str,
        num_results: int = 10,
        return_scores: bool = False,
        alpha: float = 0.5,
        rrf_k: Optional[int] = None,
        score_thresh: float = None,
    ):

        dres, d_scores = self.dense_index.search(
            query=query, num_results=self.num_items, return_dists=True
        )
        sres, s_scores = self.sparse_index.search(
            query=query, num_results=self.num_items, return_scores=True
        )

        # Combine the results
        tmp = pd.DataFrame({"doc_id": dres, "dense_score": d_scores})
        tmp_bm = pd.DataFrame({"doc_id": sres, "sparse_score": s_scores})

        tmp = tmp.merge(tmp_bm, how="inner", on="doc_id")

        # Get weighted combined scores
        sparse_wt = 1 - alpha
        # flip dense so larger means more sim
        tmp["dense_score"] = 1 - tmp["dense_score"]
        # Normalize so on same scale
        tmp["dense_score"] = self._min_max_normalize(tmp["dense_score"])
        tmp["sparse_score"] = self._min_max_normalize(tmp["sparse_score"])

        # Perform weighted calc
        if rrf_k:
            tmp = self._rrf_scoring(data=tmp, k=rrf_k)
        else:
            if alpha > 1 or alpha < 0:
                raise ValueError(f"alpha must be between 0 and 1 got: {alpha}")
            else:
                tmp["score"] = (alpha * tmp["dense_score"]) + (
                    sparse_wt * tmp["sparse_score"]
                )

        if score_thresh:
            tmp = tmp[tmp["score"] > score_thresh].copy(deep=True)

        tmp.sort_values(by="score", ascending=False, inplace=True)

        if return_scores:
            return (
                tmp["doc_id"].tolist()[:num_results],
                tmp["score"].tolist()[:num_results],
            )
        else:
            return tmp["doc_id"].tolist()[:num_results]