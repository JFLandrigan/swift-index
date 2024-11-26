from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from poc.search_utils.base_index import BaseIndex
from poc.search_utils.bm25_index import BM25Index
from poc.search_utils.faiss_index import FAISSIndex
from poc.search_utils.sparse_index import SparseIndex

SPARSE_TYPES: List[str] = ["bm25", "tfidf", "count"]


class HybridIndex(BaseIndex):
    def __init__(self, sparse_type: str) -> None:
        super().__init__()
        self.dense_index = FAISSIndex()

        if sparse_type in SPARSE_TYPES:
            self.sparse_type = sparse_type
            if sparse_type == "bm25":
                self.sparse_index = BM25Index()
            else:
                self.sparse_index = SparseIndex()
        else:
            raise ValueError(f"sparse_type expects: {SPARSE_TYPES}, got {sparse_type}")

        self.num_items: int = None

    def build(
        self,
        content_id_list: List[str],
        content: List[str],
        dense_transformer: SentenceTransformer,
        embeddings: Optional[np.ndarray] = None,
    ):
        self.num_items = len(content_id_list)

        self.dense_index.build(
            content_id_list=content_id_list,
            content=content,
            tansformer=dense_transformer,
            embeddings=embeddings,
        )

        if self.sparse_type == "bm25":
            self.sparse_index.build(content_id_list=content_id_list, content=content)
        else:
            self.sparse_index.build(
                content_id_list=content_id_list,
                content=content,
                transform=self.sparse_type,
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
        tmp = pd.DataFrame({"content_id": dres, "dense_score": d_scores})
        tmp_bm = pd.DataFrame({"content_id": sres, "sparse_score": s_scores})

        tmp = tmp.merge(tmp_bm, how="inner", on="content_id")

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
                tmp["content_id"].tolist()[:num_results],
                tmp["score"].tolist()[:num_results],
            )
        else:
            return tmp["content_id"].tolist()[:num_results]