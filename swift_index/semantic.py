from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from swift_index.base_index import BaseIndex
from swift_index.faiss_index import FAISSIndex


class SemanticIndex(BaseIndex):
    def __init__(self):
        self.index: FAISSIndex  = None
        self.lookup: Dict[int, str] = None
        super().__init__()

    def build(
        self,
        doc_ids: List[str],
        tansformer: SentenceTransformer,
        embeddings: Optional[np.ndarray] = None,
        docs: List[str] = None,
    ):

        self.index = FAISSIndex()
        self.index.build(
            doc_ids=doc_ids, 
            docs=docs, 
            transformer=transformer, 
            embeddings=embeddings, 
            docs=docs
        )

        return

    def search(
        self,
        query: str,
        num_results: int = 10,
        return_scores: bool = False,
    ) -> pd.DataFrame:
        """Query the index and return the lookup metadata for top number of
        results.

        Args:
            query (np.array): Should be numpy array based on embeddings of the index.
            num_results (int): Number of results to return

        Notes:
            dists : smaller means more similar because closer in vector space.
        """

        return self.index(query=query, num_results=num_results, return_scores=return_scores)