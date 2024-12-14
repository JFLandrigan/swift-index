from typing import Dict, List, Optional, Union

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from swift_index.base_index import BaseIndex


class FAISSIndex(BaseIndex):
    def __init__(self):
        self.index: faiss.Index = None
        self.lookup: Dict[int, str] = None
        self.transformer: SentenceTransformer = None
        super().__init__()

    def build(
        self,
        doc_ids: List[str],
        transformer: Union[SentenceTransformer, str],
        embeddings: Optional[np.ndarray] = None,
        docs: List[str] = None,
    ):

        # init the transformer
        if isinstance(transformer, SentenceTransformer):
            self.transformer = transformer
        else:
            self.transformer = SentenceTransformer(transformer)

        # Add doc_id to the list
        self.lookup = {i: doc_ids[i] for i in range(len(doc_ids))}

        # Prep the embeddings
        if embeddings is None:
            embeddings = self.transformer.encode(docs)

        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        # Load up the index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        return

    def search(
        self,
        query: str,
        num_results: int = 5,
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

        if isinstance(query, str):
            query = self.transformer.encode([query])

        dists, ids = self.index.search(x=query, k=num_results)
        indeces = [i for i in ids[0]]

        if return_scores:
            d = [i for i in dists[0]]
            l = [self.lookup[ind] for ind in indeces]
            return l, d
        else:
            return [self.lookup[ind] for ind in indeces]