from typing import List


import pandas as pd
import bm25s
from bm25s import BM25

from swift_index.base_index import BaseIndex


class BM25Index(BaseIndex):

    def __init__(self):
        self.doc_ids: List[str] = None
        self.index: BM25 = None
        super().__init__()

    def build(
        self,
        doc_ids: List[str],
        docs: List[str],
    ):

        # Add doc_id to the list
        self.doc_ids = doc_ids

        # Create index
        tokenized_corpus = bm25s.tokenize(docs, stopwords="en")
        self.index = BM25()
        self.index.index(tokenized_corpus)

        return

    def search(
        self,
        query: str,
        num_results: int = 5,
        return_scores: bool = False,
    ) -> pd.DataFrame:
        """Perform search for most sim"""

        tokenized_query = bm25s.tokenize(query, stopwords="en", show_progress=False)
        results, scores = self.index.retrieve(
            tokenized_query, corpus=self.doc_ids, k=num_results
        )

        if return_scores:
            return results[0], scores[0]
        else:
            return results[0]