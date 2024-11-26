from typing import List

import pandas as pd
import bm25s
from bm25s import BM25


class BM25Index:

    def __init__(self):
        self.corpus_ids: List[str] = None
        self.index: BM25 = None

    def build(
        self,
        content_id_list: List[str],
        content: List[str],
    ):

        # Add content_id to the list
        self.corpus_ids = content_id_list

        # Create index
        tokenized_corpus = bm25s.tokenize(content, stopwords="en")
        self.index = BM25()
        self.index.index(tokenized_corpus)

        return

    def search(
        self,
        query: str,
        num_results: int = 10,
        return_scores: bool = False,
    ) -> pd.DataFrame:
        """Perform search for most sim"""

        tokenized_query = bm25s.tokenize(query, stopwords="en", show_progress=False)
        results, scores = self.index.retrieve(
            tokenized_query, corpus=self.corpus_ids, k=num_results
        )

        if return_scores:
            return results[0], scores[0]
        else:
            return results[0]