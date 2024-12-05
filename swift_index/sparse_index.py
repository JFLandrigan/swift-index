from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TRANSFORM_TYPES = ["tfidf", "count"]


class SparseIndex:

    def __init__(self):
        self.sparse_matrix: csr_matrix = None
        self.lookup: Dict[int, str] = None
        self.transformer: Union[CountVectorizer, TfidfVectorizer] = None

    def build(
        self,
        content: List[str],
        content_id_list: List[str],
        transform: str,
    ):

        # init the transformer
        if transform in TRANSFORM_TYPES:
            if transform == "tfidf":
                self.transformer = TfidfVectorizer()
            elif transform == "count":
                self.transformer = CountVectorizer()
        else:
            raise ValueError(
                f"Expected transform of type {TRANSFORM_TYPES} got {transform}"
            )

        # Add content_id to the list
        self.lookup = {i: content_id_list[i] for i in range(len(content_id_list))}

        # Create the matrix
        self.sparse_matrix = self.transformer.fit_transform(content)

        return

    def search(
        self,
        query: Union[str, np.ndarray],
        num_results: int = 10,
        return_sims: bool = False,
    ) -> pd.DataFrame:
        """Perform search for most sim"""

        # transform the query to sparse vector if it is string.
        if isinstance(query, str) and self.transformer is not None:
            vector = self.transformer.transform([query])

        # get cosine sim scores
        similarity_matrix = cosine_similarity(vector, self.sparse_matrix)[0]
        top_inds = similarity_matrix.argsort()[-num_results:]
        top_sims = [similarity_matrix[ind] for ind in top_inds]

        if return_sims:
            return [self.lookup[ind] for ind in top_inds], top_sims
        else:
            return [self.lookup[ind] for ind in top_inds]