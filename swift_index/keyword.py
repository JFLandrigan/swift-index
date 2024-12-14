from typing import List, Union

from swift_index.base_index import BaseIndex
from swift_index.bm25_index import BM25Index
from swift_index.sparse_index import SparseIndex

SPARSE_TRANSFORMS = ['tfidf', 'count']

class KeywordIndex(BaseIndex):

    def __init__(self):
        self.index: Union[BM25Index, SparseIndex]  = None
        self.transform_type: str = None
        super().__init__()

    def build(self, doc_ids: List[str], docs: List[str], transform: str,):

        if transform in SPARSE_TRANSFORMS:
            self.transform_type = transform
            self.index = SparseIndex()
            self.index.build(doc_ids=doc_ids, docs=docs, transform=transform)
        elif transform == 'bm25':
            self.transform_type = transform
            self.index = BM25Index()
            self.index.build(doc_ids=doc_ids, docs=docs)
        else:
            raise ValueError(f"Expected bm25, tfidf, count, got {transform}")

        return

    def search(
        self, 
        query: str, 
        num_results: int = 5, 
        return_scores: bool = False,
    ):
        return self.index.search(
            query=query, 
            num_results=num_results, 
            return_scores=return_scores
        )
