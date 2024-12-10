from typing import Union

from swift_index.base_index import BaseIndex
from swift_index.bm25_index import BM25Index
from swift_index.sparse_index import SparseIndex

SPARSE_TRANSFORMS = ['tfidf', 'count']

class KeyWordIndex(BaseIndex):
    def __init__(self):
        self.index: Union[BM25Index, SparseIndex]  = None
        super().__init__()

    def build(self, doc_ids: List[str], docs: List[str], transform: str,):
        if transform in SPARSE_TRANSFORMS:
            self.index = SparseIndex()
            self.index.build(doc_ids=doc_ids, docs=docs, transform=transform)
        elif transform == 'bm25':
            self.index = BM25Index()
            self.index.build(doc_ids=doc_ids, docs=docs)
        return

    def search(self, query: Union[str, np.ndarray], num_results: int = 10, return_scores: bool = False,):
        return self.index.search(query=query, num_results=num_results, return_scores=return_scores)
