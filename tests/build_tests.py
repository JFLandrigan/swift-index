import uuid

from swift_index.bm25_index import BM25Index
from swift_index.faiss_index import FAISSIndex
from swift_index.hybrid_index import HybridIndex
from swift_index.sparse_index import SparseIndex


DOCS = [
	"This list of documents is used for testing the indeces",
	"All the indices are built using python modules for simplicity",
	"Keyword indices include count, tfidf and bm25",
	"The index types include keyword, semantic and hybrid.",
	"The hybrid index allows for different ranking methods"
]

DOCS_IDS = [str(uuid.uuid4()) for i in len(docs)]

INDECES =[
	('bm25', '', BM25Index),
	('sparse count', SparseIndex, {'transform':'count'}),
	('sparse tfidf', SparseIndex, {'transform':'tfidf'}),
	('faiss', FAISSIndex, {'transformer':'sentence-transformers/all-MiniLM-L6-v2'}),
	('hybrid', HybridIndex, {'transformer':'sentence-transformers/all-MiniLM-L6-v2', 'sparse_transform':'bm25'})
]

def test_builds():
	for name, index, params in INDECES:
		print(f"testing {name}")
		index.build(**params)

	print("All builds were successful")

