import uuid

from swift_index.keyword import KeywordIndex


DOCS = [
	"This list of documents is used for testing the indeces",
	"All the indices are built using python modules for simplicity",
	"Keyword indices include count, tfidf and bm25",
	"The index types include keyword, semantic and hybrid.",
	"The hybrid index allows for different ranking methods",
	"Writing tests is important to make sure the indices are building correctly"
]

DOC_IDS = [str(uuid.uuid4()) for i in range(len(DOCS))]

DOC_LOOKUP = dict(zip(DOC_IDS, DOCS))

INDECES =[
	('keyword count', KeywordIndex(), {'doc_ids':DOC_IDS, 'docs':DOCS, 'transform':'count'}),
	('keywrod tfidf', KeywordIndex(), {'doc_ids':DOC_IDS, 'docs':DOCS, 'transform':'tfidf'}),
	('keywrod bm25', KeywordIndex(), {'doc_ids':DOC_IDS, 'docs':DOCS, 'transform':'bm25'}),
	# ('faiss', FAISSIndex, {'transformer':'sentence-transformers/all-MiniLM-L6-v2'}),
	# ('hybrid', HybridIndex, {'transformer':'sentence-transformers/all-MiniLM-L6-v2', 'sparse_transform':'bm25'})
]

QUERY = 'Hybrid search is best'

def test_builds():
	for name, index, params in INDECES:
		print(f"testing {name}")
		index.build(**params)
		res = index.search(query='hybrid')
		res = [DOC_LOOKUP[r] for r in res]
		print(f"{name} search results: {res} \n")

	print("All builds were successful")


def main():
	test_builds()

if __name__ == '__main__':
	main()