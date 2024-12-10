from swift_index.base_index import BaseIndex
from swift_index.faiss_index import FAISSIndex


class SemanticIndex(BaseIndex):
    def __init__(self):
        self.index: FAISSIndex  = None
        self.lookup: Dict[int, str] = None
        super().__init__()

    def build(self):
        self.index.build()
        return

    def search(self):
        return
