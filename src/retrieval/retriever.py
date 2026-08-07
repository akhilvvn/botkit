from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def query(self, text: str, top_k: int = 3) -> list[str]:
        ...

    @abstractmethod
    def add_documents(self, texts: list[str]) -> None:
        ...


class LangChainRetriever(Retriever):
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore

        self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self._store = InMemoryVectorStore(self._embeddings)

    def add_documents(self, texts: list[str]) -> None:
        from langchain_core.documents import Document

        docs = [Document(page_content=t) for t in texts]
        self._store.add_documents(docs)

    def query(self, text: str, top_k: int = 3) -> list[str]:
        results = self._store.similarity_search(text, k=top_k)
        return [doc.page_content for doc in results]
