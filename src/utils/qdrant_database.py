from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

class QdrantStore:
    def __init__(self, url: str = "http://localhost:6333", collection_name: str ="PDFdoc", dim=1024) -> None:
        self.client: QdrantClient = QdrantClient(url=url)
        self.collection_name: str = collection_name
        
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dim, 
                    distance=Distance.COSINE
                ),
            )
            
    def count_vectors(self) -> int:
            """Returns the number of vectors currently in the collection"""
            return self.client.count(
                collection_name=self.collection_name
            ).count
    
    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i], 
                vector=vectors[i], 
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def similarity_search(self, query_vector, top_k: int = 5):
        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()

        # 2. Use 'query_points' (Robust replacement for .search)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        ).points


        contexts = []
        sources = set()

        for r in results:
            payload = r.payload or {}
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                contexts.append(text)
                sources.add(source)

        return {
            'contexts': contexts,
            'sources': list(sources)
        }
