# here the retrieval of similar points are done 
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
collection_name = "items_retrieval_with_vanilla_rag"
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(url = "http://localhost:6333")
def query_retrieval(query):
    hits = client.query_points(
        collection_name= collection_name,
        query = encoder.encode([query])[0].tolist(),
        with_payload=True,
        limit = 3,
        using = "dense"
    ).points

    return hits

if __name__ == "__main__":
    print("The Retrieved Points are")
    points = query_retrieval(query = "phone case for vivo")
    for point in points:
        print("score = ",point.score, "Point_id=", point.id, "doc_id = ", point.payload.get("doc_id"), "file_name = ",point.payload.get("file_name"), "chunk_id = " , point.payload.get("chunk_id"))
    