from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os
from chunking_strategy import get_chunks_of_items

#Global Config
client = QdrantClient(url = "http://localhost:6333")
encoder = SentenceTransformer("all-MiniLM-L6=v2")
file_name = "camera-screen-guards.json"
collection_name = "items_retrieval_with_vanilla_rag"
base_name = os.path.splitext(os.path.basename(file_name))[0]

#this function calls the function from the 
# chunking_strategy.py and get the chunks for embedding

def data_embedding():
    final_embeddings_of_json = []
    all_chunks = get_chunks_of_items(filename=file_name)
    for i in range(len(all_chunks)):
        embeddings = encoder.encode(all_chunks[i],show_progress_bar = True, convert_to_numpy=True)
        final_embeddings_of_json.append(embeddings)
    return final_embeddings_of_json,all_chunks

def create_collection():
    client.create_collection(
        collection_name = collection_name,
        vectors_config = {
            "dense" : models.VectorParams(
                size = encoder.get_sentence_embedding_dimension(),
                distance = models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse" : models.SparseVectorParams()
        }
    )

def storage_and_payload_creation():
    create_collection()

    client.create_payload_index(
    collection_name = "daraz_items_with_docs",
    field_name = "chunk_id",
    field_schema = "integer"
    )

    client.create_payload_index(
    collection_name = "daraz_items_with_docs",
    field_name = "doc_id",
    field_schema = "integer"
    )
    
    client.create_payload_index(
    collection_name = "daraz_items_with_docs",
    field_name = "file_name",
    field_schema="keyword"
    )

    offset = 0

    info = client.get_collection(collection_name=collection_name)
    final_embeddings,all_chunks = data_embedding()
    for doc in len(final_embeddings):
        if(info.points_count != 0):
            res, _ = client.scroll(
                collection_name = collection_name,
                with_payload=False,
                with_vectors = False,
                limit = 1,
                order_by = {
                    "key" : "chunk_id",
                    "direction" : "desc"
                }
            )
            if(res):
                last_id = res[0].id
                offset = last_id + 1
            else:
                offset = 0
        else:
            offset = 0
        for idx in len(final_embeddings[doc]):
            client.upsert(
                collection_name = collection_name,
                points = [
                    models.PointStruct(
                        id = idx+offset,
                        payload = {
                            "doc_id" : doc,
                            "chunk_id": idx + offset,
                            "chunk": all_chunks[doc][idx],
                            "file_name" : base_name
                        },
                        vector = final_embeddings[doc][idx]
                    )
                ]
            )

