import re
import json
import os
from fastembed import TextEmbedding,LateInteractionTextEmbedding,SparseTextEmbedding
from qdrant_client import QdrantClient, models
#GLOBAL CONFIG
file_path = "sport-action-camera-mounts.json"
base_name = os.path.splitext(os.path.basename(file_path))[0]
collection_name = "Reranking_Hybrid_Search"
dense_encoder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
late_colbert_embedder = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
client = QdrantClient(url = "http://localhost:6333")

def load_and_clean():
    with open(file_path,'r',encoding='utf-8') as fd:
        data = json.load(fd)
    item_list = data["mods"]["listItems"]
    return item_list

def chunks_of_each_doc(doc,chunk_size=128):
    chunks = []
    doc = json.dumps(doc)
    start = 0
    tokens = re.findall(r'\w+|[{}[\]:,",]',doc)
    while start<=len(tokens):
        end = min(start+chunk_size,len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start+=chunk_size
    return chunks

def get_chunks_from_json():
    item_list = load_and_clean()
    all_chunks = []
    for doc in item_list:
        chunks = chunks_of_each_doc(doc)
        all_chunks.append(chunks)
    return all_chunks

def create_collection_and_payload_indexes():
    if not client.collection_exists(collection_name = collection_name):
        client.create_collection(
        collection_name = collection_name,
        vectors_config= {
            "dense" : models.VectorParams(
                size = dense_encoder.embedding_size,
                distance= models.Distance.COSINE
            ),
            "lateInteraction" : models.VectorParams(
                size = late_colbert_embedder.embedding_size,
                distance = models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0)
            )
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier = models.Modifier.IDF)}
        )
    client.create_payload_index(
    collection_name=collection_name,
    field_name = "doc_id",
    field_schema = "integer"
    )

    client.create_payload_index(
    collection_name=collection_name,
    field_name = "chunk_id",
    field_schema = "integer"
    )

    client.create_payload_index(
    collection_name=collection_name,
    field_name = "file_name",
    field_schema = "keyword"
    )

def data_to_vectordb():
    create_collection_and_payload_indexes()
    all_chunks = get_chunks_from_json()

    #embedding of the data

    all_embeds = []
    for i in range(len(all_chunks)):
        dense_embed = list(dense_encoder.embed(chunk for chunk in all_chunks[i]))
        all_embeds.append(dense_embed)
    
    all_sparse_embeds = []
    for i in range(len(all_chunks)):
        sparse_embeds = list(sparse_encoder.embed(chunk for chunk in all_chunks[i]))
        all_sparse_embeds.append(sparse_embeds)

    all_colbert_embeds = []
    for i in range(len(all_chunks)):
        colbert_embeds = list(late_colbert_embedder.embed(chunk for chunk in all_chunks[i]))
        all_colbert_embeds.append(colbert_embeds)
    
    #storage of vectors into database

    offset = 0
    doc_offset = 0
    info = client.get_collection(collection_name=collection_name)

    count = info.points_count
    if(count!=0):
        res, _ = client.scroll(
        collection_name = collection_name,
        limit = 1,
        with_payload=True,
        with_vectors=False,
        order_by={
            "key" : "chunk_id",
            "direction" : "desc"
            }
        )
        if(res):
            doc_number = res[0].payload.get("doc_id")
            last_id = res[0].id
            offset = last_id + 1
            doc_offset = doc_number + 1
        else:
            offset = 0
            doc_offset = 0
    else:
        offset = 0
        doc_offset = 0

    for doc in range (len(all_chunks)):
        for idx in range(len(all_chunks[doc])):
            client.upsert(
                collection_name = collection_name,
                points = [
                    models.PointStruct(
                    id = offset,
                    payload = {
                        "doc_id" : doc_offset,
                        "chunk_id" : offset,
                        "chunk": all_chunks[doc][idx],
                        "file_name" : base_name
                    },
                    vector = {
                        "dense" : all_embeds[doc][idx],
                        "sparse" : all_sparse_embeds[doc][idx].as_object(),
                        "lateInteraction" : all_colbert_embeds[doc][idx]
                    }
                )
            ]
        )
        offset+=1
    doc_offset+=1