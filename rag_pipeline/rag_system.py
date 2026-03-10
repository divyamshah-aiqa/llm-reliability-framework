
from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Albert Einstein developed relativity",
    "Paris is capital of France",
    "RAG means Retrieval Augmented Generation"
]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

def rag_query(question):

    q_emb = embedder.encode(question, convert_to_tensor=True)

    hits = util.semantic_search(q_emb, corpus_embeddings)[0]

    docs = [corpus[h["corpus_id"]] for h in hits]

    return docs
