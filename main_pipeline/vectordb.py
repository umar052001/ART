from sentence_transformers import SentenceTransformer, util
import faiss
from elasticsearch import Elasticsearch
import numpy as np

class VectorDB:
    def __init__(self, db_type="faiss", dimension=384, es_host="localhost", es_port=9200, index_name="document_vectors"):
        self.db_type = db_type
        self.dimension = dimension
        self.index_name = index_name
        self.model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  # Load a Hugging Face model

        # Initialize Elasticsearch or FAISS based on db_type
        if db_type == "elasticsearch":
            self.es = Elasticsearch([{'host': es_host, 'port': es_port}])
            self._setup_elasticsearch_index()
        elif db_type == "faiss":
            self.index = faiss.IndexFlatL2(dimension)
    
    def _setup_elasticsearch_index(self):
        # Setup an index in Elasticsearch for storing document embeddings
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body={
                "mappings": {
                    "properties": {
                        "vector": {"type": "dense_vector", "dims": self.dimension},
                        "text": {"type": "text"}
                    }
                }
            })

    def generate_embedding(self, text):
        # Generate embeddings for the given text
        return self.model.encode(text)
    
    def index_document(self, text):
        # Generate embedding for text and index in vector DB
        embedding = self.generate_embedding(text)
        if self.db_type == "faiss":
            self.index.add(np.array([embedding]))  # FAISS indexing
        elif self.db_type == "elasticsearch":
            doc = {"text": text, "vector": embedding.tolist()}
            self.es.index(index=self.index_name, body=doc)  # Elasticsearch indexing
    
    def search(self, query, top_k=5):
        query_embedding = self.generate_embedding(query)
        if self.db_type == "faiss":
            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            return distances, indices
        elif self.db_type == "elasticsearch":
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_embedding.tolist()}
                    }
                }
            }
            response = self.es.search(index=self.index_name, body={"query": script_query, "size": top_k})
            return response['hits']['hits']

# Example usage:
db = VectorDB(db_type="faiss")
db.index_document("London is the capital of England.")
db.index_document("Paris is the capital of France.")
distances, indices = db.search("What is the capital of France?")
print(distances, indices)
