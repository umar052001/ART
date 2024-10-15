import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import faiss
import json

class RAGPipeline:
    def __init__(self, embedding_model_name: str, vector_db_type: str = 'faiss', vector_db_config: dict = None):
        """
        Initialize the RAG pipeline with embedding model and vector database configuration.
        :param embedding_model_name: The name of the Hugging Face model to be used for embeddings.
        :param vector_db_type: Type of vector database ('faiss' or 'elasticsearch').
        :param vector_db_config: Configurations for the vector database (e.g., Elasticsearch config or FAISS index size).
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_db_type = vector_db_type
        self.vector_db = None
        self.vector_db_config = vector_db_config if vector_db_config else {}

        if vector_db_type == 'faiss':
            self._init_faiss()
        elif vector_db_type == 'elasticsearch':
            self._init_elasticsearch()

    def _init_faiss(self):
        """
        Initialize FAISS index.
        """
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance metric
        self.vector_db = self.index

    def _init_elasticsearch(self):
        """
        Initialize Elasticsearch connection.
        """
        self.es_client = Elasticsearch(self.vector_db_config.get('hosts', ['http://localhost:9200']))
        self.index_name = self.vector_db_config.get('index_name', 'rag_vector_index')
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, ignore=400)

    def embed_documents(self, documents: list):
        """
        Embed a list of documents using the SentenceTransformer model.
        :param documents: A list of documents to embed.
        :return: List of embeddings.
        """
        return self.embedding_model.encode(documents, show_progress_bar=True)

    def store_embeddings(self, embeddings: np.array, metadata: list = None):
        """
        Store document embeddings in the vector database (either FAISS or Elasticsearch).
        :param embeddings: Embeddings array to store.
        :param metadata: Metadata for each document (if needed for Elasticsearch).
        """
        if self.vector_db_type == 'faiss':
            self.vector_db.add(embeddings)  # FAISS stores embeddings
        elif self.vector_db_type == 'elasticsearch':
            for i, embedding in enumerate(embeddings):
                doc = {
                    'embedding': embedding.tolist(),
                    'metadata': metadata[i] if metadata else {}
                }
                self.es_client.index(index=self.index_name, body=json.dumps(doc))

    def retrieve_similar(self, query: str, k: int = 5):
        """
        Retrieve the top-k similar documents for a given query.
        :param query: The input query.
        :param k: Number of top similar documents to retrieve.
        :return: List of similar documents.
        """
        query_embedding = self.embedding_model.encode([query])[0]

        if self.vector_db_type == 'faiss':
            distances, indices = self.vector_db.search(np.array([query_embedding]), k)
            return indices, distances
        elif self.vector_db_type == 'elasticsearch':
            query_body = {
                'query': {
                    'script_score': {
                        'query': {"match_all": {}},
                        'script': {
                            'source': "cosineSimilarity(params.query_embedding, 'embedding') + 1.0",
                            'params': {'query_embedding': query_embedding.tolist()}
                        }
                    }
                }
            }
            response = self.es_client.search(index=self.index_name, body=query_body, size=k)
            return response['hits']['hits']

    def process_documents(self, documents: list):
        """
        Complete pipeline for processing and storing documents.
        :param documents: List of document texts.
        """
        embeddings = self.embed_documents(documents)
        self.store_embeddings(embeddings)

    def query(self, query: str, k: int = 5):
        """
        Query the RAG system for the top-k results.
        :param query: The input query.
        :param k: Number of top results.
        """
        return self.retrieve_similar(query, k)

# Initialize the pipeline
pipeline = RAGPipeline(embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', vector_db_type='faiss')

# Process documents (embedding and storing)
# TODO: Import the custom splitter and loader here and pass the list of processed documents to the pipeline
documents = ["Document 1 text", "Document 2 text", "Document 3 text"] # example docs
pipeline.process_documents(documents)

# Query for similar documents
results = pipeline.query("What is the content of Document 1?", k=3)
print(results)
