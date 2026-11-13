from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient

reader = SimpleDirectoryReader(input_dir = "./data_sample")
document = reader.load_data()

# Splitting sentences without losing sentence coherence
parser = SentenceSplitter(chunk_size = 256, chunk_overlap = 50)
nodes = parser.get_nodes_from_documents(document)

model = Ollama(model = "llama3.2:3b")
embedding = OllamaEmbedding(model_name="nomic-embed-text")

# Setting up non-volatile Chroma DB
chroma_client = PersistentClient(path = "./chroma_db")
vector_store = ChromaVectorStore(chroma_client = chroma_client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes, storage_context = storage_context)
query_engine = index.as_query_engine(llm = model, embed_model = embedding, similarity_top_k = 3)

questions = [
        "What is the optimal tilt angle for a site at 40 degrees north latitude?",
        "How often should solar panels be cleaned?",
        "What is the cycle life of LiFePO4 cells at 80% depth of discharge?",
    ]

for i in questions:
    print("\n" + "=" * 60)
    print("Q:", i)
    r = query_engine.query(i)
    print("A:", r.response)