from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

reader = SimpleDirectoryReader(input_dir = "./data_sample")
document = reader.load_data()

# Splitting sentences with sentence coherence
parser = SentenceSplitter(chunk_size = 256, chunk_overlap = 50)
nodes = parser.get_nodes_from_documents(document)
print(f"Chunk count: {len(nodes)}")