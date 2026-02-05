import sys
import os

# Get the absolute path of the current directory
current_dir = os.getcwd()

# Get the parent directory (which is 'RAG_Basic')
project_root = os.path.dirname(current_dir)
print(project_root)

# Add the project root to the system path
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.llm import CustomModel
from src.utils import DocumentProcessor, PDFLoader, QdrantStore
from sentence_transformers import SentenceTransformer
import uuid

"""
---------------------------------------------------------
1. SET UP
---------------------------------------------------------
"""
llm = CustomModel()
embed_model = SentenceTransformer("BAAI/bge-m3")
processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
vector_store = QdrantStore(dim=1024, collection_name="Apple Finance 2007")
loader = PDFLoader("C:\\Users\\dangq\\OneDrive\\Máy tính\\USTH\\ICT\\Internship\\RAG Remake\\RAG_Basic\\data\\raw\\NASDAQ_AAPL_2007.pdf")


"""
---------------------------------------------------------
2. Read and Chunk Documents (Preproces):
    - Chunking
    - Create vectors embedding for each chunks
    - Construct Metadata (need to improve in the future)
    - Upsert the data to vector database
---------------------------------------------------------
"""

# Check if we already have data
current_count = vector_store.count_vectors()

if current_count == 0:
    print("==="*50)
    print("Collection is empty. Starting processing and ingestion...")
    
    docs = loader.load_docs()
    docs_texts = [doc.text for doc in docs]

    chunks = processor.chunk_Recursive_char(docs_texts)
    print(f"\nCreated {len(chunks)} chunks.")

    # Generate Embeddings
    print("Generating embeddings...")
    vectors = embed_model.encode(chunks, show_progress_bar=True)

    # Prepare Metadata for Qdrant
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    payloads = [
        {"text": chunk, "source": "2405.17247v1.pdf", "chunk_index": i} 
        for i, chunk in enumerate(chunks)
    ]

    # Upsert into vector database
    vector_store.upsert(ids, vectors, payloads)
    print("Upsert completed. Your Collection is ready to use!")
    print("==="*50)
    
else:
    print("==="*50)
    print(f"Collection '{vector_store.collection_name}' already contains {current_count} vectors.")
    print("Skipping Step 2 (Ingestion).")
    print("==="*50)



"""
---------------------------------------------------------
3. Test Similarity Search 
---------------------------------------------------------
"""

# We must embed the query using the SAME model

query = "What was the net change in Apple Inc.'s inventory from the fiscal year ending September 24, 2005, to the fiscal year ending September 30, 2006?"
query_vector = embed_model.encode(query).tolist()
search_results = vector_store.similarity_search(query_vector, top_k=3)


"""
---------------------------------------------------------
4. Run Inference
---------------------------------------------------------
"""
print("==="*100)
answer = llm.generate(query, search_results['contexts'])
print(answer)
