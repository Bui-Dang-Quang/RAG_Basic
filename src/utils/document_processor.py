from typing import Optional
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import uuid


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, model_name: str = "BAAI/bge-m3"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.separators = ["\n\n", "\n", " ", ""]
    
    def chunk_document(self, text: str) -> list[str]:
        """Chunk a document into smaller pieces based on token count"""
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        chunks = []
        start = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks

    def preprocess_documents(self, documents: list[str]) -> list[str]:
        """Preprocess a list of documents by chunking them"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def chunk_Recursive_char(self, texts: str) -> list[str]:
        """Chunk a document using RecursiveCharacterTextSplitter"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap, # Use the value from init
            length_function=self.count_tokens,
            separators=self.separators, # Standard recursive separators
            keep_separator=False,
            strip_whitespace=True
        )

        # Handle Input Flexibility (String vs List)
        if isinstance(texts, str):
            input_list = [texts] # Wrap single string in list
        else:
            input_list = texts

        # 3. Process Statelessly 
        final_chunks = []
        
        for text in input_list:
            chunks = text_splitter.split_text(text)
            final_chunks.extend(chunks)
            
        return final_chunks

    def chunk_by_markdown(self, markdown_txt: str):
        headers_to_split_on = [
            ("#", "Title"),
            ("##", "Section"),
            ("###", "Subsection"),
        ]
        
        # 2. Split
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_txt)

        return md_header_splits


    def count_tokens(self, text: str) -> int:
        """Helper to count tokens in a string"""
        return len(self.tokenizer.encode(text))
    
# if __name__ == "__main__":

    # """
    # ---------------------------------------------------------
    # 1. Load and Chunk Documents
    # ---------------------------------------------------------
    # """
    # loader = PDFLoader("C:\\Users\\dangq\\OneDrive\\Máy tính\\USTH\\ICT\\Internship\\RAG Remake\\RAG\\data\\raw\\2405.17247v1.pdf")
    # docs = loader.load_docs()
    # # print(docs[10].text)

    # docs_texts = [doc.text for doc in docs]
    # processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    # chunks = processor.chunk_Recursive_char(docs_texts)
    # print("====="*50)
    # print(f"\nCreated {len(chunks)} chunks.")

    # # Uncomment to see some sample chunks
    # # for i, chunk in enumerate(chunks[5:10]):  # Print first 5 chunks
    # #     print(f"Chunk {i+1} (Length: {processor.count_tokens(chunk)} tokens): {chunk}\n")
    

    # """
    # ---------------------------------------------------------
    # 2. Generate Embeddings (The missing step)
    # ---------------------------------------------------------
    # """
    # # embed_model = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    # embed_model = SentenceTransformer("BAAI/bge-m3")
    # vectors = embed_model.encode(chunks, show_progress_bar=True)


    # """
    # ---------------------------------------------------------
    # 3. Prepare Data for Qdrant
    # ---------------------------------------------------------
    # """
    # ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    # payloads = [
    #     {"text": chunk, "source": "2405.17247v1.pdf", "chunk_index": i} 
    #     for i, chunk in enumerate(chunks)
    # ]


    # """
    # ---------------------------------------------------------
    # 4. Upsert to Vector Database
    # ---------------------------------------------------------
    # """
    # vector_store = QdrantStore(dim=1024, collection_name="PDFdoc_test")
    # vector_store.upsert(ids, vectors, payloads)
    # print("Upsert completed.")

    # """
    # ---------------------------------------------------------
    # 5. Test Search Functionality
    # ---------------------------------------------------------
    # """
    # print("====="*50)
    # print("Testing search functionality...")
    # query = "What is the core idea of Contrastive-based VLMs?"

    # # We must embed the query using the SAME model
    # query_vector = embed_model.encode(query).tolist()

    # search_results = vector_store.similarity_search(query_vector, top_k=3)
    # print(f"Query: {query}\n")
    # for context in search_results['contexts']:
    #     print(f"--- Result ---\n{context[:200]}...\n")
