from llama_parse import LlamaParse  # noqa: E402
from llama_index.core import SimpleDirectoryReader, Document
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
        # Extract filename (e.g., "NASDAQ_AAPL_2007")
        # os.path.splitext handles "file.pdf" correctly. 
        # If you have "file.pdf.pkl", calling splitext twice or your specific logic is needed.
        # Assuming input is the raw .pdf path:
        base_name_with_ext = os.path.basename(file_path)
        file_name = os.path.splitext(base_name_with_ext)[0]

        # Define paths
        self.cache_dir = "C:/Users/dangq/OneDrive/Máy tính/USTH/ICT/Internship/RAG Remake/RAG_Basic copy/data/raw/md/"
        self.markdown_path = os.path.join(self.cache_dir, file_name + ".md")

    def load_docs(self):
        # 1. Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # 2. Check if Markdown file exists
        if os.path.exists(self.markdown_path):
            print(f"Loading existing markdown file: {self.markdown_path}")
            with open(self.markdown_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            
            # Return as a LlamaIndex Document
            return [Document(text=markdown_content, metadata={"file_name": os.path.basename(self.file_path)})]

        # 3. If not, parse PDF
        else:
            print("Parsing PDF via LlamaParse...")
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="markdown",
            )

            file_extractor = {".pdf": parser}

            documents = SimpleDirectoryReader(
                input_files=[self.file_path], 
                file_extractor=file_extractor
            ).load_data()

            print(f"Saving parsed data to markdown: {self.markdown_path}")
            
            # Combine text from pages/documents into one string
            full_text = "\n\n".join([doc.text for doc in documents])

            # Save to Markdown file
            with open(self.markdown_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            return documents

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Define the path to a test PDF file
    # Ensure this file actually exists on your disk for the test to work
    test_pdf_path = "C:/Users/dangq/OneDrive/Máy tính/USTH/ICT/Internship/RAG Remake/RAG_Basic copy/data/raw/NASDAQ_AAPL_2007.pdf"
    
    # Check if the test file exists before running
    if os.path.exists(test_pdf_path):
        print(f"Testing PDFLoader with: {test_pdf_path}")
        
        # Initialize Loader
        loader = PDFLoader(test_pdf_path)
        
        # Load Documents (This will trigger LlamaParse if .md doesn't exist, or load .md if it does)
        docs = loader.load_docs()
        
        print("\n--- Test Results ---")
        print(f"Number of documents loaded: {len(docs)}")
        if docs:
            print("Preview of loaded content (first 500 chars):")
            print("-" * 50)
            print(docs[0].text[:500])
            print("-" * 50)
    else:
        print(f"Error: Test file not found at {test_pdf_path}")
