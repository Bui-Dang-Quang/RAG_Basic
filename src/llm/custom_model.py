import lmstudio as lms
class CustomModel:
    def __init__(self, model_name: str = "microsoft_-_phi-3.5-mini-instruct", api_host: str = "127.0.0.1:1234"):
        self.model_name = model_name
        self.api_host = api_host
        self.client = None
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Connecting to: {self.model_name}... ")
            self.client = lms.Client(api_host=self.api_host)
            self.model = self.client.llm.model(self.model_name)
        except Exception as e:
            print(f"Connection Error: {e}")
            print("Tip: Ensure LM Studio is open and 'Start Server' is green.")
        
    def generate(self, user_query: str, context_input) -> str:

        if not self.model:
            return "Error: Model not loaded."
        
        if isinstance(context_input, list):
            context_str = "\n\n".join(context_input)
        else:
            context_str = str(context_input)
        
        system_instruction = f"""
            Answer the question below by first outlining the main points of context relevant to the question,
            then use that outline to generate the final answer. 

            Context: 
            {context_str}

            Question: 
            {user_query}
        
        """
        print("System Instruction: \n")
        print(system_instruction)
        print("==="*50)
        
        try: 
            print("==="*50)
            response = self.model.respond(system_instruction)
            return "Generative Answer: \n" + str(response)
        except Exception as e:
            return f"Generate Error: {e}"

# if __name__ == "__main__":
#     embed_model = SentenceTransformer("BAAI/bge-m3")
#     vector_store = QdrantStore(dim=1024, collection_name="PDFdoc_test")
#     query = "What is the core idea of Contrastive-based VLMs?"

#     # We must embed the query using the SAME model
#     query_vector = embed_model.encode(query).tolist()

#     search_results = vector_store.similarity_search(query_vector, top_k=3)
#     # print(f"Query: {query}\n")
#     # for context in search_results['contexts']:
#     #     print(f"--- Result ---\n{context[:200]}...\n")

#     llm = CustomModel()
#     answer = llm.generate(query, search_results['contexts'])
#     print(answer)
    
    
