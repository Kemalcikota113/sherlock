import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

class SherlockEngine:
    def __init__(self):
        # 1. Setup Embeddings and LLM (Using Google AI Studio / Gemini)
        # Ensure GOOGLE_API_KEY is in your environment variables
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        # 2. Initialize Vector Store (Persistent for Bonus points) 
        # Data will be stored in the /data/chroma_db folder
        self.persist_directory = "./data/chroma_db"
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def process_document(self, file_path: str):
        #Loads, chunks, and indexes a case file
        # 1. Choose loader based on extension
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
            
        documents = loader.load()
        
        # 2. Chunking: Breaking text into manageable pieces for the LLM
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # 3. Add to vector database
        self.vector_db.add_documents(chunks)
        # Chroma writes to disk automatically in newer versions, but we ensure persist
        self.vector_db.persist()

    def query(self, question: str):
        # Searches for evidence and generates a strictly evidence-based answer.
        
        # 1. Retrieve the most relevant snippets from the case files 
        # We fetch the top 4 most relevant chunks
        docs = self.vector_db.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Strict Prompting to prevent hallucinations 
        template = """
        You are Sherlock, a digital assistant for a lead detective. 
        Your task is to answer questions based ONLY on the provided evidence.
        
        Rules:
        - If the answer is not in the evidence, say: "I don't have enough evidence to answer that."
        - Do not make up clues, alibis, or suspects.
        - Be concise and professional.

        Evidence:
        {context}

        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        formatted_prompt = prompt.format(context=context, question=question)
        
        # 3. Generate response
        response = self.llm.invoke(formatted_prompt)
        
        # 4. Return answer and the document sources found
        return {
            "answer": response.content,
            "sources": [doc.metadata.get("source", "Unknown") for doc in docs]
        }