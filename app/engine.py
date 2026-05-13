import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

# attempt to disable geminis safety settings. I did this becauuse when i was testing
# with a sherlock holmes story, the model refused to answer questions absout violent crimes like murders. 
_SAFETY_OFF = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

PROMPT = """You are Sherlock, a digital assistant for a lead detective.
Answer the question using ONLY the evidence below.
If the answer is not in the evidence, reply exactly:
"I don't have enough evidence to answer that."
Do not invent clues, alibis, or suspects.

Evidence:
{context}

Question: {question}

Answer:"""

# main engine for sherlock RAG. has one function per API endpoint: ingest, query, list and clear.
class SherlockEngine:

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.google_api_key,
            transport="rest",
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            google_api_key=settings.google_api_key,
            transport="rest",
            timeout=60,
            safety_settings=_SAFETY_OFF,
        )

        self.vector_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)

    # ingest a document, split it into chunks, and add to the vector database. returns number of chunks ingested.
    def ingest(self, file_path: str, filename: str) -> int:


        loader = PyPDFLoader(file_path) if file_path.lower().endswith(".pdf") else TextLoader(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(loader.load())
        for chunk in chunks:
            chunk.metadata["source"] = filename
        self.vector_db.add_documents(chunks)
        return len(chunks)

    # Retrieve top-k evidence and ask the LLM to answer strictly from it
    def query(self, question: str) -> dict:

        docs = self.vector_db.similarity_search(question, k=4)
        if not docs:
            return {"answer": "I don't have enough evidence to answer that.", "sources": []}

        context = "\n\n".join(d.page_content for d in docs)
        response = self.llm.invoke(PROMPT.format(context=context, question=question))
        return {
            "answer": response.content,
            "sources": [
                {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page"),
                    "snippet": d.page_content[:240].strip(),
                }
                for d in docs
            ],
        }

    # Return the unique source filenames currently indexed
    def list_documents(self) -> list[str]:

        data = self.vector_db.get()
        return sorted({m["source"] for m in data.get("metadatas", []) if m and "source" in m})

    # just wipe the vector DB.
    def clear(self) -> None:
        
        self.vector_db.delete_collection()
        self.vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
