import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def ingest_docs():
    DOC_PATH = "src/rag/documents/"
    STORE_PATH = "artifacts/vectorstore/"
    
    print(f"--- Loading PDFs from {DOC_PATH} ---")
    loader = DirectoryLoader(DOC_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("❌ Error: No PDFs found!")
        return
        
    print(f"--- Splitting {len(documents)} pages into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    print(f"--- Processing {len(texts)} chunks locally via HuggingFace... ---")
    print("    (Downloading the tiny 90MB model on first run. This is completely free and has NO rate limits!)")
    
    # Use the industry standard local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Process everything at once! No loops, no API crashes.
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    print(f"\n--- Saving Vector Store to {STORE_PATH} ---")
    vectorstore.save_local(STORE_PATH)
    print("✅ Ingestion Complete!")

if __name__ == "__main__":
    ingest_docs()