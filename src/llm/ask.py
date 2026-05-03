import os
import argparse
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def ask_airsight(question):
    STORE_PATH = "artifacts/vectorstore/"
    
    print("🔍 Searching AirSight knowledge base...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve top 4 chunks
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n---\n".join([d.page_content for d in docs])
    
    # Get the source PDF names (cleaning up the paths)
    sources = set([os.path.basename(d.metadata.get('source', 'Unknown')) for d in docs])
    
    prompt = f"""
    You are the AirSight AI Assistant. Answer the user's question using ONLY the provided context from official environmental guidelines.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer clearly and professionally. Do not hallucinate outside information.
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        print("\n📋 Answer (grounded in authoritative sources):")
        print("-" * 50)
        print(response.text)
        print("\n📚 Sources consulted:")
        for source in sources:
            print(f"  - {source}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirSight Standalone Q&A")
    parser.add_argument("--question", type=str, required=True, help="Your air quality question")
    args = parser.parse_args()
    
    ask_airsight(args.question)