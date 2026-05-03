import os
import sys
import warnings
from dotenv import load_dotenv

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Force Python to recognize 'src'
sys.path.append(os.getcwd())

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def explain_with_rag(predictions, store_path=None):
    """
    Generate a RAG-grounded health advisory from CNN pollutant predictions.

    Parameters
    ----------
    predictions : dict
        Mapping of pollutant name -> real-world value, e.g.
        {"PM2.5": 167.5, "PM10": 197.0, ..., "AQI": 186.4}
    store_path : str or None
        Absolute path to the FAISS vectorstore directory.
        Defaults to the relative path "artifacts/vectorstore/" for
        backward-compatible standalone use.

    Returns
    -------
    tuple[str, list[str]]
        (advisory_text, list_of_source_filenames)
    """
    resolved_path = store_path or "artifacts/vectorstore/"

    # 1. Load the same local embedding model we used for ingestion
    # Force CPU: the tiny MiniLM model doesn't need GPU, and avoids
    # CUDA cutlass kernel errors on some RTX configurations.
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # 2. Load the local FAISS database
    vectorstore = FAISS.load_local(
        resolved_path, embeddings, allow_dangerous_deserialization=True
    )

    # 3. Create a search query based on the worst pollutant
    worst_pollutant = max(predictions, key=predictions.get)
    query = (
        f"Health effects and safety guidelines for high levels of "
        f"{worst_pollutant} and AQI {predictions['AQI']}"
    )

    # 4. Retrieve the top 3 relevant chunks from your PDFs
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n---\n".join([d.page_content for d in docs])
    sources = sorted(
        {os.path.basename(d.metadata.get("source", "Unknown")) for d in docs}
    )

    # 5. Build the RAG-augmented prompt
    rag_prompt = f"""
    You are the AirSight AI Assistant. You just analyzed an image and predicted these levels:
    {predictions}

    I have retrieved the following technical context from official air quality guidelines:
    {context}

    Using ONLY the metrics above and the provided technical context, write a professional
    health advisory. Mention the specific pollutants and explain the risks based
    on the retrieved guidelines. Keep it concise but authoritative.
    """

    # 6. Generate the final response using Gemini 2.5 Flash
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(rag_prompt)
        return response.text, sources
    except Exception as e:
        return f"RAG Explainer Error: {str(e)}", []

if __name__ == "__main__":
    # Test with the same "Unhealthy" numbers as before
    sample_preds = {
        "PM2.5": 167.50, "PM10": 197.75, "O3": 68.12, 
        "CO": 37.38, "SO2": 8.66, "NO2": 57.47, "AQI": 186.38
    }
    
    print("--- Running AirSight RAG Retrieval ---")
    advisory, sources = explain_with_rag(sample_preds)
    print(advisory)
    print("\n📚 Sources consulted:")
    for s in sources:
        print(f"  - {s}")