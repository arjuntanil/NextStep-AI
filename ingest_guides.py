# ingest_guides.py
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
JSON_FILE_PATH = "career_guides.json"
FAISS_INDEX_PATH = "guides_index" # <--- New index folder name
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def create_guides_vector_store():
    print(f"Loading career guides from {JSON_FILE_PATH}...")
    documents_to_process = []
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data_array = json.load(f)

        for guide in data_array:
            content = f"Career Path: {guide.get('category_name', '')}\n" \
                      f"Overview: {guide.get('overview', '')}\n" \
                      f"Core Roles: {guide.get('core_roles', '')}\n" \
                      f"Common Responsibilities: {guide.get('responsibilities', '')}\n" \
                      f"Key Skills and Tools: {', '.join(guide.get('key_skills_tools', []))}\n" \
                      f"Learning Path and Certifications: {guide.get('learning_path_certifications', '')}"
            metadata = {"category": guide.get('category_name')}
            documents_to_process.append(Document(page_content=content, metadata=metadata))
            
    except Exception as e:
        print(f"Error processing guides file: {e}")
        return

    print(f"Loaded {len(documents_to_process)} career guides.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents_to_process)
    
    print("Initializing embedding model and creating guide vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"✅ Career guides vector store saved to {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    create_guides_vector_store()