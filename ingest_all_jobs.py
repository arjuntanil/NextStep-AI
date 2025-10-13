# ingest_all_jobs.py

import json
import os
import time
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
# Use the correct source file name provided by you.
SOURCE_JSON_FILE = "monster_india.json" 
FAISS_INDEX_PATH = "jobs_index" # Vector store for all individual jobs
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def create_all_jobs_vector_store():
    print(f"Loading all job postings from {SOURCE_JSON_FILE}...")
    documents_to_process = []
    try:
        # Use standard json.load() to read the file as a list of dictionaries (JSON array)
        with open(SOURCE_JSON_FILE, 'r', encoding='utf-8') as f:
            data_array = json.load(f)

        for job in data_array:
            # Combine relevant fields into a single text content for embedding.
            # Use .get() for safety in case some keys are missing in certain job postings.
            content = f"Job Title: {job.get('title', '')}\n" \
                      f"Company: {job.get('company_name', '')}\n" \
                      f"Industry: {job.get('industry', '')}\n" \
                      f"Location: {job.get('address_locality', 'N/A')}, {job.get('address_region', 'N/A')}\n" \
                      f"Experience Required: {job.get('experience', 'N/A')}\n" \
                      f"Job Description: {job.get('description', '')}"
            
            metadata = {
                "source_id": str(job.get('_id')),
                "title": job.get('title'),
                "company": job.get('company_name'),
                "location": f"{job.get('address_locality', '')}, {job.get('address_region', '')}"
            }
            documents_to_process.append(Document(page_content=content, metadata=metadata))

    except FileNotFoundError:
        print(f"❌ Error: Source file not found at {SOURCE_JSON_FILE}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON from {SOURCE_JSON_FILE}. Check file integrity. Error: {e}")
        return
    except Exception as e:
        print(f"An error occurred during file reading: {e}")
        return

    print(f"Loaded {len(documents_to_process)} job postings.")

    # 2. Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents_to_process)
    print(f"Split job postings into {len(chunks)} text chunks.")

    # 3. Create embeddings using a local model
    print(f"Initializing local embedding model: {LOCAL_EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    
    # 4. Create FAISS vector store from chunks
    print("Creating vector store for all jobs... This might take several minutes.")
    start_time = time.time()
    vector_store = FAISS.from_documents(chunks, embeddings)
    end_time = time.time()
    print(f"Vector store creation took {end_time - start_time:.2f} seconds.")

    # 5. Save the completed vector store locally
    if vector_store:
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"✅ All jobs vector store saved to {FAISS_INDEX_PATH}")
    else:
        print("❌ No data processed, vector store not saved.")

if __name__ == "__main__":
    create_all_jobs_vector_store()