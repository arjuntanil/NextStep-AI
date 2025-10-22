"""
RAG Coach - Retrieval-Augmented Generation Career Coaching System
Uses LangChain + FAISS + Ollama Gemma 2B for personalized career advice

This module is completely independent and won't affect existing Resume Analyzer 
or AI Career Advisor functionality.
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:
    # LangChain imports
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Install required packages: pip install langchain langchain-community pypdf2 faiss-cpu")
    raise


class RAGCoachSystem:
    """
    Complete RAG-based Career Coaching System
    
    Features:
    - Load multiple PDF documents (career guides + user uploads)
    - Create FAISS vector embeddings for semantic search
    - Use Ollama Gemma 2B for intelligent response generation
    - Retrieve relevant context before answering queries
    """
    
    def __init__(
        self, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = None,
        vector_store_path: str = "./rag_coach_index"
    ):
        """
        Initialize RAG Coach System
        
        Args:
            embedding_model_name: HuggingFace embedding model for vector creation
            llm_model_name: Ollama model name for text generation (auto-detects available Mistral models)
            vector_store_path: Path to save/load FAISS vector store
        """
        self.embedding_model_name = embedding_model_name
        
        # Auto-detect Mistral model if not specified
        if llm_model_name is None:
            llm_model_name = self._detect_mistral_model()
        
        self.llm_model_name = llm_model_name
        self.vector_store_path = vector_store_path
        
        # Initialize components (set to None until loaded)
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
        logger.info(f"🚀 RAG Coach System initialized with model: {self.llm_model_name}")
    
    def _detect_mistral_model(self) -> str:
        """
        Auto-detect available Mistral model from Ollama
        
        Returns:
            Model name string (prefers instruct variants)
        """
        try:
            import subprocess
            import os
            
            # Try to find ollama executable
            ollama_cmd = 'ollama'
            
            # Check common Windows locations
            possible_paths = [
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama', 'ollama.exe'),
                r'C:\Program Files\Ollama\ollama.exe',
                r'C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe'.format(os.environ.get('USERNAME', '')),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    ollama_cmd = path
                    logger.info(f"✅ Found Ollama at: {path}")
                    break
            
            result = subprocess.run([ollama_cmd, 'list'], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                models = result.stdout.lower()
                
                # Priority order for models (smallest/fastest first for low RAM systems)
                preferred_models = [
                    'tinyllama',                     # 637 MB RAM - BEST for very low memory
                    'mistral:7b-instruct-q2_k',      # 800 MB RAM - Good for low memory
                    'mistral:7b-instruct-q4_k_m',    # 1.6 GB RAM
                    'mistral:7b-instruct',           # 1.6 GB RAM
                    'mistral:latest',
                    'mistral',
                ]
                
                for model in preferred_models:
                    if model in models:
                        logger.info(f"✅ Auto-detected Ollama model: {model}")
                        return model
                
                # Check if any mistral variant exists
                if 'mistral' in models:
                    logger.info(f"✅ Found Mistral model in Ollama")
                    return 'mistral'
                    
                # Check for tinyllama
                if 'tinyllama' in models:
                    logger.info(f"✅ Found TinyLlama model in Ollama")
                    return 'tinyllama'
            
            logger.warning("⚠️  No model found in Ollama, using default: tinyllama")
            return 'tinyllama'
            
        except Exception as e:
            logger.warning(f"⚠️  Could not detect Ollama models: {e}")
            return 'tinyllama'
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings model"""
        if self.embeddings is None:
            logger.info(f"📦 Loading embedding model: {self.embedding_model_name}")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("✅ Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise
    
    def _initialize_llm(self):
        """Initialize Ollama LLM with SPEED OPTIMIZATIONS + CPU-ONLY MODE"""
        if self.llm is None:
            logger.info(f"🤖 Connecting to Ollama with model: {self.llm_model_name}")
            try:
                # Force CPU-only mode to avoid GPU memory errors
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
                
                # ULTRA-CONSERVATIVE settings to prevent crashes
                self.llm = Ollama(
                    model=self.llm_model_name,
                    temperature=0.7,
                    num_ctx=512,      # MINIMAL context to prevent crashes
                    top_k=20,
                    top_p=0.9,
                    num_predict=256,  # Moderate output length
                    repeat_penalty=1.1,
                    # CPU-specific optimizations
                    num_thread=2,     # Reduced threads for stability
                    num_gpu=0         # Force CPU-only mode
                )
                # Quick connection test
                test_response = self.llm.invoke("Hi")
                logger.info("✅ Ollama LLM connected successfully (CPU MODE - STABLE)")
            except Exception as e:
                logger.error("❌ Failed to connect to Ollama")
                logger.error("\n🔧 To fix this issue:")
                logger.error("1. Install Ollama: https://ollama.ai/download")
                logger.error("2. Start Ollama service")
                logger.error(f"3. Pull the model: ollama pull {self.llm_model_name}")
                logger.error("4. Verify: ollama list")
                raise ConnectionError(f"Ollama not available: {e}")
    
    def load_pdf_documents(self, pdf_paths: List[str]) -> List[Document]:
        """
        Load multiple PDF files and extract text content with enhanced metadata
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of LangChain Document objects with text content and doc_type metadata
        """
        logger.info(f"📄 Loading {len(pdf_paths)} PDF documents...")
        all_documents = []
        
        def _detect_document_type(text, filename):
            """Detect if document is resume or job description"""
            text_lower = text.lower()
            fname_lower = filename.lower()
            
            # Filename-based detection (primary)
            if 'resume' in fname_lower or 'cv' in fname_lower or 'profile' in fname_lower:
                return 'RESUME'
            elif 'job' in fname_lower or 'description' in fname_lower or 'jd' in fname_lower or 'position' in fname_lower:
                return 'JOB_DESCRIPTION'
            
            # Content-based detection (fallback)
            resume_indicators = [
                'technical skills', 'education', 'professional experience', 
                'key projects', 'certifications', 'achievements',
                'linkedin', 'github', 'email', 'mobile', 'phone'
            ]
            
            job_indicators = [
                'job title', 'job summary', 'key responsibilities', 
                'required skills', 'qualifications', 'job type',
                'we are looking for', 'ideal candidate', 'apply', 'salary'
            ]
            
            resume_score = sum(1 for ind in resume_indicators if ind in text_lower)
            job_score = sum(1 for ind in job_indicators if ind in text_lower)
            
            if resume_score > job_score:
                return 'RESUME'
            elif job_score > resume_score:
                return 'JOB_DESCRIPTION'
            else:
                return 'UNKNOWN'
        
        for i, pdf_path in enumerate(pdf_paths):
            if not os.path.exists(pdf_path):
                logger.warning(f"⚠️ File not found: {pdf_path}")
                continue
            
            try:
                # Load PDF using LangChain's PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Get full text to detect document type
                full_text = "\n".join([doc.page_content for doc in documents])
                filename = os.path.basename(pdf_path)
                doc_type = _detect_document_type(full_text, filename)
                
                # Add enhanced metadata to identify source AND document type
                for doc in documents:
                    doc.metadata['source'] = filename
                    doc.metadata['doc_type'] = doc_type
                    doc.metadata['doc_index'] = i
                
                all_documents.extend(documents)
                logger.info(f"  ✅ Loaded: {filename} ({len(documents)} pages) [Type: {doc_type}]")
                
            except Exception as e:
                logger.error(f"  ❌ Error loading {pdf_path}: {e}")
                continue
        
        logger.info(f"✅ Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def split_documents(
        self, 
        documents: List[Document], 
        chunk_size: int = 600,  # REDUCED for faster processing (was 1000)
        chunk_overlap: int = 100  # REDUCED overlap (was 200)
    ) -> List[Document]:
        """
        Split documents into smaller chunks - OPTIMIZED FOR SPEED
        
        Args:
            documents: List of Document objects
            chunk_size: Maximum size of each chunk (reduced for speed)
            chunk_overlap: Overlap between consecutive chunks (reduced)
            
        Returns:
            List of split Document chunks
        """
        logger.info("✂️ Splitting documents into chunks (FAST MODE)...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        logger.info(f"✅ Created {len(splits)} text chunks (optimized)")
        
        return splits
    
    def build_vector_store(self, pdf_files: List[str], force_rebuild: bool = False):
        """
        Build FAISS vector store from PDF documents
        
        Args:
            pdf_files: List of paths to PDF files to index
            force_rebuild: If True, rebuild even if vector store exists
        """
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Check if vector store already exists
        if os.path.exists(self.vector_store_path) and not force_rebuild:
            logger.info(f"📚 Loading existing vector store from {self.vector_store_path}")
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Vector store loaded successfully")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing vector store: {e}")
                logger.info("🔄 Building new vector store...")
        
        # Load and process documents
        documents = self.load_pdf_documents(pdf_files)
        
        if not documents:
            raise ValueError("No documents loaded. Please check PDF file paths.")
        
        # Split into chunks
        splits = self.split_documents(documents)
        
        # Create FAISS vector store
        logger.info("🧠 Creating FAISS vector embeddings...")
        try:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Save for future use
            self.vector_store.save_local(self.vector_store_path)
            logger.info(f"💾 Vector store saved to {self.vector_store_path}")
            logger.info("✅ Vector store built successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to build vector store: {e}")
            raise
    
    def setup_qa_chain(self):
        """
        Setup the Retrieval QA chain with LLM and vector store
        """
        # Initialize LLM
        self._initialize_llm()
        
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
        
        # Create retriever - SPEED OPTIMIZED
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}  # REDUCED from 4 to 2 for faster retrieval
        )
        
        # SHORTENED prompt for faster generation
        prompt_template = """Expert AI Career Coach. Use context to answer briefly and actionably.

Context: {context}

Q: {question}

A (concise):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("✅ QA chain configured successfully")
    
    def answer_query(
        self, 
        question: str, 
        show_context: bool = True
    ) -> Dict[str, any]:
        """
        Answer a user query using RAG
        
        Args:
            question: User's question
            show_context: If True, return retrieved context chunks
            
        Returns:
            Dictionary with 'answer', 'source_documents', and 'context' keys
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")
        
        logger.info(f"\n❓ Query: {question}")
        logger.info("🔍 Retrieving relevant context...")
        
        # Detect query intent to prioritize correct document type
        question_lower = question.lower()
        job_keywords = [
            'job', 'position', 'role', 'required', 'qualification', 
            'responsibility', 'candidate', 'need to add', 'skills to add',
            'missing', 'requirements', 'description'
        ]
        resume_keywords = [
            'my', 'i have', 'my skills', 'my experience', 'my resume',
            'what do i know', 'my background', 'my projects'
        ]
        
        query_about_job = any(kw in question_lower for kw in job_keywords)
        query_about_resume = any(kw in question_lower for kw in resume_keywords)
        
        try:
            # Get answer from RAG chain
            result = self.qa_chain.invoke({"query": question})
            
            answer = result['result']
            source_docs = result['source_documents']
            
            # Filter source documents based on query intent
            if query_about_job and not query_about_resume:
                # Prioritize JOB_DESCRIPTION documents
                job_docs = [doc for doc in source_docs if doc.metadata.get('doc_type') == 'JOB_DESCRIPTION']
                if job_docs:
                    source_docs = job_docs
                    logger.info(f"🎯 Filtered to {len(job_docs)} JOB_DESCRIPTION chunks (query about job requirements)")
            elif query_about_resume and not query_about_job:
                # Prioritize RESUME documents
                resume_docs = [doc for doc in source_docs if doc.metadata.get('doc_type') == 'RESUME']
                if resume_docs:
                    source_docs = resume_docs
                    logger.info(f"🎯 Filtered to {len(resume_docs)} RESUME chunks (query about user's background)")
            
            # Extract context from source documents
            context_chunks = []
            for i, doc in enumerate(source_docs, 1):
                context_chunks.append({
                    'chunk_number': i,
                    'text': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    'doc_type': doc.metadata.get('doc_type', 'UNKNOWN')
                })
            
            if show_context:
                logger.info("\n📚 Retrieved Context:")
                for chunk in context_chunks:
                    logger.info(f"\n  [{chunk['chunk_number']}] From: {chunk['source']} (Page {chunk['page']}) [Type: {chunk['doc_type']}]")
                    logger.info(f"  {chunk['text']}")
            
            logger.info(f"\n💬 Final Answer:\n{answer}\n")
            
            return {
                'answer': answer,
                'source_documents': source_docs,
                'context_chunks': context_chunks
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            raise


def main():
    """
    Demo script showing RAG Coach in action
    """
    print("="*70)
    print("🎓 RAG Coach - AI Career Coaching Assistant")
    print("="*70)
    
    # Initialize RAG system
    rag_coach = RAGCoachSystem()
    
    # Example PDF files (adjust paths as needed)
    # These are placeholder paths - replace with actual files
    career_guide_pdfs = [
        "./career_guides/data_analyst.pdf",
        "./career_guides/resume_tips.pdf",
        "./career_guides/interview_tips.pdf",
    ]
    
    # User uploads (example)
    user_uploads = [
        "./uploads/resume.pdf",
        "./uploads/job_description.pdf"
    ]
    
    # Combine all PDFs
    all_pdfs = career_guide_pdfs + user_uploads
    
    # Filter to only existing files for demo
    existing_pdfs = [pdf for pdf in all_pdfs if os.path.exists(pdf)]
    
    if not existing_pdfs:
        print("\n⚠️ No PDF files found. Using demo mode without actual files.")
        print("\n📝 To use RAG Coach:")
        print("1. Create a 'career_guides' folder")
        print("2. Add PDF files: data_analyst.pdf, resume_tips.pdf, interview_tips.pdf")
        print("3. Create an 'uploads' folder for user files")
        return
    
    try:
        # Step 1: Build vector store
        print("\n🔨 Step 1: Building vector store...")
        rag_coach.build_vector_store(existing_pdfs)
        
        # Step 2: Setup QA chain
        print("\n⚙️ Step 2: Setting up QA chain...")
        rag_coach.setup_qa_chain()
        
        # Step 3: Answer sample queries
        print("\n💡 Step 3: Answering career questions...")
        
        sample_questions = [
            "How can I tailor my resume to match a job description?",
            "What should I focus on during a technical interview?",
            "How can I highlight my data analysis skills effectively?"
        ]
        
        for question in sample_questions[:1]:  # Answer first question for demo
            result = rag_coach.answer_query(question, show_context=True)
            print("\n" + "-"*70)
        
        print("\n✅ RAG Coach demo completed successfully!")
        
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n🔧 Setup Instructions:")
        print("1. Install Ollama: https://ollama.ai/download")
        print("2. Start Ollama (it runs as a service)")
        print("3. Pull Gemma model: ollama pull gemma:2b")
        print("4. Verify installation: ollama list")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
