# app.py
import streamlit as st
import requests
import json
import io
import time
from docx import Document
from docx.shared import Inches
import graphviz 

# --- 1. API Endpoint Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
ANALYZE_API_URL = f"{API_BASE_URL}/analyze_resume/"
QUERY_API_URL = f"{API_BASE_URL}/query-career-path/"
CAREER_ADVICE_AI_URL = f"{API_BASE_URL}/career-advice-ai"
MODEL_STATUS_URL = f"{API_BASE_URL}/model-status"
HISTORY_ANALYSES_URL = f"{API_BASE_URL}/history/analyses"
HISTORY_QUERIES_URL = f"{API_BASE_URL}/history/queries"
USER_ME_URL = f"{API_BASE_URL}/users/me"
LOGIN_URL = f"{API_BASE_URL}/auth/login"
RAG_COACH_QUERY_URL = f"{API_BASE_URL}/rag-coach/query"
RAG_COACH_UPLOAD_URL = f"{API_BASE_URL}/rag-coach/upload"
RAG_COACH_STATUS_URL = f"{API_BASE_URL}/rag-coach/status"
BACKEND_URL = API_BASE_URL

st.set_page_config(page_title="NextStepAI - Career Navigator", layout="wide")

# --- Helper Function (for potential future use, a bit orphaned now) ---
def create_ats_resume_docx(data):
    document = Document()
    # ... (code for docx creation) ...
    return io.BytesIO() # Simplified return for placeholder

# --- Session State Initialization ---
st.session_state.setdefault('token', None)
st.session_state.setdefault('user_info', None)
st.session_state.setdefault('analysis_data', None)
st.session_state.setdefault('history', None)

# --- Seamless Login Handler ---
if 'token' in st.query_params and st.session_state.token is None:
    st.session_state.token = st.query_params['token']
    st.query_params.clear()
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        response = requests.get(USER_ME_URL, headers=headers)
        if response.status_code == 200:
            st.session_state.user_info = response.json()
        else:
            st.session_state.token = None
    except requests.exceptions.RequestException:
        st.session_state.token = None
    st.rerun()

# --- Main App UI ---
st.title("✨ NextStepAI: Your Career Navigator")

# --- Conditional UI based on Login State ---
if st.session_state.token:
    st.sidebar.success(f"✅ Logged in as {st.session_state.user_info.get('email', 'user')}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.token = None; st.session_state.user_info = None; st.query_params.clear()
        st.rerun()
    # MODIFICATION: Removed "ATS Resume Builder" from tabs list
    tabs = st.tabs(["📄 Resume Analyzer", "💬 AI Career Advisor", "🧑‍💼 RAG Coach", "🗂️ My History"])
else:
    st.sidebar.info("🔐 **Login to save and view your history**")
    if st.sidebar.button("🔑 Login with Google", type="primary"):
        # Redirect to backend OAuth endpoint
        st.markdown(f'<meta http-equiv="refresh" content="0;url={LOGIN_URL}">', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("Without login, your analysis results won't be saved.")
    # MODIFICATION: Removed "ATS Resume Builder" from tabs list
    tabs = st.tabs(["📄 Resume Analyzer", "💬 AI Career Advisor", "🧑‍💼 RAG Coach"])

headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

# --- Tab Definitions ---

# --- Tab 1: Resume Analyzer ---
with tabs[0]: 
    st.header("Analyze Your Existing Resume")
    st.markdown("Login to automatically save your results." if not st.session_state.token else "Your results will be saved to your history.")
    resume_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx"], key="analyzer_uploader")

    if resume_file:
        with st.spinner("Analyzing resume, generating roadmap, and finding jobs..."):
            files = {"file": (resume_file.name, resume_file.getvalue(), resume_file.type)}
            try:
                response = requests.post(ANALYZE_API_URL, files=files, headers=headers, timeout=120) # Increased timeout for LLM call
                if response.status_code == 200:
                    st.session_state.analysis_data = response.json()
                    st.success("Analysis complete!")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {e}")

    if st.session_state.analysis_data:
        data = st.session_state.analysis_data
        recommended_job = data.get("recommended_job_title", "N/A")
        match_percentage = data.get("match_percentage", 0)
        skills_to_add_with_links = data.get("missing_skills_with_links", [])
        live_jobs = data.get("live_jobs", [])
        layout_feedback = data.get("layout_feedback")

        st.header(f"🎯 Recommended Role: {recommended_job}")
        st.metric(label="Your Skill Match Score", value=f"{match_percentage:.1f}%")
        st.progress(int(match_percentage))
        st.markdown("---")

        # Roadmap Visualization
        st.subheader("Your Personalized Roadmap")
        roadmap_chart = graphviz.Digraph()
        roadmap_chart.attr(rankdir='LR') 
        roadmap_chart.node("current", "Your Current Profile", shape="box", style="filled", fillcolor="#D6EAF8")
        roadmap_chart.node("target", f"Target Role:\n{recommended_job}", shape="box", style="filled", fillcolor="#D5F5E3")
        
        if skills_to_add_with_links:
            roadmap_chart.node("learn", "Learning Path\n(Missing Skills)", shape="diamond", style="filled", fillcolor="#FEF9E7")
            roadmap_chart.edge("current", "learn", label="Upskill")
            for item in skills_to_add_with_links[:3]: 
                roadmap_chart.edge("learn", f"skill_{item['skill_name']}", label=item['skill_name'])
                roadmap_chart.node(f"skill_{item['skill_name']}", item['skill_name'], style="dashed")
            roadmap_chart.edge("learn", "target")
        else:
             roadmap_chart.edge("current", "target", label="Direct Match!")
        st.graphviz_chart(roadmap_chart)
        st.markdown("---")

        # Main content columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your Learning Plan")
            if skills_to_add_with_links:
                st.warning(f"Focus on these {len(skills_to_add_with_links)} skills to match your target role:")
                for item in skills_to_add_with_links:
                    st.markdown(f"- **{item['skill_name']}**: [Watch Tutorial]({item['youtube_link']})")
            else:
                st.success("**Congratulations! Your skillset strongly matches the requirements!**")

            with st.expander("View Your Skills vs. Required Skills"):
                st.subheader("✅ Your Skills")
                st.write(", ".join(f"`{s}`" for s in data.get("resume_skills", [])))
                st.subheader(f"🛠️ Top Skills for a {recommended_job}")
                st.write(", ".join(f"`{s}`" for s in data.get("required_skills", [])))

        with col2:
            st.subheader(f"Live Job Postings for {recommended_job}")
            if live_jobs:
                for job in live_jobs:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
                            <strong><a href="{job['link']}" target="_blank">{job['title']}</a></strong><br>
                            <span>{job['company']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No live job postings found for this role at the moment.")
        
        # --- NEW POSITION: Generative Layout Feedback at the bottom ---
        st.markdown("---")
        if layout_feedback:
            with st.container(border=True):
                st.subheader("💡 AI Feedback on Resume Layout")
                st.markdown(layout_feedback)

# --- Tab 2: AI Career Advisor ---
with tabs[1]: 
    st.header("🤖 Fine-tuned AI Career Advisor")
    
    # Model Status Check
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Powered by Ai_career_Advisor**")
        with col2:
            if st.button("🔍 Check Status"):
                try:
                    status_response = requests.get(MODEL_STATUS_URL, timeout=10)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        finetuned_status = status_data.get("finetuned_career_advisor", {})
                        if finetuned_status.get("loaded", False):
                            st.success(f"✅ Ready ({finetuned_status.get('device', 'unknown')})")
                        else:
                            st.warning("⚠️ Not loaded")
                    else:
                        st.error("❌ Status check failed")
                except:
                    st.error("❌ Cannot connect")
    
    st.markdown("Ask about career paths to receive AI-generated advice from our fine-tuned model and see live job postings.")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            use_finetuned = st.checkbox("Use Fine-tuned Model", value=True, help="Ultra-fast specialized model (5-15 seconds)")
            max_length = st.slider("Response Length", 50, 120, 80, help="⚡ 50-60=5-8s | 70-80=8-12s | 100-120=15-20s")
        with col2:
            temperature = st.slider("Creativity", 0.1, 1.0, 0.5, help="Lower = much faster (recommended: 0.5)")
    
    user_query = st.text_input("💬 Your Career Question:", placeholder="Example: Tell me about a career in Data Science", key="unified_query")
    
    if user_query:
        # Show different spinner messages based on model selection
        spinner_msg = "⚡ EXTREME SPEED: Generating in 5-15 seconds..." if use_finetuned else "🧠 Generating career advice with RAG system..."
        
        with st.spinner(spinner_msg):
            try:
                if use_finetuned:
                    # Use the dedicated fine-tuned endpoint
                    payload = {
                        "text": user_query,
                        "max_length": max_length,
                        "temperature": 0.5  # Force low temp for speed
                    }
                    response = requests.post(CAREER_ADVICE_AI_URL, json=payload, timeout=45)
                    
                    if response.status_code == 200:
                        data = response.json()
                        advice = data.get("advice")
                        model_used = data.get("model_used")
                        confidence = data.get("confidence")
                        live_jobs = data.get("live_jobs", [])
                        matched_group = data.get("matched_job_group", "relevant roles")

                        # Display model info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", model_used)
                        with col2:
                            st.metric("Confidence", confidence)
                        with col3:
                            st.metric("Response Length", len(advice.split()) if advice else 0)

                        st.subheader("🎯 AI Career Coach Analysis:")
                        if advice: 
                            st.markdown(f"**Question:** {user_query}")
                            st.markdown(f"**Advice:** {advice}")
                        
                        st.divider()
                        st.subheader(f"📋 Live Job Postings for {matched_group}")
                        if live_jobs:
                            for job in live_jobs:
                                st.markdown(f"**[{job['title']}]({job['link']})** at {job['company']}", unsafe_allow_html=True)
                        else:
                            st.info("No live job postings found for this query.")
                    else:
                        # Fallback to original endpoint
                        st.warning("Fine-tuned model unavailable, falling back to RAG system...")
                        payload = {"text": user_query}
                        response = requests.post(QUERY_API_URL, json=payload, headers=headers, timeout=90)
                        
                        if response.status_code == 200:
                            data = response.json()
                            advice = data.get("generative_advice")
                            live_jobs = data.get("live_jobs", [])
                            matched_group = data.get("matched_job_group", "relevant roles")

                            st.info("📡 Using RAG-based Career Advisor")
                            st.subheader("AI Career Coach Analysis:")
                            if advice: st.markdown(advice)
                            st.divider()
                            st.subheader(f"Live Job Postings for {matched_group}")
                            if live_jobs:
                                for job in live_jobs:
                                    st.markdown(f"**[{job['title']}]({job['link']})** at {job['company']}", unsafe_allow_html=True)
                            else:
                                st.info("No live job postings found for this query.")
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                else:
                    # Use original RAG endpoint
                    payload = {"text": user_query}
                    response = requests.post(QUERY_API_URL, json=payload, headers=headers, timeout=90)
                    
                    if response.status_code == 200:
                        data = response.json()
                        advice = data.get("generative_advice")
                        live_jobs = data.get("live_jobs", [])
                        matched_group = data.get("matched_job_group", "relevant roles")

                        st.info("📡 Using RAG-based Career Advisor")
                        st.subheader("AI Career Coach Analysis:")
                        if advice: st.markdown(advice)
                        st.divider()
                        st.subheader(f"Live Job Postings for {matched_group}")
                        if live_jobs:
                            for job in live_jobs:
                                st.markdown(f"**[{job['title']}]({job['link']})** at {job['company']}", unsafe_allow_html=True)
                        else:
                            st.info("No live job postings found for this query.")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                        
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Error: {e}")
                st.markdown("**Troubleshooting:**")
                st.markdown("- Make sure the backend server is running on port 8000")
                st.markdown("- Check your internet connection")
                st.markdown("- Try refreshing the page")

# --- Tab 3: RAG Coach ---
with tabs[2]:
    st.header("🧑‍💼 RAG Coach - PDF-Powered Career Guidance")
    st.markdown("Upload your resume and job description PDFs to get personalized career advice using RAG (Retrieval-Augmented Generation) with Ollama Mistral 7B.")
    
    col1, col2 = st.columns(2)
    with col1:
        resume_pdf = st.file_uploader("📄 Upload Your Resume (PDF)", type=["pdf"], key="rag_resume")
    with col2:
        job_desc_pdf = st.file_uploader("📋 Upload Job Description (PDF)", type=["pdf"], key="rag_job_desc")
    
    # Upload PDFs if provided
    if resume_pdf or job_desc_pdf:
        if st.button("📤 Upload & Analyze Documents"):
            with st.spinner("Uploading documents and analyzing..."):
                files_to_upload = []
                if resume_pdf:
                    files_to_upload.append(("files", (resume_pdf.name, resume_pdf.getvalue(), "application/pdf")))
                if job_desc_pdf:
                    files_to_upload.append(("files", (job_desc_pdf.name, job_desc_pdf.getvalue(), "application/pdf")))
                
                try:
                    # Upload PDFs with automatic processing enabled
                    upload_response = requests.post(
                        RAG_COACH_UPLOAD_URL, 
                        files=files_to_upload,
                        data={"process_resume_job": "true"},
                        timeout=30
                    )
                    if upload_response.status_code == 200:
                        upload_data = upload_response.json()
                        st.success(f"✅ {upload_data['message']}")
                        uploaded_files = upload_data.get('files_uploaded', [])
                        st.info(f"📄 Uploaded: {', '.join(uploaded_files)}")
                        
                        # Show processing status
                        status_placeholder = st.empty()
                        result_placeholder = st.empty()
                        
                        # Poll for processing completion
                        max_wait = 120  # 2 minutes max
                        poll_interval = 2
                        elapsed = 0
                        
                        while elapsed < max_wait:
                            try:
                                status_response = requests.get(RAG_COACH_STATUS_URL, timeout=10)
                                if status_response.status_code == 200:
                                    status = status_response.json()
                                    
                                    # Check if automatic processing is complete
                                    if status.get("processing_ready", False):
                                        status_placeholder.success("✅ Analysis Complete!")
                                        
                                        # Fetch the processed result
                                        result_response = requests.get(f"{BACKEND_URL}/rag-coach/processed-result", timeout=10)
                                        if result_response.status_code == 200:
                                            processed_data = result_response.json()
                                            formatted_output = processed_data.get("result", {}).get("formatted", "")
                                            
                                            # Display the automatic analysis
                                            result_placeholder.markdown("### 🎯 Resume Enhancement Suggestions")
                                            result_placeholder.markdown(formatted_output)
                                            
                                            # Mark that documents are uploaded and processed
                                            st.session_state["rag_documents_uploaded"] = True
                                        break
                                    
                                    # Show current status
                                    if status.get("processing", False):
                                        status_placeholder.info("⏳ Analyzing your documents... (30-60 seconds)")
                                    elif status.get("vector_store_ready", False):
                                        status_placeholder.info("📚 Building knowledge base...")
                                    else:
                                        status_placeholder.info("🔄 Preparing analysis...")
                                    
                            except requests.exceptions.RequestException:
                                pass
                            
                            time.sleep(poll_interval)
                            elapsed += poll_interval
                        
                        if elapsed >= max_wait:
                            status_placeholder.warning("⏱️ Analysis is taking longer than expected. You can still ask questions below.")
                            st.session_state["rag_documents_uploaded"] = True
                    else:
                        error_detail = upload_response.json().get('detail', 'Unknown error')
                        st.error(f"Upload failed: {error_detail}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: {e}")
    
    # Only show query section if documents have been uploaded and processed
    if st.session_state.get("rag_documents_uploaded", False):
        st.markdown("---")
        st.subheader("💬 Ask Follow-up Questions")
        rag_query = st.text_area(
            "Have more questions? Ask away:",
            placeholder="Example: How can I better highlight my leadership experience?",
            height=100,
            key="rag_query_input"
        )
        
        if st.button("🚀 Get Answer") and rag_query:
            with st.spinner("🔍 Searching documents and generating answer..."):
                try:
                    query_payload = {"question": rag_query}
                    query_response = requests.post(RAG_COACH_QUERY_URL, json=query_payload, timeout=90)
                    
                    if query_response.status_code == 200:
                        result = query_response.json()
                        
                        # Display answer
                        st.success("📝 RAG Coach Answer:")
                        st.markdown(result.get("answer", "No answer generated"))
                        
                        # Display retrieved context
                        context_chunks = result.get("context_chunks", [])
                        sources = result.get("sources", [])
                        
                        if context_chunks:
                            with st.expander(f"📚 Retrieved Context ({len(context_chunks)} chunks from {len(sources)} sources)"):
                                for idx, chunk in enumerate(context_chunks, 1):
                                    st.markdown(f"**Chunk {idx}** (from `{chunk.get('source', 'Unknown')}`)")
                                    st.text(chunk.get('content', '')[:500] + "..." if len(chunk.get('content', '')) > 500 else chunk.get('content', ''))
                                    st.markdown("---")
                        else:
                            st.info("No source documents retrieved")
                        
                        # Display source files
                        if sources:
                            st.caption(f"📄 Sources: {', '.join(sources)}")
                        
                    else:
                        error_detail = query_response.json().get('detail', 'Unknown error')
                        st.error(f"Query failed: {error_detail}")
                        if "ollama" in error_detail.lower() and "mistral" in error_detail.lower():
                            st.warning("⚠️ Ollama Mistral model not found. Please run:")
                            st.code("ollama pull mistral:7b-q4", language="bash")
                        elif "vector store" in error_detail.lower() or "no documents" in error_detail.lower():
                            st.info("💡 Tip: Upload some PDFs first to build the knowledge base")
                            
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: {e}")
                    st.markdown("**Troubleshooting:**")
                    st.markdown("- Ensure Ollama is installed and running")
                    st.markdown("- Run: `ollama pull mistral:7b-q4`")
                    st.markdown("- Check if backend is running on port 8000")

# --- Tab 4: My History (Index updated from 2 to 3) ---
if len(tabs) == 4:
    with tabs[3]:
        st.header("📚 Your Saved History")
        if st.button("🔄 Refresh History"):
            try:
                analyses_res = requests.get(HISTORY_ANALYSES_URL, headers=headers)
                queries_res = requests.get(HISTORY_QUERIES_URL, headers=headers)
                rag_queries_res = requests.get(f"{API_BASE_URL}/history/rag-queries", headers=headers)
                
                if analyses_res.status_code == 200 and queries_res.status_code == 200:
                    st.session_state.history = {
                        "analyses": analyses_res.json(), 
                        "queries": queries_res.json(),
                        "rag_queries": rag_queries_res.json() if rag_queries_res.status_code == 200 else []
                    }
                    st.success("✅ History updated!")
                else:
                    st.error("Could not fetch history. Your session may have expired.")
            except Exception as e:
                st.error(f"Error fetching history: {e}")
        
        if st.session_state.history:
            # Resume Analyses History
            st.subheader("📄 Past Resume Analyses")
            analyses = st.session_state.history.get("analyses", [])
            if analyses:
                for item in analyses:
                    with st.expander(f"Analysis for **{item['recommended_job_title']}** (Match: {item['match_percentage']}%)"):
                        skills = json.loads(item.get('skills_to_add', '[]')) # Use .get() for safety
                        st.write("Skills to add:", ", ".join(f"`{s}`" for s in skills) if skills else "No skills needed!")
            else:
                st.info("No resume analyses found.")
            
            st.markdown("---")
            
            # Career Advisor Queries History
            st.subheader("💬 Past AI Career Advisor Queries")
            queries = st.session_state.history.get("queries", [])
            if queries:
                for item in queries:
                    st.info(f"You asked about **'{item['user_query_text']}'** ➡️ Matched to **{item['matched_job_group']}**.")
            else:
                st.info("No career queries found.")
            
            st.markdown("---")
            
            # RAG Coach Queries History
            st.subheader("🧑‍💼 Past RAG Coach Interactions")
            rag_queries = st.session_state.history.get("rag_queries", [])
            if rag_queries:
                for item in rag_queries:
                    with st.expander(f"Q: {item['question'][:80]}..." if len(item['question']) > 80 else f"Q: {item['question']}"):
                        st.markdown("**Answer:**")
                        st.write(item['answer'])
                        sources = json.loads(item.get('sources', '[]'))
                        if sources:
                            st.caption(f"📄 Sources: {', '.join(sources)}")
            else:
                st.info("No RAG Coach queries found.")