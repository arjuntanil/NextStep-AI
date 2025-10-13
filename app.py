# app.py
import streamlit as st
import requests
import json
import io
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
    st.sidebar.success(f"Logged in as {st.session_state.user_info.get('email', 'user')}")
    if st.sidebar.button("Logout"):
        st.session_state.token = None; st.session_state.user_info = None; st.query_params.clear()
        st.rerun()
    # MODIFICATION: Removed "ATS Resume Builder" from tabs list
    tabs = st.tabs(["📄 Resume Analyzer", "💬 AI Career Advisor", "🗂️ My History"])
else:
    st.sidebar.info("Login to save and view your history.")
    # MODIFICATION: Removed "ATS Resume Builder" from tabs list
    tabs = st.tabs(["📄 Resume Analyzer", "💬 AI Career Advisor"])

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
            use_finetuned = st.checkbox("Use Fine-tuned Model", value=True, help="Use the specialized career advisor model")
            max_length = st.slider("Response Length", 100, 300, 200)
        with col2:
            temperature = st.slider("Creativity", 0.1, 1.0, 0.7, help="Higher values = more creative responses")
    
    user_query = st.text_input("💬 Your Career Question:", placeholder="Example: Tell me about a career in Data Science", key="unified_query")
    
    if user_query:
        with st.spinner("🧠 Generating personalized career advice..."):
            try:
                if use_finetuned:
                    # Use the dedicated fine-tuned endpoint
                    payload = {
                        "text": user_query,
                        "max_length": max_length,
                        "temperature": temperature
                    }
                    response = requests.post(CAREER_ADVICE_AI_URL, json=payload, timeout=90)
                    
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

# --- Tab 3: My History (Index updated from 3 to 2) ---
if len(tabs) == 3: 
    with tabs[2]:
        st.header("Your Saved History")
        if st.button("🔄 Refresh History"):
            try:
                analyses_res = requests.get(HISTORY_ANALYSES_URL, headers=headers)
                queries_res = requests.get(HISTORY_QUERIES_URL, headers=headers)
                if analyses_res.status_code == 200 and queries_res.status_code == 200:
                    st.session_state.history = {"analyses": analyses_res.json(), "queries": queries_res.json()}
                    st.success("History updated!")
                else:
                    st.error("Could not fetch history. Your session may have expired.")
            except Exception as e:
                st.error(f"Error fetching history: {e}")
        
        if st.session_state.history:
            st.subheader("Past Resume Analyses")
            analyses = st.session_state.history.get("analyses", [])
            if analyses:
                for item in analyses:
                    with st.expander(f"Analysis for **{item['recommended_job_title']}** (Match: {item['match_percentage']}%)"):
                        skills = json.loads(item.get('skills_to_add', '[]')) # Use .get() for safety
                        st.write("Skills to add:", ", ".join(f"`{s}`" for s in skills))
            else:
                st.write("No resume analyses found.")
            st.subheader("Past Career Queries")
            queries = st.session_state.history.get("queries", [])
            if queries:
                for item in queries:
                    st.info(f"You asked about **'{item['user_query_text']}'** ➡️ Matched to **{item['matched_job_group']}**.")
            else:
                st.write("No career queries found.")