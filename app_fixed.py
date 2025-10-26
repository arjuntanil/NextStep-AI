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
REGISTER_URL = f"{API_BASE_URL}/auth/register"
LOGIN_URL = f"{API_BASE_URL}/auth/manual-login"
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
st.session_state.setdefault('login_error', None)
st.session_state.setdefault('register_error', None)
st.session_state.setdefault('register_success', False)

# Function to validate and set token
def validate_and_set_user_token(token):
    """Validate token and store user info"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(USER_ME_URL, headers=headers, timeout=5)
        
        if response.status_code == 200:
            user_data = response.json()
            
            # Store in session state (persists across reruns)
            st.session_state.token = token
            st.session_state.user_info = user_data
            return True
        else:
            return False
    except Exception as e:
        return False

# Check for token in URL parameters (auto-login from portal)
query_params = st.query_params
if 'token' in query_params:
    token = query_params.get('token')
    # Only validate if we don't have a token or if it's different
    if not st.session_state.token or st.session_state.token != token:
        if validate_and_set_user_token(token):
            # Keep token in URL for persistence across refreshes
            pass  # Don't clear the URL parameter
        else:
            # Invalid token, clear it
            st.query_params.clear()
elif not st.session_state.token:
    # No token in URL and no session token - show login
    pass

# --- Main App UI ---
st.title("✨ NextStepAI: Your Career Navigator")

# --- Conditional UI based on Login State ---
if st.session_state.token:
    st.sidebar.success(f"✅ Logged in as {st.session_state.user_info.get('email', 'user')}")
    
    # Check if user is admin
    user_role = st.session_state.user_info.get('role', 'user')
    is_admin = (user_role == 'admin')
    
    # Admin Panel Toggle
    if is_admin:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👨‍💼 Admin Controls")
        show_admin = st.sidebar.checkbox("📊 Show Admin Dashboard", value=False, key="show_admin_panel")
    else:
        show_admin = False
    
    if st.sidebar.button("🚪 Logout"):
        # Clear session state
        st.session_state.token = None
        st.session_state.user_info = None
        st.session_state.login_error = None
        st.session_state.register_error = None
        st.session_state.register_success = False
        
        # Clear token from URL to prevent auto-login
        st.query_params.clear()
        
        st.sidebar.success("✅ Logged out successfully")
        time.sleep(0.5)
        st.rerun()
    
    # Show Admin Dashboard if admin checkbox is enabled
    if show_admin:
        # Import and display admin dashboard content
        import requests
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        import pandas as pd
        
        st.title("👨‍💼 Admin Dashboard")
        st.markdown("---")
        
        # Fetch admin stats
        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            response = requests.get(f"{API_BASE_URL}/admin/stats", headers=headers, timeout=10)
            
            if response.status_code == 200:
                stats = response.json()
                
                # Overview Metrics
                st.subheader("📊 System Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("👥 Total Users", stats.get('total_users', 0))
                with col2:
                    st.metric("📄 Resume Analyses", stats.get('total_analyses', 0))
                with col3:
                    st.metric("💬 Career Queries", stats.get('total_queries', 0))
                with col4:
                    st.metric("🤖 RAG Queries", stats.get('total_rag_queries', 0))
                
                st.markdown("---")
                
                # Recent Activity
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("New Users (7 days)", stats.get('new_users_7days', 0))
                with col2:
                    st.metric("Analyses (7 days)", stats.get('analyses_7days', 0))
                with col3:
                    st.metric("Queries (7 days)", stats.get('queries_7days', 0))
                
                st.markdown("---")
                
                # User Management
                st.subheader("👥 User Management")
                
                users_response = requests.get(f"{API_BASE_URL}/admin/users?limit=100", headers=headers, timeout=10)
                
                if users_response.status_code == 200:
                    users_data = users_response.json()
                    users = users_data.get('users', [])
                    
                    if users:
                        # Create user table
                        user_df = pd.DataFrame(users)
                        
                        # Display key columns
                        display_columns = ['email', 'full_name', 'role', 'is_active', 'created_at']
                        available_columns = [col for col in display_columns if col in user_df.columns]
                        
                        st.dataframe(
                            user_df[available_columns],
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.caption(f"Total: {len(users)} users")
                    else:
                        st.info("No users found")
                else:
                    st.error(f"Failed to load users: {users_response.status_code}")
                
                st.markdown("---")
                
                # Activity Chart
                st.subheader("📈 Activity Trends")
                
                # Sample chart - you can enhance this with real data
                activity_data = {
                    'Date': pd.date_range(end=datetime.now(), periods=7).strftime('%Y-%m-%d').tolist(),
                    'Users': [5, 8, 12, 10, 15, 18, 20],
                    'Analyses': [10, 15, 20, 18, 25, 30, 35],
                    'Queries': [20, 25, 30, 28, 35, 40, 45]
                }
                
                df = pd.DataFrame(activity_data)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Users'], mode='lines+markers', name='Users'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Analyses'], mode='lines+markers', name='Analyses'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Queries'], mode='lines+markers', name='Queries'))
                
                fig.update_layout(
                    title="7-Day Activity Trends",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # System Info
                st.markdown("---")
                st.subheader("⚙️ System Information")
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.info(f"**Average Match Score:** {stats.get('avg_match_percentage', 0)}%")
                    st.info(f"**7-Day Retention:** {stats.get('retention_7days', 0)}%")
                
                with info_col2:
                    st.info(f"**30-Day Retention:** {stats.get('retention_30days', 0)}%")
                    st.info(f"**Total Queries:** {stats.get('total_queries', 0)}")
                
            else:
                st.error(f"Failed to load admin stats: {response.status_code}")
                st.info("Make sure you're logged in as an admin and the backend is running on port 8000")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API")
            st.info("Please ensure the backend is running: `python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000`")
        except Exception as e:
            st.error(f"Error loading admin dashboard: {str(e)}")
    
    else:
        # Regular user tabs (show only if not viewing admin dashboard)
        # Updated tab names: Resume Analyzer → CV Analyzer, RAG Coach → Resume Analyzer using JD
        tabs = st.tabs(["📄 CV Analyzer", "💬 AI Career Advisor", "🧑‍💼 Resume Analyzer using JD", "🗂️ My History"])
        show_tabs = True
else:
    show_tabs = True
    st.sidebar.info("🔐 **Login or Register to save your history**")
    
    # Login/Register Tabs
    auth_tab1, auth_tab2 = st.sidebar.tabs(["Login", "Register"])
    
    # Login Tab
    with auth_tab1:
        st.markdown("#### 🔑 Login")
        login_email = st.text_input("Email", key="login_email", placeholder="your@email.com")
        login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if login_email and login_password:
                try:
                    response = requests.post(LOGIN_URL, json={
                        "email": login_email,
                        "password": login_password
                    })
                    
                    if response.status_code == 200:
                        token_data = response.json()
                        token = token_data["access_token"]
                        st.session_state.login_error = None
                        
                        # Validate and store token (also stores in localStorage)
                        if validate_and_set_user_token(token):
                            # Add token to URL for persistence across refreshes
                            st.query_params['token'] = token
                            
                            # Success message for all users (admin and regular)
                            if st.session_state.user_info.get('role') == 'admin':
                                st.success("✅ Admin login successful! Check the sidebar for Admin Controls.")
                            else:
                                st.success("✅ Login successful!")
                            
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Failed to validate user credentials")
                    else:
                        error_detail = response.json().get('detail', 'Invalid email or password')
                        st.session_state.login_error = error_detail
                        st.error(f"❌ {error_detail}")
                except requests.exceptions.RequestException as e:
                    st.session_state.login_error = f"Connection error: {e}"
                    st.error(f"❌ Connection error: {e}")
            else:
                st.warning("⚠️ Please enter both email and password")
        
        if st.session_state.login_error:
            st.error(f"❌ {st.session_state.login_error}")
    
    # Register Tab
    with auth_tab2:
        st.markdown("#### 📝 Register")
        register_name = st.text_input("Full Name", key="register_name", placeholder="John Doe")
        register_email = st.text_input("Email", key="register_email", placeholder="your@email.com")
        register_password = st.text_input("Password", type="password", key="register_password", placeholder="Create a password")
        register_confirm = st.text_input("Confirm Password", type="password", key="register_confirm", placeholder="Confirm password")
        
        if st.button("Register", type="primary", use_container_width=True):
            if register_name and register_email and register_password and register_confirm:
                if register_password != register_confirm:
                    st.error("❌ Passwords do not match!")
                elif len(register_password) < 6:
                    st.error("❌ Password must be at least 6 characters!")
                else:
                    try:
                        response = requests.post(REGISTER_URL, json={
                            "email": register_email,
                            "full_name": register_name,
                            "password": register_password
                        })
                        
                        if response.status_code == 200:
                            token_data = response.json()
                            st.session_state.token = token_data["access_token"]
                            st.session_state.register_error = None
                            st.session_state.register_success = True
                            
                            # Fetch user info
                            headers = {"Authorization": f"Bearer {st.session_state.token}"}
                            user_response = requests.get(USER_ME_URL, headers=headers)
                            if user_response.status_code == 200:
                                st.session_state.user_info = user_response.json()
                            
                            st.success("✅ Registration successful! Welcome!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            error_detail = response.json().get('detail', 'Registration failed')
                            st.session_state.register_error = error_detail
                            st.error(f"❌ {error_detail}")
                    except requests.exceptions.RequestException as e:
                        st.session_state.register_error = f"Connection error: {e}"
                        st.error(f"❌ Connection error: {e}")
            else:
                st.warning("⚠️ Please fill in all fields")
        
        if st.session_state.register_error:
            st.error(f"❌ {st.session_state.register_error}")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Without login, your analysis results won't be saved.")
    # Updated tab names: Resume Analyzer → CV Analyzer, RAG Coach → Resume Analyzer using JD
    tabs = st.tabs(["📄 CV Analyzer", "💬 AI Career Advisor", "🧑‍💼 Resume Analyzer using JD"])

# Only render tab content if we're showing tabs (not admin dashboard)
if 'show_tabs' in locals() and show_tabs:
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

    # --- Tab Definitions ---

    # --- Tab 1: CV Analyzer ---
    with tabs[0]: 
        st.header("Analyze Your Existing CV")
        st.markdown("Login to automatically save your results." if not st.session_state.token else "Your results will be saved to your history.")
        resume_file = st.file_uploader("Upload Your CV", type=["pdf", "docx"], key="analyzer_uploader")

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
    st.header("🤖 AI Career Advisor")
    
    st.markdown("Ask about career paths to receive AI-generated advice and see live job postings.")
    
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

    # --- Tab 3: Resume Analyzer using JD ---
with tabs[2]:
    st.header("🧑‍💼 Resume Analyzer using JD")
    st.markdown("Upload your resume and job description PDFs to get personalized career advice and gap analysis.")
    
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

                                            # Render similarity visualizations if available
                                            sim = processed_data.get("result", {}).get("similarity_metrics", {})
                                            if sim:
                                                try:
                                                    st.markdown("---")
                                                    st.subheader("🔗 Resume ↔️ Job Description Match")
                                                    col_a, col_b, col_c = st.columns([1,2,1])
                                                    with col_a:
                                                        st.metric("Overall Match (%)", f"{sim.get('overall_score_pct', 0)}%")
                                                        st.caption(f"Matched: {sim.get('num_matched_skills',0)} | JD: {sim.get('num_job_skills',0)} | Resume: {sim.get('num_resume_skills',0)}")
                                                        st.progress(int(sim.get('overall_score_pct', 0)))
                                                    with col_b:
                                                        import plotly.express as px
                                                        matched = len(sim.get('matched_job_skills', []))
                                                        missing = len(sim.get('missing_job_skills', []))
                                                        resume_only = max(0, sim.get('num_resume_skills', 0) - matched)
                                                        labels = ['Matched (JD)', 'Missing (JD)', 'Resume-only']
                                                        values = [matched, missing, resume_only]
                                                        fig = px.pie(names=labels, values=values, hole=0.45, color=labels,
                                                                     color_discrete_map={'Matched (JD)':'#2ca02c','Missing (JD)':'#d62728','Resume-only':'#1f77b4'})
                                                        fig.update_traces(textposition='inside', textinfo='label+percent')
                                                        fig.update_layout(margin={'t':10,'b':10,'l':10,'r':10}, height=280)
                                                        st.plotly_chart(fig, use_container_width=True)
                                                    with col_c:
                                                        per = sim.get('per_skill_scores', [])
                                                        if per:
                                                            import pandas as pd
                                                            df = pd.DataFrame(per)
                                                            df = df.sort_values('score', ascending=False).head(8)
                                                            bar = px.bar(df, x='score', y='resume_skill', orientation='h', labels={'score':'Match score','resume_skill':'Resume Skill'}, color='score', color_continuous_scale='Blues')
                                                            bar.update_layout(margin={'t':10,'b':10,'l':10,'r':10}, height=280)
                                                            st.plotly_chart(bar, use_container_width=True)
                                                        else:
                                                            st.info("No per-skill scores available")

                                                except Exception as e:
                                                    st.warning(f"Could not render similarity visualizations: {e}")

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
                        st.success("📝 Answer:")
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
            
            # Resume Analyzer using JD History
            st.subheader("🧑‍💼 Past Resume Analysis (with JD)")
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
                st.info("No resume analysis queries found.")