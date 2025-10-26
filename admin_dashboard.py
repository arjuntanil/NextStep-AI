"""
NextStepAI Admin Dashboard
Comprehensive analytics and user management system
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import time
from collections import Counter

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"
ADMIN_LOGIN_URL = f"{API_BASE_URL}/admin/login"
ADMIN_STATS_URL = f"{API_BASE_URL}/admin/stats"
ADMIN_USERS_URL = f"{API_BASE_URL}/admin/users"
ADMIN_USER_DETAILS_URL = f"{API_BASE_URL}/admin/user"

# Page Config
st.set_page_config(
    page_title="NextStepAI Admin Dashboard",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with persistence
if 'admin_token' not in st.session_state:
    st.session_state.admin_token = None
if 'admin_info' not in st.session_state:
    st.session_state.admin_info = None
if 'persist_login' not in st.session_state:
    st.session_state.persist_login = True  # Default: keep login across refreshes

# Function to validate and set token
def validate_and_set_token(token):
    """Validate token and store user info"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        user_me_url = f"{API_BASE_URL}/users/me"
        user_response = requests.get(user_me_url, headers=headers, timeout=5)
        
        if user_response.status_code == 200:
            user_data = user_response.json()
            
            # Verify user is admin
            if user_data.get('role') == 'admin':
                email = user_data.get('email', 'admin@nextstepai.com')
                full_name = user_data.get('full_name', 'Administrator')
                
                # Store in session state (this persists across reruns)
                st.session_state.admin_token = token
                st.session_state.admin_info = {
                    "email": email,
                    "full_name": full_name
                }
                return True
            else:
                st.error("❌ Access Denied: Admin privileges required")
                return False
        else:
            return False
    except Exception as e:
        return False

# Check for token in URL parameters (auto-login from portal)
query_params = st.query_params
if 'token' in query_params:
    token = query_params.get('token')
    # Only validate if we don't have a token or if it's different
    if not st.session_state.admin_token or st.session_state.admin_token != token:
        if validate_and_set_token(token):
            # Keep token in URL for persistence across refreshes
            pass  # Don't clear the URL parameter
        else:
            # Invalid token, clear it
            st.query_params.clear()
elif not st.session_state.admin_token:
    # No token in URL and no session token - show login
    pass

# Helper Functions
def get_headers():
    """Get authorization headers"""
    if st.session_state.admin_token:
        return {"Authorization": f"Bearer {st.session_state.admin_token}"}
    return {}

def fetch_stats():
    """Fetch dashboard statistics from backend"""
    try:
        response = requests.get(ADMIN_STATS_URL, headers=get_headers(), timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch stats: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return None

def fetch_users(page=1, limit=50, search=""):
    """Fetch users list"""
    try:
        params = {"page": page, "limit": limit, "search": search}
        response = requests.get(ADMIN_USERS_URL, headers=get_headers(), params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return None

# Login Page
def show_login():
    st.title("🔐 Admin Login")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container(border=True):
            st.subheader("Administrator Access")
            
            email = st.text_input("📧 Email", placeholder="admin@nextstepai.com")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            if st.button("🚀 Login", use_container_width=True, type="primary"):
                if email and password:
                    with st.spinner("Authenticating..."):
                        try:
                            response = requests.post(
                                ADMIN_LOGIN_URL,
                                json={"email": email, "password": password},
                                timeout=10
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                token = data.get('access_token')
                                
                                # Validate and store token
                                if validate_and_set_token(token):
                                    st.success("✅ Login successful! Redirecting...")
                                    # Redirect to same page with token in URL for persistence
                                    st.query_params['token'] = token
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to validate admin credentials")
                            else:
                                error_detail = response.json().get('detail', 'Invalid credentials')
                                st.error(f"❌ {error_detail}")
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Cannot connect to backend. Make sure backend is running on port 8000.")
                            st.caption("Try: python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload")
                        except requests.exceptions.Timeout:
                            st.error("❌ Connection timeout. Backend is not responding.")
                        except Exception as e:
                            st.error(f"❌ Connection error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter both email and password")
            
            st.markdown("---")
            st.caption("🔒 Secure admin access only")

# Dashboard Page
def show_dashboard():
    # Sidebar
    with st.sidebar:
        st.title("👨‍💼 Admin Panel")
        st.markdown(f"**Welcome, {st.session_state.admin_info.get('full_name', 'Admin')}**")
        st.markdown(f"📧 {st.session_state.admin_info.get('email')}")
        
        st.markdown("---")
        
        page = st.radio(
            "📊 Navigation",
            ["Dashboard", "User Management", "Analytics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            # Clear session state
            st.session_state.admin_token = None
            st.session_state.admin_info = None
            
            # Clear token from URL
            st.query_params.clear()
            
            st.success("✅ Logged out successfully")
            time.sleep(0.5)
            st.rerun()
    
    # Main Content
    if page == "Dashboard":
        show_dashboard_page()
    elif page == "User Management":
        show_user_management()
    elif page == "Analytics":
        show_analytics_page()

# Dashboard Overview
def show_dashboard_page():
    st.title("📊 Dashboard Overview")
    st.markdown("Real-time analytics and system insights")
    
    # Fetch stats
    with st.spinner("Loading dashboard data..."):
        stats = fetch_stats()
    
    if not stats:
        st.error("Failed to load dashboard data")
        return
    
    # === TOP METRICS ROW ===
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="👥 Total Users",
            value=stats.get('total_users', 0),
            delta=f"+{stats.get('new_users_7days', 0)} this week"
        )
    
    with col2:
        st.metric(
            label="🟢 Active Users (30d)",
            value=stats.get('active_users_30days', 0),
            delta=f"{stats.get('active_users_7days', 0)} in 7d"
        )
    
    with col3:
        st.metric(
            label="📄 CV Analyses",
            value=stats.get('total_analyses', 0),
            delta=f"+{stats.get('analyses_7days', 0)} this week"
        )
    
    with col4:
        st.metric(
            label="💬 Career Queries",
            value=stats.get('total_queries', 0),
            delta=f"+{stats.get('queries_7days', 0)} this week"
        )
    
    with col5:
        st.metric(
            label="📊 Avg Match Score",
            value=f"{stats.get('avg_match_percentage', 0):.1f}%",
            delta="Overall"
        )
    
    st.markdown("---")
    
    # === CHARTS ROW 1 ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 User Growth (Last 30 Days)")
        if stats.get('user_growth'):
            df_growth = pd.DataFrame(stats['user_growth'])
            fig = px.line(
                df_growth, 
                x='date', 
                y='count',
                markers=True,
                title="Cumulative User Growth"
            )
            fig.update_traces(line_color='#1f77b4', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No growth data available")
    
    with col2:
        st.markdown("#### 🏆 Top Recommended Jobs")
        if stats.get('top_jobs'):
            df_jobs = pd.DataFrame(stats['top_jobs'])
            fig = px.bar(
                df_jobs,
                x='count',
                y='job',
                orientation='h',
                title="Most Recommended Career Paths",
                color='count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No job data available")
    
    # === CHARTS ROW 2 ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Most Missing Skills")
        if stats.get('top_missing_skills'):
            df_skills = pd.DataFrame(stats['top_missing_skills'])
            fig = px.bar(
                df_skills,
                x='count',
                y='skill',
                orientation='h',
                title="Skills Gap Analysis",
                color='count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No skills data available")
    
    with col2:
        st.markdown("#### 📊 Match Score Distribution")
        if stats.get('match_distribution'):
            # Backend returns array of scores, convert to DataFrame for histogram
            scores = stats['match_distribution']
            fig = px.histogram(
                x=scores,
                nbins=10,
                title="User-Job Match Scores",
                labels={'x': 'Match %', 'y': 'Count'},
                color_discrete_sequence=['#2ecc71']
            )
            fig.update_layout(showlegend=False, xaxis_title="Match %", yaxis_title="Number of Analyses")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No match data available")
    
    # === RECENT ACTIVITY ===
    st.markdown("---")
    st.markdown("#### 🕒 Recent Activity")
    
    if stats.get('recent_activity'):
        # Create a more structured display
        for activity in stats['recent_activity'][:10]:
            col1, col2, col3 = st.columns([2, 5, 2])
            with col1:
                timestamp = activity.get('timestamp', 'N/A')
                # Format timestamp nicely
                try:
                    from dateutil import parser
                    dt = parser.parse(timestamp)
                    formatted_time = dt.strftime('%m/%d %H:%M')
                except:
                    formatted_time = timestamp[:16] if len(timestamp) > 16 else timestamp
                st.caption(formatted_time)
            with col2:
                # Show user and action
                user_email = activity.get('user', activity.get('user_email', 'Unknown'))
                action_text = activity.get('action', 'Unknown action')
                st.text(f"{user_email}: {action_text}")
            with col3:
                # Show activity type badge
                activity_type = activity.get('type', 'unknown')
                if activity_type == 'resume_analysis':
                    st.caption("📄 CV Analysis")
                elif activity_type == 'career_query':
                    st.caption("💬 Career Query")
                else:
                    st.caption("📊 Activity")
    else:
        st.info("No recent activity")

# User Management Page
def show_user_management():
    st.title("👥 User Management")
    st.markdown("View, search, and manage all users")
    
    # Search and Filter
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search = st.text_input("🔍 Search users", placeholder="Email or name...")
    with col2:
        page = st.number_input("Page", min_value=1, value=1, step=1)
    with col3:
        limit = st.selectbox("Per page", [10, 25, 50, 100], index=2)
    
    # Fetch users
    with st.spinner("Loading users..."):
        users_data = fetch_users(page=page, limit=limit, search=search)
    
    if not users_data:
        st.error("Failed to load users")
        return
    
    users = users_data.get('users', [])
    total = users_data.get('total', 0)
    
    st.markdown(f"**Total Users:** {total}")
    
    # Users Table
    if users:
        for user in users:
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**{user.get('full_name', 'N/A')}**")
                    st.caption(f"📧 {user.get('email')}")
                
                with col2:
                    role_emoji = "👨‍💼" if user.get('role') == 'admin' else "👤"
                    status_emoji = "🟢" if user.get('is_active') else "🔴"
                    st.markdown(f"{role_emoji} {user.get('role', 'user').title()}")
                    st.caption(f"{status_emoji} {'Active' if user.get('is_active') else 'Suspended'}")
                
                with col3:
                    st.caption(f"Joined: {user.get('created_at', 'N/A')[:10]}")
                    st.caption(f"Last active: {user.get('last_active', 'N/A')[:10]}")
                
                with col4:
                    if st.button("👁️ View Details", key=f"view_{user.get('id')}"):
                        st.session_state.selected_user_id = user.get('id')
                        st.session_state.show_user_details = True
                    
                    if user.get('role') != 'admin':
                        if user.get('is_active'):
                            if st.button("🚫 Suspend", key=f"suspend_{user.get('id')}"):
                                st.warning("Suspend functionality - implement backend endpoint")
                        else:
                            if st.button("✅ Activate", key=f"activate_{user.get('id')}"):
                                st.success("Activate functionality - implement backend endpoint")
    else:
        st.info("No users found")
    
    # Pagination
    total_pages = (total + limit - 1) // limit
    st.caption(f"Page {page} of {total_pages}")
    
    # User Details Modal
    if st.session_state.get('show_user_details', False):
        user_id = st.session_state.get('selected_user_id')
        if user_id:
            st.markdown("---")
            st.markdown("### 👤 User Details")
            
            # Fetch detailed user info
            headers = {"Authorization": f"Bearer {st.session_state.admin_token}"}
            try:
                response = requests.get(f"{API_BASE_URL}/admin/user/{user_id}", headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    user_detail = data.get('user', {})
                    summary = data.get('summary', {})
                    analyses = data.get('analyses', [])
                    career_queries = data.get('career_queries', [])
                    rag_queries = data.get('rag_queries', [])
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"#### 📧 {user_detail.get('email', 'N/A')}")
                        st.markdown(f"**Full Name:** {user_detail.get('full_name', 'N/A')}")
                        st.markdown(f"**Role:** {user_detail.get('role', 'user').title()}")
                        st.markdown(f"**Status:** {'🟢 Active' if user_detail.get('is_active') else '🔴 Suspended'}")
                        st.markdown(f"**Created:** {user_detail.get('created_at', 'N/A')[:10]}")
                        st.markdown(f"**Last Active:** {user_detail.get('last_active', 'N/A')[:10]}")
                    
                    with col2:
                        if st.button("❌ Close", key="close_user_details"):
                            st.session_state.show_user_details = False
                            st.session_state.selected_user_id = None
                            st.rerun()
                    
                    # Activity Statistics
                    st.markdown("#### 📊 Activity Summary")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("Resume Analyses", summary.get('total_analyses', 0))
                    with stat_col2:
                        st.metric("Career Queries", summary.get('total_queries', 0))
                    with stat_col3:
                        st.metric("RAG Coach Queries", summary.get('total_rag_queries', 0))
                    
                    # Recent Analyses
                    if analyses:
                        st.markdown("#### 📄 Recent Resume Analyses")
                        for i, analysis in enumerate(analyses[:5]):  # Show top 5
                            with st.expander(f"Analysis #{i+1} - {analysis.get('recommended_job', 'N/A')} ({analysis.get('match_percentage', 0)}%)"):
                                st.write(f"**Match Percentage:** {analysis.get('match_percentage', 0)}%")
                                st.write(f"**Skills Count:** {analysis.get('total_skills_count', 0)}")
                                st.write(f"**Date:** {analysis.get('created_at', 'N/A')[:10]}")
                    
                    # Recent Career Queries
                    if career_queries:
                        st.markdown("#### 💬 Recent Career Queries")
                        for i, query in enumerate(career_queries[:5]):  # Show top 5
                            with st.expander(f"Query #{i+1} - {query.get('created_at', 'N/A')[:10]}"):
                                st.write(f"**Question:** {query.get('question', 'N/A')}")
                                st.write(f"**Model:** {query.get('model_used', 'N/A')}")
                                st.write(f"**Response Time:** {query.get('response_time', 0):.2f}s")
                    
                    # Recent RAG Queries
                    if rag_queries:
                        st.markdown("#### 🤖 Recent RAG Coach Queries")
                        for i, rag in enumerate(rag_queries[:5]):  # Show top 5
                            with st.expander(f"Query #{i+1} - {rag.get('created_at', 'N/A')[:10]}"):
                                st.write(f"**Question:** {rag.get('question', 'N/A')[:100]}...")
                                st.write(f"**Query Length:** {rag.get('query_length', 0)} chars")
                                st.write(f"**Answer Length:** {rag.get('answer_length', 0)} chars")
                    
                else:
                    st.error(f"Failed to load user details: {response.status_code}")
                    if st.button("❌ Close"):
                        st.session_state.show_user_details = False
                        st.session_state.selected_user_id = None
                        st.rerun()
            except Exception as e:
                st.error(f"Error loading user details: {str(e)}")
                if st.button("❌ Close"):
                    st.session_state.show_user_details = False
                    st.session_state.selected_user_id = None
                    st.rerun()

# Analytics Page
def show_analytics_page():
    st.title("📊 Advanced Analytics")
    st.markdown("Detailed insights and trends")
    
    stats = fetch_stats()
    if not stats:
        st.error("Failed to load analytics")
        return
    
    tab1, tab2, tab3 = st.tabs(["📈 User Analytics", "💼 Job Market Insights", "🎯 Skill Analytics"])
    
    with tab1:
        st.markdown("### 📊 User Engagement Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("7-Day Retention", f"{stats.get('retention_7days', 0)}%")
            st.caption("Users who return within 7 days")
        with col2:
            st.metric("30-Day Retention", f"{stats.get('retention_30days', 0)}%")
            st.caption("Users who return within 30 days")
        with col3:
            st.metric("Overall Activity Rate", f"{stats.get('retention_rate', 0):.1f}%")
            st.caption("Active users / Total users")
        
        st.markdown("---")
        st.markdown("### 🔥 User Activity Heatmap")
        if stats.get('activity_heatmap'):
            df_heat = pd.DataFrame(stats['activity_heatmap'])
            if not df_heat.empty:
                fig = px.density_heatmap(
                    df_heat,
                    x='hour',
                    y='day',
                    z='count',
                    title="User Activity by Day and Hour",
                    color_continuous_scale='Blues',
                    labels={'hour': 'Hour of Day', 'day': 'Day of Week', 'count': 'Activity Count'}
                )
                fig.update_layout(
                    xaxis={'dtick': 2},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough activity data for heatmap")
        else:
            st.info("No heatmap data available")
    
    with tab2:
        st.markdown("### Job Distribution")
        if stats.get('top_jobs'):
            df_dist = pd.DataFrame(stats['top_jobs'])
            fig = px.pie(
                df_dist,
                values='count',
                names='job',
                title="Career Path Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No job distribution data")
        
        st.markdown("### Trending Careers")
        if stats.get('top_jobs'):
            df_trend = pd.DataFrame(stats['top_jobs'])
            st.dataframe(df_trend, use_container_width=True)
        else:
            st.info("No trending data")
    
    with tab3:
        st.markdown("### Most In-Demand Skills")
        if stats.get('top_missing_skills'):
            st.markdown("Skills most frequently missing from analyzed resumes:")
            df_skills = pd.DataFrame(stats['top_missing_skills'])
            # Show as a bar chart
            fig = px.bar(
                df_skills,
                x='skill',
                y='count',
                title="Top Skills to Learn",
                color='count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False, xaxis_title="Skill", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No skill data")
        
        st.markdown("### Skills Leaderboard")
        if stats.get('top_missing_skills'):
            st.dataframe(
                pd.DataFrame(stats['top_missing_skills']),
                use_container_width=True
            )

# Main App
def main():
    if not st.session_state.admin_token:
        show_login()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
