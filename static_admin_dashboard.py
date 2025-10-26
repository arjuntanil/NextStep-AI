"""
NextStepAI - Static Admin Dashboard
Professional data-focused analytics dashboard with static data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Page Configuration
st.set_page_config(
    page_title="NextStepAI - Admin Dashboard",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1f2937;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
    }
    h2, h3 {
        color: #374151;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== STATIC DATA ====================

# Static Users Data
STATIC_USERS = [
    {
        "id": 1,
        "full_name": "Sarah Johnson",
        "email": "sarah.johnson@email.com",
        "role": "user",
        "is_active": True,
        "created_at": "2025-09-15",
        "last_active": "2025-10-24",
        "analyses_count": 12,
        "queries_count": 34,
        "avg_match_score": 87.5
    },
    {
        "id": 2,
        "full_name": "Michael Chen",
        "email": "michael.chen@email.com",
        "role": "user",
        "is_active": True,
        "created_at": "2025-09-20",
        "last_active": "2025-10-25",
        "analyses_count": 8,
        "queries_count": 22,
        "avg_match_score": 72.3
    },
    {
        "id": 3,
        "full_name": "Emily Rodriguez",
        "email": "emily.rodriguez@email.com",
        "role": "user",
        "is_active": True,
        "created_at": "2025-10-01",
        "last_active": "2025-10-23",
        "analyses_count": 15,
        "queries_count": 45,
        "avg_match_score": 91.2
    },
    {
        "id": 4,
        "full_name": "David Kumar",
        "email": "david.kumar@email.com",
        "role": "user",
        "is_active": True,
        "created_at": "2025-10-05",
        "last_active": "2025-10-25",
        "analyses_count": 6,
        "queries_count": 18,
        "avg_match_score": 68.9
    },
    {
        "id": 5,
        "full_name": "Jessica Williams",
        "email": "jessica.williams@email.com",
        "role": "user",
        "is_active": False,
        "created_at": "2025-09-10",
        "last_active": "2025-10-10",
        "analyses_count": 3,
        "queries_count": 7,
        "avg_match_score": 55.4
    }
]

# User Growth Data (Last 30 days)
USER_GROWTH_DATA = [
    {"date": "2025-09-26", "count": 0},
    {"date": "2025-09-27", "count": 0},
    {"date": "2025-09-28", "count": 0},
    {"date": "2025-09-29", "count": 0},
    {"date": "2025-09-30", "count": 0},
    {"date": "2025-10-01", "count": 1},
    {"date": "2025-10-02", "count": 1},
    {"date": "2025-10-03", "count": 1},
    {"date": "2025-10-04", "count": 1},
    {"date": "2025-10-05", "count": 2},
    {"date": "2025-10-06", "count": 2},
    {"date": "2025-10-07", "count": 2},
    {"date": "2025-10-08", "count": 2},
    {"date": "2025-10-09", "count": 2},
    {"date": "2025-10-10", "count": 2},
    {"date": "2025-10-11", "count": 2},
    {"date": "2025-10-12", "count": 2},
    {"date": "2025-10-13", "count": 2},
    {"date": "2025-10-14", "count": 2},
    {"date": "2025-10-15", "count": 3},
    {"date": "2025-10-16", "count": 3},
    {"date": "2025-10-17", "count": 3},
    {"date": "2025-10-18", "count": 3},
    {"date": "2025-10-19", "count": 3},
    {"date": "2025-10-20", "count": 4},
    {"date": "2025-10-21", "count": 4},
    {"date": "2025-10-22", "count": 4},
    {"date": "2025-10-23", "count": 4},
    {"date": "2025-10-24", "count": 4},
    {"date": "2025-10-25", "count": 5},
]

# Top Jobs Data
TOP_JOBS_DATA = [
    {"job": "Software Developer", "count": 18},
    {"job": "Data Scientist", "count": 12},
    {"job": "Product Manager", "count": 8},
    {"job": "UX Designer", "count": 6},
    {"job": "DevOps Engineer", "count": 5}
]

# Top Missing Skills
TOP_SKILLS_DATA = [
    {"skill": "Python", "count": 25},
    {"skill": "Machine Learning", "count": 18},
    {"skill": "React", "count": 15},
    {"skill": "Docker", "count": 12},
    {"skill": "AWS", "count": 10},
    {"skill": "SQL", "count": 9},
    {"skill": "Git", "count": 8}
]

# Match Score Distribution
MATCH_SCORES = [87.5, 72.3, 91.2, 68.9, 55.4, 78.6, 82.1, 65.3, 88.9, 73.2, 
                85.7, 70.1, 92.4, 67.8, 81.5, 74.9, 86.3, 69.5, 90.1, 76.4]

# Recent Activity
RECENT_ACTIVITY = [
    {
        "timestamp": "2025-10-25 14:30:22",
        "user_email": "david.kumar@email.com",
        "action": "Analyzed resume for DevOps Engineer role",
        "type": "resume_analysis"
    },
    {
        "timestamp": "2025-10-25 13:15:10",
        "user_email": "emily.rodriguez@email.com",
        "action": "Asked career advice about Data Science",
        "type": "career_query"
    },
    {
        "timestamp": "2025-10-25 11:45:33",
        "user_email": "michael.chen@email.com",
        "action": "Uploaded resume and job description",
        "type": "rag_query"
    },
    {
        "timestamp": "2025-10-24 16:20:15",
        "user_email": "sarah.johnson@email.com",
        "action": "Analyzed resume for Product Manager role",
        "type": "resume_analysis"
    },
    {
        "timestamp": "2025-10-24 14:55:42",
        "user_email": "emily.rodriguez@email.com",
        "action": "Asked about career transition to AI",
        "type": "career_query"
    },
    {
        "timestamp": "2025-10-24 10:30:18",
        "user_email": "david.kumar@email.com",
        "action": "Analyzed resume for Software Developer",
        "type": "resume_analysis"
    },
    {
        "timestamp": "2025-10-23 15:40:29",
        "user_email": "michael.chen@email.com",
        "action": "Asked about UX Design career path",
        "type": "career_query"
    },
    {
        "timestamp": "2025-10-23 12:10:55",
        "user_email": "sarah.johnson@email.com",
        "action": "Uploaded JD for gap analysis",
        "type": "rag_query"
    }
]

# ==================== SESSION STATE ====================

if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False
if 'admin_email' not in st.session_state:
    st.session_state.admin_email = None

# ==================== LOGIN FUNCTION ====================

def show_login():
    """Display admin login form"""
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>🔐 Admin Login</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6b7280;'>NextStepAI Admin Dashboard</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div style='background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            """, unsafe_allow_html=True)
            
            email = st.text_input("📧 Email", placeholder="admin@gmail.com", key="admin_email_input")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter password", key="admin_password_input")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Login to Dashboard", type="primary", use_container_width=True):
                if email == "admin@gmail.com" and password == "admin":
                    st.session_state.admin_authenticated = True
                    st.session_state.admin_email = email
                    st.success("✅ Login successful! Redirecting to dashboard...")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Use: admin@gmail.com / admin")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; color: #9ca3af; font-size: 0.85em;'>
                <p>🔒 Secure admin access only</p>
                <p>Default credentials: admin@gmail.com / admin</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== DASHBOARD FUNCTIONS ====================

def show_dashboard():
    """Display the main admin dashboard"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👨‍💼 Admin Panel")
        st.markdown(f"**{st.session_state.admin_email}**")
        st.markdown("---")
        
        page = st.radio(
            "📊 Navigation",
            ["📈 Dashboard Overview", "👥 User Management", "📊 Analytics & Insights"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📌 Quick Stats")
        st.metric("Total Users", "5")
        st.metric("Active Today", "4")
        st.metric("System Status", "🟢 Healthy")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.session_state.admin_email = None
            st.rerun()
    
    # Main Content
    if page == "📈 Dashboard Overview":
        show_dashboard_overview()
    elif page == "👥 User Management":
        show_user_management()
    elif page == "📊 Analytics & Insights":
        show_analytics()

def show_dashboard_overview():
    """Dashboard overview page with key metrics and charts"""
    
    st.title("📊 Dashboard Overview")
    st.markdown("Real-time analytics and system insights")
    st.markdown("---")
    
    # === KEY METRICS ROW ===
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="👥 Total Users",
            value="5",
            delta="+2 this week",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="🟢 Active Users",
            value="4",
            delta="80% active",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="📄 Total Analyses",
            value="44",
            delta="+6 this week",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="💬 Career Queries",
            value="126",
            delta="+18 this week",
            delta_color="normal"
        )
    
    with col5:
        st.metric(
            label="📊 Avg Match Score",
            value="75.1%",
            delta="+3.2%",
            delta_color="normal"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === CHARTS ROW 1 ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 User Growth (Last 30 Days)")
        df_growth = pd.DataFrame(USER_GROWTH_DATA)
        fig = px.line(
            df_growth,
            x='date',
            y='count',
            markers=True,
            title="Cumulative User Growth"
        )
        fig.update_traces(line_color='#3b82f6', line_width=3, marker=dict(size=8))
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#374151'),
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏆 Top Recommended Career Paths")
        df_jobs = pd.DataFrame(TOP_JOBS_DATA)
        fig = px.bar(
            df_jobs,
            x='count',
            y='job',
            orientation='h',
            title="Most Popular Job Recommendations",
            color='count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#374151'),
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
            yaxis=dict(showgrid=False, categoryorder='total ascending'),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === CHARTS ROW 2 ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Most In-Demand Skills")
        df_skills = pd.DataFrame(TOP_SKILLS_DATA)
        fig = px.bar(
            df_skills,
            x='count',
            y='skill',
            orientation='h',
            title="Skills Gap Analysis",
            color='count',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#374151'),
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
            yaxis=dict(showgrid=False, categoryorder='total ascending'),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Match Score Distribution")
        fig = px.histogram(
            x=MATCH_SCORES,
            nbins=10,
            title="User-Job Match Scores",
            labels={'x': 'Match %', 'count': 'Number of Analyses'},
            color_discrete_sequence=['#10b981']
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#374151'),
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb', title='Match Percentage'),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb', title='Count'),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === RECENT ACTIVITY ===
    st.markdown("---")
    st.markdown("#### 🕒 Recent Activity")
    
    activity_df = pd.DataFrame(RECENT_ACTIVITY)
    
    for idx, activity in enumerate(RECENT_ACTIVITY[:10]):
        col1, col2, col3 = st.columns([2, 5, 2])
        
        with col1:
            st.caption(activity['timestamp'])
        
        with col2:
            icon = "📄" if activity['type'] == 'resume_analysis' else "💬" if activity['type'] == 'career_query' else "🧑‍💼"
            st.text(f"{icon} {activity['action']}")
        
        with col3:
            st.caption(activity['user_email'].split('@')[0])
        
        if idx < 9:
            st.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

def show_user_management():
    """User management page"""
    
    st.title("👥 User Management")
    st.markdown("View and manage all registered users")
    st.markdown("---")
    
    # Search and filters
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search = st.text_input("🔍 Search users", placeholder="Email or name...")
    
    with col2:
        filter_status = st.selectbox("Status", ["All", "Active", "Inactive"])
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Name", "Join Date", "Activity"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filter users based on search and status
    filtered_users = STATIC_USERS
    
    if search:
        filtered_users = [u for u in filtered_users if search.lower() in u['email'].lower() or search.lower() in u['full_name'].lower()]
    
    if filter_status == "Active":
        filtered_users = [u for u in filtered_users if u['is_active']]
    elif filter_status == "Inactive":
        filtered_users = [u for u in filtered_users if not u['is_active']]
    
    st.markdown(f"**Total Users:** {len(filtered_users)}")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display users as cards
    for user in filtered_users:
        with st.container():
            st.markdown(f"""
            <div style='background: white; padding: 20px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                st.markdown(f"**{user['full_name']}**")
                st.caption(f"📧 {user['email']}")
            
            with col2:
                role_emoji = "👨‍💼" if user['role'] == 'admin' else "👤"
                status_emoji = "🟢" if user['is_active'] else "🔴"
                st.markdown(f"{role_emoji} {user['role'].title()}")
                st.caption(f"{status_emoji} {'Active' if user['is_active'] else 'Inactive'}")
            
            with col3:
                st.caption(f"Joined: {user['created_at']}")
                st.caption(f"Last active: {user['last_active']}")
            
            with col4:
                st.metric("Analyses", user['analyses_count'], label_visibility="visible")
                st.metric("Queries", user['queries_count'], label_visibility="visible")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Expandable details
            with st.expander("📋 View Detailed Statistics"):
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("User ID", f"#{user['id']}")
                
                with stat_col2:
                    st.metric("Avg Match Score", f"{user['avg_match_score']}%")
                
                with stat_col3:
                    st.metric("Total Interactions", user['analyses_count'] + user['queries_count'])
                
                with stat_col4:
                    days_active = (datetime.strptime("2025-10-25", "%Y-%m-%d") - 
                                 datetime.strptime(user['created_at'], "%Y-%m-%d")).days
                    st.metric("Days Active", days_active)

def show_analytics():
    """Advanced analytics page"""
    
    st.title("📊 Advanced Analytics & Business Intelligence")
    st.markdown("Detailed insights and market trends")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📈 User Analytics", "💼 Job Market Insights", "🎯 Skill Trends"])
    
    with tab1:
        st.markdown("### User Engagement Analytics")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Users (7d)", "4", "+1 from last week")
            st.metric("Active Users (30d)", "5", "+3 from last month")
        
        with col2:
            st.metric("Retention Rate (7d)", "80%", "+10%")
            st.metric("Retention Rate (30d)", "100%", "↑")
        
        with col3:
            st.metric("Avg Session Duration", "12 min", "+2 min")
            st.metric("Feature Usage", "High", "↑")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # User engagement over time
        st.markdown("#### User Engagement Over Time")
        engagement_data = {
            'date': pd.date_range(start='2025-09-26', end='2025-10-25', freq='D'),
            'analyses': [random.randint(0, 5) for _ in range(30)],
            'queries': [random.randint(0, 10) for _ in range(30)]
        }
        df_engagement = pd.DataFrame(engagement_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_engagement['date'], y=df_engagement['analyses'], 
                                mode='lines+markers', name='Resume Analyses',
                                line=dict(color='#3b82f6', width=2)))
        fig.add_trace(go.Scatter(x=df_engagement['date'], y=df_engagement['queries'],
                                mode='lines+markers', name='Career Queries',
                                line=dict(color='#10b981', width=2)))
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#374151'),
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Job Market Intelligence")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Job distribution pie chart
            st.markdown("#### Career Path Distribution")
            df_jobs = pd.DataFrame(TOP_JOBS_DATA)
            fig = px.pie(
                df_jobs,
                values='count',
                names='job',
                title="Distribution of Career Recommendations",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#374151'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trending careers
            st.markdown("#### Top Career Paths")
            st.markdown("<br>", unsafe_allow_html=True)
            
            for idx, job in enumerate(TOP_JOBS_DATA, 1):
                percentage = (job['count'] / sum([j['count'] for j in TOP_JOBS_DATA])) * 100
                st.markdown(f"""
                <div style='background: white; padding: 15px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='font-weight: 600; color: #1f2937;'>#{idx} {job['job']}</span>
                        </div>
                        <div>
                            <span style='color: #6b7280;'>{job['count']} recommendations</span>
                            <span style='color: #3b82f6; margin-left: 10px;'>({percentage:.1f}%)</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Skills Demand Analysis")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Skills demand chart
        df_skills = pd.DataFrame(TOP_SKILLS_DATA)
        fig = px.bar(
            df_skills,
            x='skill',
            y='count',
            title="Most In-Demand Skills",
            color='count',
            color_continuous_scale='Viridis',
            text='count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#374151'),
            xaxis=dict(showgrid=False, title='Skill'),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb', title='Frequency'),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Skills table
        st.markdown("#### Detailed Skills Analysis")
        skills_table_data = []
        for idx, skill in enumerate(TOP_SKILLS_DATA, 1):
            percentage = (skill['count'] / sum([s['count'] for s in TOP_SKILLS_DATA])) * 100
            skills_table_data.append({
                "Rank": idx,
                "Skill": skill['skill'],
                "Frequency": skill['count'],
                "Percentage": f"{percentage:.1f}%",
                "Trend": "📈" if idx <= 3 else "➡️"
            })
        
        df_skills_table = pd.DataFrame(skills_table_data)
        st.dataframe(df_skills_table, use_container_width=True, hide_index=True)

# ==================== MAIN APPLICATION ====================

def main():
    if not st.session_state.admin_authenticated:
        show_login()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
