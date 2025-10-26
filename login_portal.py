"""
NextStepAI - Unified Login Portal
Automatically redirects users based on their role:
- Admin users → Admin Dashboard (port 8502)
- Regular users → User App (port 8501)
"""

import streamlit as st
import requests
import time

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"
LOGIN_URL = f"{API_BASE_URL}/auth/manual-login"
USER_ME_URL = f"{API_BASE_URL}/users/me"

# Page Configuration
st.set_page_config(
    page_title="NextStepAI - Login Portal",
    page_icon="🔐",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: transparent;
    }
    div[data-testid="stForm"] {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: white;'>✨ NextStepAI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Career Navigator Portal</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Login Form
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.form("login_form", clear_on_submit=False):
        st.markdown("### 🔑 Login")
        
        email = st.text_input(
            "Email Address",
            placeholder="your@email.com",
            help="Enter your registered email address"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            help="Enter your password"
        )
        
        submit_button = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
        
        if submit_button:
            if email and password:
                with st.spinner("🔄 Authenticating..."):
                    try:
                        # Authenticate user
                        response = requests.post(LOGIN_URL, json={
                            "email": email,
                            "password": password
                        })
                        
                        if response.status_code == 200:
                            token_data = response.json()
                            access_token = token_data["access_token"]
                            
                            # Fetch user information
                            headers = {"Authorization": f"Bearer {access_token}"}
                            user_response = requests.get(USER_ME_URL, headers=headers)
                            
                            if user_response.status_code == 200:
                                user_info = user_response.json()
                                user_role = user_info.get('role', 'user')
                                user_name = user_info.get('full_name', 'User')
                                
                                # Redirect based on role
                                if user_role == 'admin':
                                    st.success(f"✅ Welcome Admin {user_name}!")
                                    st.info("🔄 Redirecting to Admin Dashboard...")
                                    
                                    # Auto-redirect with JavaScript
                                    st.markdown(f"""
                                    <div style='text-align: center; padding: 20px; background: #e8f5e9; border-radius: 10px; margin-top: 20px;'>
                                        <h3>🎯 Admin Dashboard</h3>
                                        <p>Redirecting automatically in 2 seconds...</p>
                                        <p style='color: #666; font-size: 0.9em;'>Or click below if not redirected:</p>
                                        <a href='http://localhost:8502?token={access_token}' target='_self' 
                                           style='display: inline-block; padding: 12px 30px; background: #667eea; 
                                                  color: white; text-decoration: none; border-radius: 5px; 
                                                  font-weight: bold; margin-top: 10px;'>
                                            Open Admin Dashboard →
                                        </a>
                                    </div>
                                    <script>
                                        setTimeout(function() {{
                                            window.location.href = 'http://localhost:8502?token={access_token}';
                                        }}, 2000);
                                    </script>
                                    """, unsafe_allow_html=True)
                                    
                                else:
                                    st.success(f"✅ Welcome {user_name}!")
                                    st.info("🔄 Redirecting to Career Navigator...")
                                    
                                    # Auto-redirect with JavaScript
                                    st.markdown(f"""
                                    <div style='text-align: center; padding: 20px; background: #e3f2fd; border-radius: 10px; margin-top: 20px;'>
                                        <h3>🎯 Career Navigator</h3>
                                        <p>Redirecting automatically in 2 seconds...</p>
                                        <p style='color: #666; font-size: 0.9em;'>Or click below if not redirected:</p>
                                        <a href='http://localhost:8501?token={access_token}' target='_self' 
                                           style='display: inline-block; padding: 12px 30px; background: #764ba2; 
                                                  color: white; text-decoration: none; border-radius: 5px; 
                                                  font-weight: bold; margin-top: 10px;'>
                                            Open Career Navigator →
                                        </a>
                                    </div>
                                    <script>
                                        setTimeout(function() {{
                                            window.location.href = 'http://localhost:8501?token={access_token}';
                                        }}, 2000);
                                    </script>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error("❌ Failed to fetch user information")
                        
                        else:
                            error_detail = response.json().get('detail', 'Invalid email or password')
                            st.error(f"❌ {error_detail}")
                    
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend server. Please ensure the backend is running on port 8000.")
                        st.info("💡 Run: `uvicorn backend_api:app --reload`")
                    
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Connection error: {e}")
            
            else:
                st.warning("⚠️ Please enter both email and password")
    
    # Additional Information
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>Don't have an account? Register in the User App at <a href='http://localhost:8501' target='_blank'>localhost:8501</a></p>
        <p style='margin-top: 10px; color: #999;'>Admin credentials: admin@gmail.com / admin</p>
    </div>
    """, unsafe_allow_html=True)
