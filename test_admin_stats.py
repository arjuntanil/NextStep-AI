"""
Test admin stats endpoint
"""
import sys
sys.path.insert(0, 'E:/NextStepAI')

from backend_api import app, get_db
from fastapi.testclient import TestClient
import json

# Create test client
client = TestClient(app)

# Login as admin
print("1. Logging in as admin...")
login_response = client.post("/admin/login", json={
    "email": "admin@gmail.com",
    "password": "admin"
})
print(f"   Login Status: {login_response.status_code}")

if login_response.status_code != 200:
    print(f"   Login failed: {login_response.text}")
    sys.exit(1)

token = login_response.json()["access_token"]
print(f"   Token obtained: {token[:50]}...")

# Test admin stats
print("\n2. Testing /admin/stats endpoint...")
headers = {"Authorization": f"Bearer {token}"}
stats_response = client.get("/admin/stats", headers=headers)

print(f"   Status Code: {stats_response.status_code}")

if stats_response.status_code == 200:
    stats = stats_response.json()
    print("\n✅ SUCCESS! Admin stats endpoint working!")
    print(f"\nStats Summary:")
    print(f"  - Total Users: {stats.get('total_users')}")
    print(f"  - Active Users (30d): {stats.get('active_users_30d')}")
    print(f"  - Active Users (7d): {stats.get('active_users_7d')}")
    print(f"  - Retention Rate: {stats.get('retention_rate')}%")
    print(f"  - User Growth Data Points: {len(stats.get('user_growth', []))}")
    print(f"  - Top Jobs: {len(stats.get('top_jobs', []))}")
    print(f"  - Top Missing Skills: {len(stats.get('top_missing_skills', []))}")
    print(f"  - Recent Activities: {len(stats.get('recent_activity', []))}")
    
    if stats.get('top_jobs'):
        print(f"\n  Top 3 Jobs:")
        for job in stats['top_jobs'][:3]:
            print(f"    - {job['job']}: {job['count']} times")
else:
    print(f"\n❌ ERROR {stats_response.status_code}")
    print(f"Response: {stats_response.text}")
    
    # Try to get more details
    try:
        error_detail = stats_response.json()
        print(f"\nError Detail: {json.dumps(error_detail, indent=2)}")
    except:
        pass

print("\n" + "="*50)
