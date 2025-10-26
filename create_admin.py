"""
Create Admin User Script
Creates the first admin user: admin@gmail.com / admin
"""

import sqlite3
from datetime import datetime
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_admin():
    print("=" * 60)
    print("CREATE ADMIN USER - NextStepAI")
    print("=" * 60)
    print()
    
    conn = sqlite3.connect('nextstepai.db')
    cursor = conn.cursor()
    
    try:
        # Check if admin already exists
        cursor.execute("SELECT id, email FROM users WHERE email = ?", ("admin@gmail.com",))
        existing_admin = cursor.fetchone()
        
        if existing_admin:
            print(f"⚠️  Admin user already exists: {existing_admin[1]}")
            print()
            response = input("Update admin password to 'admin'? (y/n): ")
            
            if response.lower() == 'y':
                hashed_password = pwd_context.hash("admin")
                cursor.execute("""
                    UPDATE users 
                    SET password_hash = ?, role = 'admin', is_active = 1, last_active = ?
                    WHERE email = ?
                """, (hashed_password, datetime.utcnow().isoformat(), "admin@gmail.com"))
                conn.commit()
                print("✅ Admin password updated!")
            else:
                print("❌ Admin password not changed.")
        else:
            # Create new admin user
            hashed_password = pwd_context.hash("admin")
            now = datetime.utcnow().isoformat()
            
            cursor.execute("""
                INSERT INTO users (email, full_name, password_hash, role, is_active, created_at, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "admin@gmail.com",
                "System Administrator",
                hashed_password,
                "admin",
                1,
                now,
                now
            ))
            conn.commit()
            
            print("✅ Admin user created successfully!")
            print()
            print("Admin Credentials:")
            print("  📧 Email: admin@gmail.com")
            print("  🔑 Password: admin")
            print()
            print("⚠️  IMPORTANT: Change this password after first login!")
        
        print()
        print("=" * 60)
        print("Admin dashboard will be available at:")
        print("  🌐 http://localhost:8502")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error creating admin: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    create_admin()
