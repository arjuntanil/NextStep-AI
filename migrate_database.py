"""
Database Migration Script
Adds new fields to existing database while preserving current users
"""

import sqlite3
from datetime import datetime
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def migrate_database():
    print("=" * 60)
    print("DATABASE MIGRATION - NextStepAI")
    print("=" * 60)
    print()
    
    conn = sqlite3.connect('nextstepai.db')
    cursor = conn.cursor()
    
    try:
        # Check if migration is needed
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        
        print("Current user table columns:", columns)
        print()
        
        # Add new columns if they don't exist
        migrations = []
        
        if 'password_hash' not in columns:
            migrations.append(("password_hash", "ALTER TABLE users ADD COLUMN password_hash TEXT DEFAULT 'temp_hash'"))
        
        if 'role' not in columns:
            migrations.append(("role", "ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"))
        
        if 'is_active' not in columns:
            migrations.append(("is_active", "ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"))
        
        if 'created_at' not in columns:
            migrations.append(("created_at", f"ALTER TABLE users ADD COLUMN created_at TEXT DEFAULT '{datetime.utcnow().isoformat()}'"))
        
        if 'last_active' not in columns:
            migrations.append(("last_active", f"ALTER TABLE users ADD COLUMN last_active TEXT DEFAULT '{datetime.utcnow().isoformat()}'"))
        
        # Execute migrations
        if migrations:
            print(f"Applying {len(migrations)} migrations to users table...")
            for col_name, sql in migrations:
                print(f"  ✓ Adding column: {col_name}")
                cursor.execute(sql)
            conn.commit()
            print("✅ User table migration complete!")
        else:
            print("✅ User table already up to date!")
        
        print()
        
        # Set default password for existing users
        cursor.execute("SELECT id, email FROM users WHERE password_hash = 'temp_hash' OR password_hash IS NULL")
        users_needing_password = cursor.fetchall()
        
        if users_needing_password:
            print(f"Setting default password for {len(users_needing_password)} existing users...")
            default_password = "password123"  # Users should change this
            hashed = pwd_context.hash(default_password)
            
            for user_id, email in users_needing_password:
                cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hashed, user_id))
                print(f"  ✓ {email} - Password set to: {default_password}")
            
            conn.commit()
            print()
            print("⚠️  IMPORTANT: Existing users have default password: 'password123'")
            print("   They should change it after first login!")
        
        print()
        
        # Migrate resume_analyses table
        cursor.execute("PRAGMA table_info(resume_analyses)")
        columns = [col[1] for col in cursor.fetchall()]
        
        analyses_migrations = []
        if 'resume_filename' not in columns:
            analyses_migrations.append(("resume_filename", "ALTER TABLE resume_analyses ADD COLUMN resume_filename TEXT"))
        if 'total_skills_count' not in columns:
            analyses_migrations.append(("total_skills_count", "ALTER TABLE resume_analyses ADD COLUMN total_skills_count INTEGER DEFAULT 0"))
        if 'created_at' not in columns:
            analyses_migrations.append(("created_at", f"ALTER TABLE resume_analyses ADD COLUMN created_at TEXT DEFAULT '{datetime.utcnow().isoformat()}'"))
        
        if analyses_migrations:
            print(f"Applying {len(analyses_migrations)} migrations to resume_analyses table...")
            for col_name, sql in analyses_migrations:
                print(f"  ✓ Adding column: {col_name}")
                cursor.execute(sql)
            conn.commit()
            print("✅ Resume analyses table migration complete!")
        else:
            print("✅ Resume analyses table already up to date!")
        
        print()
        
        # Migrate career_queries table
        cursor.execute("PRAGMA table_info(career_queries)")
        columns = [col[1] for col in cursor.fetchall()]
        
        queries_migrations = []
        if 'model_used' not in columns:
            queries_migrations.append(("model_used", "ALTER TABLE career_queries ADD COLUMN model_used TEXT DEFAULT 'rag'"))
        if 'response_time_seconds' not in columns:
            queries_migrations.append(("response_time_seconds", "ALTER TABLE career_queries ADD COLUMN response_time_seconds INTEGER DEFAULT 0"))
        if 'created_at' not in columns:
            queries_migrations.append(("created_at", f"ALTER TABLE career_queries ADD COLUMN created_at TEXT DEFAULT '{datetime.utcnow().isoformat()}'"))
        
        if queries_migrations:
            print(f"Applying {len(queries_migrations)} migrations to career_queries table...")
            for col_name, sql in queries_migrations:
                print(f"  ✓ Adding column: {col_name}")
                cursor.execute(sql)
            conn.commit()
            print("✅ Career queries table migration complete!")
        else:
            print("✅ Career queries table already up to date!")
        
        print()
        
        # Migrate rag_coach_queries table
        cursor.execute("PRAGMA table_info(rag_coach_queries)")
        columns = [col[1] for col in cursor.fetchall()]
        
        rag_migrations = []
        if 'query_length' not in columns:
            rag_migrations.append(("query_length", "ALTER TABLE rag_coach_queries ADD COLUMN query_length INTEGER DEFAULT 0"))
        if 'answer_length' not in columns:
            rag_migrations.append(("answer_length", "ALTER TABLE rag_coach_queries ADD COLUMN answer_length INTEGER DEFAULT 0"))
        if 'created_at' not in columns:
            rag_migrations.append(("created_at", f"ALTER TABLE rag_coach_queries ADD COLUMN created_at TEXT DEFAULT '{datetime.utcnow().isoformat()}'"))
        
        if rag_migrations:
            print(f"Applying {len(rag_migrations)} migrations to rag_coach_queries table...")
            for col_name, sql in rag_migrations:
                print(f"  ✓ Adding column: {col_name}")
                cursor.execute(sql)
            conn.commit()
            print("✅ RAG coach queries table migration complete!")
        else:
            print("✅ RAG coach queries table already up to date!")
        
        print()
        print("=" * 60)
        print("✅ MIGRATION COMPLETE!")
        print("=" * 60)
        print()
        print("Summary:")
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"  📊 Total users: {user_count}")
        
        cursor.execute("SELECT COUNT(*) FROM resume_analyses")
        analyses_count = cursor.fetchone()[0]
        print(f"  📄 Total analyses: {analyses_count}")
        
        cursor.execute("SELECT COUNT(*) FROM career_queries")
        queries_count = cursor.fetchone()[0]
        print(f"  💬 Total career queries: {queries_count}")
        
        cursor.execute("SELECT COUNT(*) FROM rag_coach_queries")
        rag_count = cursor.fetchone()[0]
        print(f"  🧑‍💼 Total RAG queries: {rag_count}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()
