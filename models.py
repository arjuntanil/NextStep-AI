# models.py
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime

# 1. Database Configuration
DATABASE_URL = "sqlite:///./nextstepai.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. Database Models (Tables)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String)
    password_hash = Column(String, nullable=False)  # Hashed password
    role = Column(String, default="user")  # "user" or "admin"
    is_active = Column(Boolean, default=True)  # For ban/suspend
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    # Relationships: A user can have many analyses and queries
    analyses = relationship("ResumeAnalysis", back_populates="owner")
    queries = relationship("CareerQuery", back_populates="owner")
    rag_queries = relationship("RAGCoachQuery", back_populates="owner")

class ResumeAnalysis(Base):
    __tablename__ = "resume_analyses"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    recommended_job_title = Column(String)
    match_percentage = Column(Integer)
    skills_to_add = Column(Text) # Storing list as a JSON string
    resume_filename = Column(String)
    total_skills_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="analyses")

class CareerQuery(Base):
    __tablename__ = "career_queries"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    user_query_text = Column(String)
    matched_job_group = Column(String)
    model_used = Column(String)  # "finetuned" or "rag"
    response_time_seconds = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="queries")

class RAGCoachQuery(Base):
    __tablename__ = "rag_coach_queries"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    question = Column(Text)
    answer = Column(Text)
    sources = Column(Text)  # JSON string of source documents
    query_length = Column(Integer)
    answer_length = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="rag_queries")

# 3. Create Database Tables
# This function should be called once from your main backend file on startup
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)