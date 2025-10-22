# models.py
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

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
    
    owner = relationship("User", back_populates="analyses")

class CareerQuery(Base):
    __tablename__ = "career_queries"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    user_query_text = Column(String)
    matched_job_group = Column(String)
    
    owner = relationship("User", back_populates="queries")

class RAGCoachQuery(Base):
    __tablename__ = "rag_coach_queries"
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    question = Column(Text)
    answer = Column(Text)
    sources = Column(Text)  # JSON string of source documents
    
    owner = relationship("User", back_populates="rag_queries")

# 3. Create Database Tables
# This function should be called once from your main backend file on startup
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)