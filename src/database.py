import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

# Load environment variables from the hidden .env file
load_dotenv()

# --- 1. Connection Setup ---
DB_URL = os.getenv("DATABASE_URL", "postgresql://YOUR_USERNAME:YOUR_PASSWORD@localhost:5432/mental_health_monitor")

engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 2. Schema Definitions ---

class StudentProfile(Base):
    """
    Acts as both the Authentication Profile and the AI Baseline Tracker.
    """
    __tablename__ = 'student_profiles' 

    student_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Authentication Data (Simplified to Plaintext)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False) # Changed from password_hash
    
    # Historical AI Averages
    avg_speech_pace = Column(Float, default=0.0)
    base_participation_rate = Column(Float, default=0.0) 
    base_visual_engagement = Column(Float, default=0.0)  
    
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    sessions = relationship("ClassSession", back_populates="student")

class ClassSession(Base):
    """
    Records the telemetry and AI analysis for a single class.
    """
    __tablename__ = 'class_sessions'

    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = Column(String, ForeignKey('student_profiles.student_id'))
    
    date = Column(DateTime, default=datetime.utcnow)
    
    pre_class_mood = Column(Integer, nullable=True) 
    dominant_affect = Column(String) 
    engagement_score = Column(Float) 
    behavioral_telemetry = Column(JSONB) 
    audio_mfcc_profile = Column(ARRAY(Float)) 
    anomaly_flag = Column(Boolean, default=False) 

    student = relationship("StudentProfile", back_populates="sessions")

# --- 3. Database Initialization & Seeding ---
def init_db():
    print("⚙️ Initializing Simplified PostgreSQL Database Schema...")
    Base.metadata.drop_all(bind=engine) 
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully.")
    
    db = SessionLocal()
    test_email = "student@university.edu"
    
    existing_user = db.query(StudentProfile).filter(StudentProfile.email == test_email).first()
    if not existing_user:
        print("👤 Creating Test Student Account...")
        test_student = StudentProfile(
            name="Test Student",
            email=test_email,
            password="password123" # Stored as plain text!
        )
        db.add(test_student)
        db.commit()
        print(f"✅ Test Student Created! Email: {test_email} | Password: password123")
    db.close()

if __name__ == "__main__":
    init_db()