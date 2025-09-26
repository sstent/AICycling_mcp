"""
SQLAlchemy models for persistent data storage
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class UserProfile(Base):
    """User profile data"""
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True)
    ftp = Column(Float, nullable=True)
    max_hr = Column(Integer, nullable=True)
    # Add other profile fields as needed
    
    # Relationship to workouts if needed
    workouts = relationship("Workout", back_populates="user_profile")

class Workout(Base):
    """Workout data from Garmin"""
    __tablename__ = 'workouts'
    
    id = Column(Integer, primary_key=True)
    activity_id = Column(String, unique=True, index=True)
    date = Column(DateTime, default=datetime.utcnow)
    data_quality = Column(String, default='complete')
    is_indoor = Column(Boolean, default=False)
    # Raw activity data as JSON for flexibility
    raw_data = Column(Text)
    validation_warnings = Column(Text, nullable=True)
    
    # Relationship to metrics
    metrics = relationship("Metrics", back_populates="workout", uselist=True)
    
    # Relationship to user profile (optional, for future)
    user_profile_id = Column(Integer, ForeignKey('user_profiles.id'), nullable=True)
    user_profile = relationship("UserProfile", back_populates="workouts")

class Metrics(Base):
    """Calculated metrics for workouts"""
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True)
    workout_id = Column(String, ForeignKey('workouts.activity_id'), unique=True)
    
    # Key metrics from Garmin and calculations
    avg_speed_kmh = Column(Float, nullable=True)
    avg_hr = Column(Float, nullable=True)
    avg_power = Column(Float, nullable=True)
    estimated_ftp = Column(Float, nullable=True)
    duration_minutes = Column(Float, nullable=True)
    distance_km = Column(Float, nullable=True)
    elevation_gain_m = Column(Float, nullable=True)
    training_stress_score = Column(Float, nullable=True)
    intensity_factor = Column(Float, nullable=True)
    workout_classification = Column(String, nullable=True)
    
    # Relationship back to workout
    workout = relationship("Workout", back_populates="metrics")