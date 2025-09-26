"""
Persistent Cache Manager using SQLite and SQLAlchemy
"""

import time
import logging
from typing import Any, Dict, Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from datetime import datetime

from .database import Database
from .models import Base, Workout, Metrics

logger = logging.getLogger(__name__)

class PersistentCacheManager:
    """Persistent cache using SQLite database with TTL support"""
    
    def __init__(self, db: Database, default_ttl: int = 300):
        self.db = db
        self.default_ttl = default_ttl
        self.Session = sessionmaker(bind=self.db.engine)
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set cache entry with TTL (stored in DB)"""
        ttl = ttl or self.default_ttl
        session = self.Session()
        try:
            cache_entry = session.query(CacheEntry).filter_by(key=key).first()
            if cache_entry:
                cache_entry.data = data
                cache_entry.timestamp = time.time()
                cache_entry.ttl = ttl
            else:
                cache_entry = CacheEntry(key=key, data=data, timestamp=time.time(), ttl=ttl)
                session.add(cache_entry)
            session.commit()
            logger.debug(f"Persisted data for key '{key}' with TTL {ttl}s")
        except Exception as e:
            logger.error(f"Error setting cache for '{key}': {e}")
        finally:
            session.close()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cache entry, return default if expired or missing"""
        session = self.Session()
        try:
            cache_entry = session.query(CacheEntry).filter_by(key=key).first()
            if not cache_entry:
                logger.debug(f"Cache miss for key '{key}'")
                return default
            
            if time.time() - cache_entry.timestamp > cache_entry.ttl:
                logger.debug(f"Cache expired for key '{key}'")
                session.delete(cache_entry)
                session.commit()
                return default
            
            logger.debug(f"Cache hit for key '{key}'")
            return cache_entry.data
        except Exception as e:
            logger.error(f"Error getting cache for '{key}': {e}")
            return default
        finally:
            session.close()
    
    def has(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        session = self.Session()
        try:
            cache_entry = session.query(CacheEntry).filter_by(key=key).first()
            if cache_entry:
                session.delete(cache_entry)
                session.commit()
                logger.debug(f"Deleted cache entry for key '{key}'")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting cache for '{key}': {e}")
            return False
        finally:
            session.close()
    
    def clear(self) -> None:
        """Clear all cache entries"""
        session = self.Session()
        try:
            count = session.query(CacheEntry).delete()
            session.commit()
            logger.debug(f"Cleared {count} cache entries")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        finally:
            session.close()
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries and return count removed"""
        session = self.Session()
        try:
            expired_count = session.query(CacheEntry).filter(
                time.time() - CacheEntry.timestamp > CacheEntry.ttl
            ).delete()
            session.commit()
            if expired_count:
                logger.debug(f"Cleaned up {expired_count} expired cache entries")
            return expired_count
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
        finally:
            session.close()
    
    def get_all(self) -> Dict[str, Any]:
        """Get all non-expired cache entries"""
        self.cleanup_expired()
        session = self.Session()
        try:
            entries = session.query(CacheEntry).filter(
                time.time() - CacheEntry.timestamp <= CacheEntry.ttl
            ).all()
            return {entry.key: entry.data for entry in entries}
        except Exception as e:
            logger.error(f"Error getting all cache: {e}")
            return {}
        finally:
            session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self.cleanup_expired()
        session = self.Session()
        try:
            total = session.query(CacheEntry).count()
            active = session.query(CacheEntry).filter(
                time.time() - CacheEntry.timestamp <= CacheEntry.ttl
            ).count()
            return {
                "total_entries": total,
                "active_entries": active,
                "expired_entries": total - active
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
        finally:
            session.close()
    
    def set_multiple(self, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set multiple cache entries at once"""
        session = self.Session()
        try:
            for key, value in data.items():
                self.set(key, value, ttl)
        finally:
            session.close()
    
    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple cache entries at once"""
        return {key: self.get(key) for key in keys}
    
    # Cycling-specific methods (adapted for persistence)
    def cache_user_profile(self, profile_data: Dict[str, Any]) -> None:
        """Cache user profile data"""
        self.set("user_profile", profile_data, ttl=3600)
    
    def cache_activities(self, activities: List[Dict[str, Any]]) -> None:
        """Cache activities list"""
        self.set("recent_activities", activities, ttl=900)
    
    def cache_activity_details(self, activity_id: str, details: Dict[str, Any]) -> None:
        """Cache specific activity details"""
        self.set(f"activity_details_{activity_id}", details, ttl=3600)
    
    def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """Get cached user profile"""
        return self.get("user_profile")
    
    def get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get cached recent activities"""
        return self.get("recent_activities", [])
    
    def get_activity_details(self, activity_id: str) -> Optional[Dict[str, Any]]:
        """Get cached activity details"""
        return self.get(f"activity_details_{activity_id}")
    
    def cache_workout_analysis(self, workout_id: str, analysis: str) -> None:
        """Cache workout analysis results"""
        self.set(f"analysis_{workout_id}", analysis, ttl=86400)
    
    def get_workout_analysis(self, workout_id: str) -> Optional[str]:
        """Get cached workout analysis"""
        return self.get(f"analysis_{workout_id}")
    
    # Enhanced metrics tracking (adapted for DB)
    def set_user_profile_metrics(self, ftp: Optional[float] = None, max_hr: Optional[int] = None):
        """Set user profile for accurate calculations"""
        # Store in DB if needed, but for now, keep in memory for calculator
        self.metrics_calculator = CyclingMetricsCalculator(user_ftp=ftp, user_max_hr=max_hr)
        logger.info(f"Metrics calculator configured: FTP={ftp}, Max HR={max_hr}")
    
    def cache_workout_with_metrics(self, activity_id: str, activity_data: Dict[str, Any]) -> WorkoutMetrics:
        """Cache workout data and calculate comprehensive metrics with validation"""
        if not hasattr(self, 'metrics_calculator') or not self.metrics_calculator:
            self.metrics_calculator = CyclingMetricsCalculator()
        
        # Validate and normalize input data
        validated_data = self._validate_activity_data(activity_data)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_workout_metrics(validated_data)
        
        # Store raw data and metrics in DB
        session = self.Session()
        try:
            # Store raw activity as Workout
            workout = session.query(Workout).filter_by(activity_id=activity_id).first()
            if not workout:
                workout = Workout(activity_id=activity_id)
                session.add(workout)
            
            workout.date = validated_data.get('startTimeGmt', datetime.now())
            workout.data_quality = validated_data.get('data_quality', 'complete')
            workout.is_indoor = validated_data.get('is_indoor', False)
            workout.raw_data = json.dumps(activity_data)
            workout.validation_warnings = json.dumps(validated_data.get('validation_warnings', []))
            
            # Store metrics
            metrics_entry = session.query(Metrics).filter_by(workout_id=activity_id).first()
            if not metrics_entry:
                metrics_entry = Metrics(workout_id=activity_id)
                session.add(metrics_entry)
            
            # Update metrics fields
            for field in ['avg_speed_kmh', 'avg_hr', 'avg_power', 'estimated_ftp', 'duration_minutes', 'distance_km', 'elevation_gain_m', 'training_stress_score', 'intensity_factor', 'workout_classification']:
                if hasattr(metrics, field):
                    setattr(metrics_entry, field, getattr(metrics, field))
            
            session.commit()
            logger.info(f"Persisted workout {activity_id} with calculated metrics")
            return metrics
        except Exception as e:
            logger.error(f"Error caching workout {activity_id}: {e}")
            return metrics
        finally:
            session.close()
    
    def _validate_activity_data(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize activity data for safe metric calculation (from cache_manager)"""
        # Same as in cache_manager
        if not isinstance(activity_data, dict):
            logger.warning("Invalid activity data - creating minimal structure")
            return {"data_quality": "invalid", "summaryDTO": {}}
        
        summary_dto = activity_data.get('summaryDTO', {})
        if not isinstance(summary_dto, dict):
            summary_dto = {}
        
        data_quality = "complete"
        warnings = []
        
        critical_fields = ['duration', 'distance']
        for field in critical_fields:
            if summary_dto.get(field) is None:
                data_quality = "incomplete"
                warnings.append(f"Missing {field}")
                if field == 'duration':
                    summary_dto['duration'] = 0
                elif field == 'distance':
                    summary_dto['distance'] = 0
        
        is_indoor = activity_data.get('is_indoor', False)
        if is_indoor:
            if summary_dto.get('averageSpeed') is None and summary_dto.get('averagePower') is not None:
                summary_dto['averageSpeed'] = None
                warnings.append("Indoor activity - speed estimated from power")
            
            if 'elevationGain' in summary_dto:
                summary_dto['elevationGain'] = 0
                summary_dto['elevationLoss'] = 0
                warnings.append("Indoor activity - elevation set to 0")
        
        expected_fields = [
            'averageSpeed', 'maxSpeed', 'averageHR', 'maxHR', 'averagePower',
            'maxPower', 'normalizedPower', 'trainingStressScore', 'elevationGain',
            'elevationLoss', 'distance', 'duration'
        ]
        for field in expected_fields:
            if field not in summary_dto:
                summary_dto[field] = None
                if data_quality == "complete":
                    data_quality = "incomplete"
                    warnings.append(f"Missing {field}")
        
        activity_data['summaryDTO'] = summary_dto
        activity_data['data_quality'] = data_quality
        activity_data['validation_warnings'] = warnings
        
        if warnings:
            logger.debug(f"Activity validation warnings: {', '.join(warnings)}")
        
        return activity_data
    
    def get_workout_metrics(self, activity_id: str) -> Optional[WorkoutMetrics]:
        """Get calculated metrics for a workout from DB"""
        session = self.Session()
        try:
            metrics_entry = session.query(Metrics).filter_by(workout_id=activity_id).first()
            if metrics_entry:
                return WorkoutMetrics(
                    avg_speed_kmh=metrics_entry.avg_speed_kmh,
                    avg_hr=metrics_entry.avg_hr,
                    avg_power=metrics_entry.avg_power,
                    estimated_ftp=metrics_entry.estimated_ftp,
                    duration_minutes=metrics_entry.duration_minutes,
                    distance_km=metrics_entry.distance_km,
                    elevation_gain_m=metrics_entry.elevation_gain_m,
                    training_stress_score=metrics_entry.training_stress_score,
                    intensity_factor=metrics_entry.intensity_factor,
                    workout_classification=metrics_entry.workout_classification
                )
            return None
        finally:
            session.close()
    
    def get_training_load(self, days: int = 42) -> Optional[TrainingLoad]:
        """Calculate current training load metrics from DB"""
        if not hasattr(self, 'metrics_calculator') or not self.metrics_calculator:
            self.metrics_calculator = CyclingMetricsCalculator()
        
        session = self.Session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_workouts = session.query(Workout).filter(Workout.date >= cutoff_date).all()
            
            if not recent_workouts:
                return None
            
            # Reconstruct activity data
            recent_activity_data = []
            for workout in recent_workouts:
                raw_data = json.loads(workout.raw_data) if workout.raw_data else {}
                recent_activity_data.append(raw_data)
            
            training_load = self.metrics_calculator.calculate_training_load(recent_activity_data)
            
            # Cache in DB if needed, but for now, return
            return training_load
        except Exception as e:
            logger.error(f"Error getting training load: {e}")
            return None
        finally:
            session.close()
    
    # Add other methods as needed, adapting from cache_manager
    # Note: Performance history will be handled via DB queries in future steps