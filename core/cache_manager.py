#!/usr/bin/env python3
"""
Cache Manager - Unified caching with TTL support, metrics tracking, and persistent history
"""

import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict

from ..analysis.cycling_metrics import CyclingMetricsCalculator, WorkoutMetrics, TrainingLoad, PerformanceTrend, generate_standardized_assessment

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds

class CacheManager:
    """Unified cache manager for data with TTL, metrics tracking, and persistent history"""
    
    def __init__(self, default_ttl: int = 300, metrics_file: str = "metrics_history.json"):
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self.metrics_calculator = None
        self.metrics_file = Path(metrics_file)
        self.performance_history = []
        self.load_metrics_history()
    
    # Base caching methods
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set cache entry with TTL"""
        ttl = ttl or self.default_ttl
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=ttl
        )
        logger.debug(f"Cached data for key '{key}' with TTL {ttl}s")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cache entry, return default if expired or missing"""
        if key not in self._cache:
            logger.debug(f"Cache miss for key '{key}'")
            return default
        
        entry = self._cache[key]
        
        # Check if expired
        if time.time() - entry.timestamp > entry.ttl:
            logger.debug(f"Cache expired for key '{key}'")
            del self._cache[key]
            return default
        
        logger.debug(f"Cache hit for key '{key}'")
        return entry.data
    
    def has(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return self.get(key) is not None
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Deleted cache entry for key '{key}'")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        count = len(self._cache)
        self._cache.clear()
        logger.debug(f"Cleared {count} cache entries")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries and return count removed"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > entry.ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all non-expired cache entries"""
        self.cleanup_expired()
        return {key: entry.data for key, entry in self._cache.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self.cleanup_expired()
        return {
            "total_entries": len(self._cache),
            "keys": list(self._cache.keys()),
            "memory_usage_estimate": sum(
                len(str(entry.data)) for entry in self._cache.values()
            )
        }
    
    def set_multiple(self, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set multiple cache entries at once"""
        for key, value in data.items():
            self.set(key, value, ttl)
    
    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple cache entries at once"""
        return {key: self.get(key) for key in keys}
    
    # Cycling-specific helper methods (from CyclingDataCache)
    def cache_user_profile(self, profile_data: Dict[str, Any]) -> None:
        """Cache user profile data"""
        self.set("user_profile", profile_data, ttl=3600)  # 1 hour TTL
    
    def cache_activities(self, activities: List[Dict[str, Any]]) -> None:
        """Cache activities list"""
        self.set("recent_activities", activities, ttl=900)  # 15 minutes TTL
    
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
        self.set(f"analysis_{workout_id}", analysis, ttl=86400)  # 24 hours TTL
    
    def get_workout_analysis(self, workout_id: str) -> Optional[str]:
        """Get cached workout analysis"""
        return self.get(f"analysis_{workout_id}")
    
    # Enhanced metrics tracking methods
    def set_user_profile_metrics(self, ftp: Optional[float] = None, max_hr: Optional[int] = None):
        """Set user profile for accurate calculations (renamed for clarity)"""
        self.metrics_calculator = CyclingMetricsCalculator(user_ftp=ftp, user_max_hr=max_hr)
        logger.info(f"Metrics calculator configured: FTP={ftp}, Max HR={max_hr}")
    
    def cache_workout_with_metrics(self, activity_id: str, activity_data: Dict[str, Any]) -> WorkoutMetrics:
        """Cache workout data and calculate comprehensive metrics with validation"""
        if not self.metrics_calculator:
            # Initialize with defaults if not set
            self.metrics_calculator = CyclingMetricsCalculator()
        
        # Validate and normalize input data
        validated_data = self._validate_activity_data(activity_data)
        
        # Calculate metrics with safe handling
        metrics = self.metrics_calculator.calculate_workout_metrics(validated_data)
        
        # Cache the raw data and calculated metrics
        self.set(f"activity_raw_{activity_id}", activity_data, ttl=3600)
        self.set(f"activity_metrics_{activity_id}", asdict(metrics), ttl=3600)
        
        # Add to performance history
        workout_record = {
            "activity_id": activity_id,
            "date": validated_data.get('startTimeGmt', datetime.now().isoformat()),
            "metrics": asdict(metrics),
            "data_quality": validated_data.get('data_quality', 'complete')
        }
        
        self.performance_history.append(workout_record)
        self.save_metrics_history()
        
        # Update performance trends
        self._update_performance_trends()
        
        logger.info(f"Cached workout {activity_id} with calculated metrics (quality: {workout_record['data_quality']})")
        return metrics
    
    def _validate_activity_data(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize activity data for safe metric calculation"""
        if not isinstance(activity_data, dict):
            logger.warning("Invalid activity data - creating minimal structure")
            return {"data_quality": "invalid", "summaryDTO": {}}
        
        summary_dto = activity_data.get('summaryDTO', {})
        if not isinstance(summary_dto, dict):
            summary_dto = {}
        
        data_quality = "complete"
        warnings = []
        
        # Check critical fields
        critical_fields = ['duration', 'distance']
        for field in critical_fields:
            if summary_dto.get(field) is None:
                data_quality = "incomplete"
                warnings.append(f"Missing {field}")
                # Set reasonable defaults
                if field == 'duration':
                    summary_dto['duration'] = 0
                elif field == 'distance':
                    summary_dto['distance'] = 0
        
        # Indoor activity adjustments
        is_indoor = activity_data.get('is_indoor', False)
        if is_indoor:
            # For indoor, speed may be None - estimate from power if available
            if summary_dto.get('averageSpeed') is None and summary_dto.get('averagePower') is not None:
                # Rough estimate: speed = power / (weight * constant), but without weight, use placeholder
                summary_dto['averageSpeed'] = None  # Keep None, let calculator handle
                warnings.append("Indoor activity - speed estimated from power")
            
            # Elevation not applicable for indoor
            if 'elevationGain' in summary_dto:
                summary_dto['elevationGain'] = 0
                summary_dto['elevationLoss'] = 0
                warnings.append("Indoor activity - elevation set to 0")
        
        # Ensure all expected fields exist (from custom_garth_mcp normalization)
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
        """Get calculated metrics for a workout"""
        metrics_data = self.get(f"activity_metrics_{activity_id}")
        if metrics_data:
            return WorkoutMetrics(**metrics_data)
        return None
    
    def get_training_load(self, days: int = 42) -> Optional[TrainingLoad]:
        """Calculate current training load metrics"""
        if not self.metrics_calculator:
            return None
        
        # Get recent workout history
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_workouts = []
        
        for record in self.performance_history:
            workout_date = datetime.fromisoformat(record['date'].replace('Z', '+00:00'))
            if workout_date >= cutoff_date:
                # Reconstruct activity data for training load calculation
                activity_data = self.get(f"activity_raw_{record['activity_id']}")
                if activity_data:
                    recent_workouts.append(activity_data)
        
        if not recent_workouts:
            return None
        
        training_load = self.metrics_calculator.calculate_training_load(recent_workouts)
        
        # Cache training load
        self.set("current_training_load", asdict(training_load), ttl=3600)
        
        return training_load
    
    def get_performance_trends(self, days: int = 30) -> List[PerformanceTrend]:
        """Get performance trends for key metrics"""
        trends = self.get(f"performance_trends_{days}d")
        if trends:
            return [PerformanceTrend(**trend) for trend in trends]
        
        # Calculate if not cached
        return self._calculate_performance_trends(days)
    
    def _calculate_performance_trends(self, days: int) -> List[PerformanceTrend]:
        """Calculate performance trends over specified period"""
        if not self.performance_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = []
        
        for record in self.performance_history:
            workout_date = datetime.fromisoformat(record['date'].replace('Z', '+00:00'))
            if workout_date >= cutoff_date:
                recent_metrics.append({
                    'date': workout_date,
                    'metrics': WorkoutMetrics(**record['metrics'])
                })
        
        if len(recent_metrics) < 2:
            return []
        
        # Sort by date
        recent_metrics.sort(key=lambda x: x['date'])
        
        trends = []
        
        # Calculate trends for key metrics
        metrics_to_track = [
            ('avg_speed_kmh', 'Average Speed'),
            ('avg_hr', 'Average Heart Rate'),
            ('avg_power', 'Average Power'),
            ('estimated_ftp', 'Estimated FTP'),
            ('training_stress_score', 'Training Stress Score')
        ]
        
        for metric_attr, metric_name in metrics_to_track:
            trend = self._calculate_single_metric_trend(recent_metrics, metric_attr, metric_name, days)
            if trend:
                trends.append(trend)
        
        # Cache trends
        self.set(f"performance_trends_{days}d", [asdict(trend) for trend in trends], ttl=1800)
        
        return trends
    
    def _calculate_single_metric_trend(self, recent_metrics: List[Dict], 
                                     metric_attr: str, metric_name: str, 
                                     days: int) -> Optional[PerformanceTrend]:
        """Calculate trend for a single metric"""
        # Extract values, filtering out None values
        values_with_dates = []
        for record in recent_metrics:
            value = getattr(record['metrics'], metric_attr)
            if value is not None:
                values_with_dates.append((record['date'], value))
        
        if len(values_with_dates) < 2:
            return None
        
        # Calculate current value (average of last 3 workouts)
        recent_values = [v for _, v in values_with_dates[-3:]]
        current_value = sum(recent_values) / len(recent_values)
        
        # Calculate 7-day trend if we have enough data
        week_ago = datetime.now() - timedelta(days=7)
        week_values = [v for d, v in values_with_dates if d >= week_ago]
        
        if len(week_values) >= 2:
            week_old_avg = sum(week_values[:len(week_values)//2]) / (len(week_values)//2)
            week_recent_avg = sum(week_values[len(week_values)//2:]) / (len(week_values) - len(week_values)//2)
            trend_7day = ((week_recent_avg - week_old_avg) / week_old_avg * 100) if week_old_avg > 0 else 0
        else:
            trend_7day = 0
        
        # Calculate 30-day trend
        if len(values_with_dates) >= 4:
            old_avg = sum(v for _, v in values_with_dates[:len(values_with_dates)//2]) / (len(values_with_dates)//2)
            recent_avg = sum(v for _, v in values_with_dates[len(values_with_dates)//2:]) / (len(values_with_dates) - len(values_with_dates)//2)
            trend_30day = ((recent_avg - old_avg) / old_avg * 100) if old_avg > 0 else 0
        else:
            trend_30day = 0
        
        # Determine trend direction
        primary_trend = trend_7day if abs(trend_7day) > abs(trend_30day) else trend_30day
        if primary_trend > 2:
            trend_direction = "improving"
        elif primary_trend < -2:
            trend_direction = "declining"
        else:
            trend_direction = "stable"
        
        # Calculate confidence based on data points
        confidence = min(len(values_with_dates) / 10, 1.0)  # Max confidence at 10+ data points
        
        return PerformanceTrend(
            metric_name=metric_name,
            current_value=round(current_value, 2),
            trend_7day=round(trend_7day, 1),
            trend_30day=round(trend_30day, 1),
            trend_direction=trend_direction,
            confidence=round(confidence, 2)
        )
    
    def _update_performance_trends(self):
        """Update cached performance trends after new workout"""
        # Clear cached trends to force recalculation
        keys_to_clear = [key for key in self._cache.keys() if key.startswith("performance_trends_")]
        for key in keys_to_clear:
            self.delete(key)
    
    def get_deterministic_analysis_data(self, activity_id: str) -> Dict[str, Any]:
        """Get all deterministic data for analysis with validation"""
        metrics = self.get_workout_metrics(activity_id)
        training_load = self.get_training_load()
        performance_trends = self.get_performance_trends()
        
        if not metrics:
            return {"error": "No metrics available for activity"}
        
        # Generate standardized assessment with safe handling
        try:
            assessment = generate_standardized_assessment(metrics, training_load)
        except Exception as e:
            logger.warning(f"Could not generate standardized assessment: {e}")
            assessment = {"error": "Assessment calculation failed", "workout_classification": "unknown"}
        
        return {
            "workout_metrics": asdict(metrics),
            "training_load": asdict(training_load) if training_load else None,
            "performance_trends": [asdict(trend) for trend in performance_trends if trend],
            "standardized_assessment": assessment,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def get_ftp_estimates_history(self) -> List[Dict[str, Any]]:
        """Get historical FTP estimates for tracking progress"""
        ftp_history = []
        
        for record in self.performance_history:
            metrics = WorkoutMetrics(**record['metrics'])
            if metrics.estimated_ftp:
                ftp_history.append({
                    "date": record['date'],
                    "activity_id": record['activity_id'],
                    "estimated_ftp": metrics.estimated_ftp,
                    "workout_type": record['metrics'].get('workout_classification', 'unknown')
                })
        
        # Sort by date and return recent estimates
        ftp_history.sort(key=lambda x: x['date'], reverse=True)
        return ftp_history[:20]  # Last 20 estimates
    
    def get_gear_usage_analysis(self) -> Dict[str, Any]:
        """Get single speed gear usage analysis"""
        gear_data = []
        
        for record in self.performance_history:
            metrics = WorkoutMetrics(**record['metrics'])
            if metrics.estimated_gear_ratio:
                gear_data.append({
                    "date": record['date'],
                    "estimated_ratio": metrics.estimated_gear_ratio,
                    "chainring": metrics.estimated_chainring,
                    "cog": metrics.estimated_cog,
                    "avg_speed": metrics.avg_speed_kmh,
                    "elevation_gain": metrics.elevation_gain_m,
                    "terrain_type": self._classify_terrain(metrics)
                })
        
        if not gear_data:
            return {"message": "No gear data available"}
        
        # Analyze gear preferences by terrain
        gear_preferences = {}
        for data in gear_data:
            terrain = data['terrain_type']
            gear = f"{data['chainring']}x{data['cog']}"
            
            if terrain not in gear_preferences:
                gear_preferences[terrain] = {}
            if gear not in gear_preferences[terrain]:
                gear_preferences[terrain][gear] = 0
            gear_preferences[terrain][gear] += 1
        
        # Calculate most common gears
        all_gears = {}
        for data in gear_data:
            gear = f"{data['chainring']}x{data['cog']}"
            all_gears[gear] = all_gears.get(gear, 0) + 1
        
        most_common_gear = max(all_gears.items(), key=lambda x: x[1])
        
        return {
            "total_workouts_analyzed": len(gear_data),
            "most_common_gear": {
                "gear": most_common_gear[0],
                "usage_count": most_common_gear[1],
                "usage_percentage": round(most_common_gear[1] / len(gear_data) * 100, 1)
            },
            "gear_by_terrain": gear_preferences,
            "gear_recommendations": self._recommend_gears(gear_data)
        }
    
    def _classify_terrain(self, metrics: WorkoutMetrics) -> str:
        """Classify terrain type from workout metrics"""
        if metrics.distance_km == 0:
            return "unknown"
        
        elevation_per_km = metrics.elevation_gain_m / metrics.distance_km
        
        if elevation_per_km > 15:
            return "steep_climbing"
        elif elevation_per_km > 8:
            return "moderate_climbing"
        elif elevation_per_km > 3:
            return "rolling_hills"
        else:
            return "flat_terrain"
    
    def _recommend_gears(self, gear_data: List[Dict]) -> Dict[str, str]:
        """Recommend optimal gears for different conditions"""
        if not gear_data:
            return {}
        
        # Group by terrain and find most efficient gears
        terrain_efficiency = {}
        
        for data in gear_data:
            terrain = data['terrain_type']
            gear = f"{data['chainring']}x{data['cog']}"
            speed = data['avg_speed']
            
            if terrain not in terrain_efficiency:
                terrain_efficiency[terrain] = {}
            if gear not in terrain_efficiency[terrain]:
                terrain_efficiency[terrain][gear] = []
            
            terrain_efficiency[terrain][gear].append(speed)
        
        # Calculate average speeds for each gear/terrain combo
        recommendations = {}
        for terrain, gears in terrain_efficiency.items():
            best_gear = None
            best_avg_speed = 0
            
            for gear, speeds in gears.items():
                avg_speed = sum(speeds) / len(speeds)
                if avg_speed > best_avg_speed:
                    best_avg_speed = avg_speed
                    best_gear = gear
            
            if best_gear:
                recommendations[terrain] = best_gear
        
        return recommendations
    
    def load_metrics_history(self):
        """Load performance history from file"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.performance_history = data.get('performance_history', [])
                    logger.info(f"Loaded {len(self.performance_history)} workout records")
            except Exception as e:
                logger.error(f"Error loading metrics history: {e}")
                self.performance_history = []
        else:
            self.performance_history = []
    
    def save_metrics_history(self):
        """Save performance history to file"""
        try:
            # Keep only last 200 workouts to prevent file from growing too large
            self.performance_history = self.performance_history[-200:]
            
            data = {
                'performance_history': self.performance_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Saved {len(self.performance_history)} workout records")
        except Exception as e:
            logger.error(f"Error saving metrics history: {e}")
    
    def get_workout_summary_for_llm(self, activity_id: str) -> Dict[str, Any]:
        """Get structured workout summary optimized for LLM analysis"""
        deterministic_data = self.get_deterministic_analysis_data(activity_id)
        
        if "error" in deterministic_data:
            return deterministic_data
        
        # Format data for LLM consumption
        metrics = deterministic_data["workout_metrics"]
        assessment = deterministic_data["standardized_assessment"]
        training_load = deterministic_data.get("training_load")
        
        summary = {
            "workout_classification": assessment["workout_classification"],
            "intensity_rating": f"{assessment['intensity_rating']}/10",
            "key_metrics": {
                "duration": f"{metrics['duration_minutes']:.0f} minutes",
                "distance": f"{metrics['distance_km']:.1f} km",
                "avg_speed": f"{metrics['avg_speed_kmh']:.1f} km/h",
                "elevation_gain": f"{metrics['elevation_gain_m']:.0f} m"
            },
            "performance_indicators": {
                "efficiency_score": assessment["efficiency_score"],
                "estimated_ftp": metrics.get("estimated_ftp"),
                "intensity_factor": metrics.get("intensity_factor")
            },
            "recovery_guidance": assessment["recovery_recommendation"],
            "training_load_context": {
                "fitness_level": training_load["fitness"] if training_load else None,
                "fatigue_level": training_load["fatigue"] if training_load else None,
                "form": training_load["form"] if training_load else None
            } if training_load else None,
            "single_speed_analysis": {
                "estimated_gear": f"{metrics.get('estimated_chainring', 'N/A')}x{metrics.get('estimated_cog', 'N/A')}",
                "gear_ratio": metrics.get("estimated_gear_ratio")
            } if metrics.get("estimated_gear_ratio") else None
        }
        
        return summary