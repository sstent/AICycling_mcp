#!/usr/bin/env python3
"""
Cycling Metrics Calculator - Deterministic metrics for cycling workouts
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class WorkoutMetrics:
    """Standardized workout metrics"""
    # Basic metrics
    duration_minutes: float
    distance_km: float
    avg_speed_kmh: float
    max_speed_kmh: float
    elevation_gain_m: float
    
    # Heart rate metrics (if available)
    avg_hr: Optional[float] = None
    max_hr: Optional[float] = None
    hr_zones: Optional[Dict[str, float]] = None  # Time in each zone
    
    # Power metrics (if available)
    avg_power: Optional[float] = None
    max_power: Optional[float] = None
    normalized_power: Optional[float] = None
    power_zones: Optional[Dict[str, float]] = None
    
    # Calculated metrics
    intensity_factor: Optional[float] = None
    training_stress_score: Optional[float] = None
    estimated_ftp: Optional[float] = None
    variability_index: Optional[float] = None
    
    # Single speed specific
    estimated_gear_ratio: Optional[float] = None
    estimated_chainring: Optional[int] = None
    estimated_cog: Optional[int] = None
    gear_usage_distribution: Optional[Dict[str, float]] = None

@dataclass
class TrainingLoad:
    """Training load metrics over time"""
    acute_training_load: float  # 7-day average
    chronic_training_load: float  # 42-day average
    training_stress_balance: float  # CTL - ATL
    fitness: float  # Chronic Training Load
    fatigue: float  # Acute Training Load
    form: float  # Training Stress Balance

class CyclingMetricsCalculator:
    """Calculate deterministic cycling metrics"""
    
    def __init__(self, user_ftp: Optional[float] = None, user_max_hr: Optional[int] = None):
        self.user_ftp = user_ftp
        self.user_max_hr = user_max_hr
        
        # Single speed gear options
        self.chainrings = [46, 38]  # teeth
        self.cogs = [14, 15, 16, 17, 18, 19, 20]  # teeth
        self.wheel_circumference_m = 2.096  # 700x25c wheel circumference in meters
    
    def calculate_workout_metrics(self, activity_data: Dict[str, Any]) -> WorkoutMetrics:
        """Calculate comprehensive metrics for a workout"""
        # Extract basic data
        duration_seconds = activity_data.get('duration', 0)
        duration_minutes = duration_seconds / 60.0
        
        distance_m = activity_data.get('distance', 0)
        distance_km = distance_m / 1000.0
        
        avg_speed_ms = activity_data.get('averageSpeed', 0)
        avg_speed_kmh = avg_speed_ms * 3.6
        
        max_speed_ms = activity_data.get('maxSpeed', 0)
        max_speed_kmh = max_speed_ms * 3.6
        
        elevation_gain = activity_data.get('elevationGain', 0)
        
        # Heart rate data
        avg_hr = activity_data.get('averageHR')
        max_hr = activity_data.get('maxHR')
        
        # Power data
        avg_power = activity_data.get('avgPower')
        max_power = activity_data.get('maxPower')
        
        # Calculate derived metrics
        metrics = WorkoutMetrics(
            duration_minutes=duration_minutes,
            distance_km=distance_km,
            avg_speed_kmh=avg_speed_kmh,
            max_speed_kmh=max_speed_kmh,
            elevation_gain_m=elevation_gain,
            avg_hr=avg_hr,
            max_hr=max_hr,
            avg_power=avg_power,
            max_power=max_power
        )
        
        # Calculate advanced metrics if power data available
        if avg_power and self.user_ftp:
            metrics.intensity_factor = avg_power / self.user_ftp
            metrics.training_stress_score = self._calculate_tss(duration_minutes, avg_power, self.user_ftp)
            
            if max_power and avg_power:
                metrics.variability_index = max_power / avg_power
        
        # Estimate FTP if no power meter but have HR data
        if not self.user_ftp and avg_hr and max_hr:
            metrics.estimated_ftp = self._estimate_ftp_from_hr(avg_hr, max_hr, duration_minutes, distance_km, elevation_gain)
        
        # Calculate gear ratios for single speed
        if avg_speed_kmh > 0:
            gear_analysis = self._analyze_single_speed_gears(avg_speed_kmh, duration_minutes, elevation_gain)
            metrics.estimated_gear_ratio = gear_analysis['estimated_ratio']
            metrics.estimated_chainring = gear_analysis['estimated_chainring']
            metrics.estimated_cog = gear_analysis['estimated_cog']
            metrics.gear_usage_distribution = gear_analysis['gear_distribution']
        
        return metrics
    
    def _calculate_tss(self, duration_minutes: float, avg_power: float, ftp: float) -> float:
        """Calculate Training Stress Score"""
        if_factor = avg_power / ftp
        tss = (duration_minutes * avg_power * if_factor) / (ftp * 60) * 100
        return round(tss, 1)
    
    def _estimate_ftp_from_hr(self, avg_hr: float, max_hr: float, duration_minutes: float, 
                             distance_km: float, elevation_gain: float) -> float:
        """Estimate FTP from heart rate and performance data"""
        # Basic estimation using heart rate zones and performance
        # This is a simplified model - real FTP estimation requires more sophisticated analysis
        
        # Calculate relative intensity from HR
        if self.user_max_hr:
            hr_intensity = avg_hr / self.user_max_hr
        else:
            # Estimate max HR using age formula (less accurate)
            estimated_max_hr = 220 - 30  # Assuming 30 years old, should be configurable
            hr_intensity = avg_hr / estimated_max_hr
        
        # Calculate speed-based power estimate
        # This is very rough and assumes flat terrain
        avg_speed_ms = (distance_km * 1000) / (duration_minutes * 60)
        
        # Rough power estimation based on speed (watts = speed^3 * factor)
        # Adjusted for elevation gain
        elevation_factor = 1 + (elevation_gain / distance_km / 1000) * 0.1
        estimated_power = (avg_speed_ms ** 2.5) * 3.5 * elevation_factor
        
        # Estimate FTP as power at ~75% max HR
        ftp_ratio = 0.75 / hr_intensity if hr_intensity > 0.75 else 1.0
        estimated_ftp = estimated_power * ftp_ratio
        
        return round(estimated_ftp, 0)
    
    def _analyze_single_speed_gears(self, avg_speed_kmh: float, duration_minutes: float, 
                                   elevation_gain: float) -> Dict[str, Any]:
        """Analyze single speed gear usage"""
        # Calculate average cadence assumption (80-90 RPM is typical)
        assumed_cadence = 85  # RPM
        
        # Calculate required gear ratio for average speed
        speed_ms = avg_speed_kmh / 3.6
        distance_per_pedal_revolution = speed_ms * 60 / assumed_cadence  # meters per revolution
        required_gear_ratio = distance_per_pedal_revolution / self.wheel_circumference_m
        
        # Find best matching gear combinations
        gear_options = []
        for chainring in self.chainrings:
            for cog in self.cogs:
                ratio = chainring / cog
                ratio_error = abs(ratio - required_gear_ratio) / required_gear_ratio
                gear_options.append({
                    'chainring': chainring,
                    'cog': cog,
                    'ratio': ratio,
                    'error': ratio_error
                })
        
        # Sort by best match
        gear_options.sort(key=lambda x: x['error'])
        best_gear = gear_options[0]
        
        # Estimate gear usage distribution based on terrain
        gear_distribution = self._estimate_gear_distribution(elevation_gain, duration_minutes, gear_options)
        
        return {
            'estimated_ratio': best_gear['ratio'],
            'estimated_chainring': best_gear['chainring'],
            'estimated_cog': best_gear['cog'],
            'gear_distribution': gear_distribution,
            'all_options': gear_options[:3]  # Top 3 matches
        }
    
    def _estimate_gear_distribution(self, elevation_gain: float, duration_minutes: float, 
                                   gear_options: List[Dict]) -> Dict[str, float]:
        """Estimate how much time was spent in each gear"""
        # Simplified model based on elevation profile
        climbing_factor = elevation_gain / (duration_minutes * 10)  # rough climbing intensity
        
        distribution = {}
        for gear in gear_options[:4]:  # Top 4 gears
            gear_name = f"{gear['chainring']}x{gear['cog']}"
            
            if climbing_factor > 2.0:  # Lots of climbing
                # Favor easier gears (lower ratios)
                weight = 1.0 / gear['ratio']
            elif climbing_factor < 0.5:  # Mostly flat
                # Favor harder gears (higher ratios)
                weight = gear['ratio']
            else:
                # Mixed terrain
                weight = 1.0
            
            distribution[gear_name] = weight
        
        # Normalize to percentages
        total_weight = sum(distribution.values())
        if total_weight > 0:
            distribution = {k: round(v / total_weight * 100, 1) for k, v in distribution.items()}
        
        return distribution
    
    def calculate_training_load(self, workout_history: List[Dict[str, Any]], 
                               current_date: datetime = None) -> TrainingLoad:
        """Calculate training load metrics"""
        if not current_date:
            current_date = datetime.now()
        
        # Calculate TSS for each workout
        tss_by_date = {}
        for workout in workout_history:
            workout_date = datetime.fromisoformat(workout.get('startTimeGmt', '').replace('Z', '+00:00'))
            metrics = self.calculate_workout_metrics(workout)
            tss = metrics.training_stress_score or self._estimate_tss_without_power(metrics)
            tss_by_date[workout_date.date()] = tss
        
        # Calculate Acute Training Load (7-day average)
        atl_days = 7
        atl_total = 0
        atl_count = 0
        for i in range(atl_days):
            date = (current_date - timedelta(days=i)).date()
            if date in tss_by_date:
                atl_total += tss_by_date[date]
                atl_count += 1
        
        atl = atl_total / atl_days if atl_count > 0 else 0
        
        # Calculate Chronic Training Load (42-day average)
        ctl_days = 42
        ctl_total = 0
        ctl_count = 0
        for i in range(ctl_days):
            date = (current_date - timedelta(days=i)).date()
            if date in tss_by_date:
                ctl_total += tss_by_date[date]
                ctl_count += 1
        
        ctl = ctl_total / ctl_days if ctl_count > 0 else 0
        
        # Training Stress Balance
        tsb = ctl - atl
        
        return TrainingLoad(
            acute_training_load=round(atl, 1),
            chronic_training_load=round(ctl, 1),
            training_stress_balance=round(tsb, 1),
            fitness=round(ctl, 1),
            fatigue=round(atl, 1),
            form=round(tsb, 1)
        )
    
    def _estimate_tss_without_power(self, metrics: WorkoutMetrics) -> float:
        """Estimate TSS without power data using HR and duration"""
        if metrics.avg_hr and self.user_max_hr:
            # Use TRIMP method as TSS proxy
            hr_ratio = metrics.avg_hr / self.user_max_hr
            duration_hours = metrics.duration_minutes / 60
            
            # Simplified TSS estimation
            estimated_tss = duration_hours * 60 * (hr_ratio ** 1.92)
            return round(estimated_tss, 1)
        else:
            # Very rough estimation based on duration and intensity
            duration_hours = metrics.duration_minutes / 60
            speed_factor = min(metrics.avg_speed_kmh / 25, 2.0)  # Cap at 2x for high speeds
            elevation_factor = 1 + (metrics.elevation_gain_m / (metrics.distance_km * 1000) * 0.1)
            
            estimated_tss = duration_hours * 40 * speed_factor * elevation_factor
            return round(estimated_tss, 1)
    
    def get_performance_trends(self, workout_history: List[Dict[str, Any]], 
                              days: int = 30) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_workouts = []
        for workout in workout_history:
            workout_date = datetime.fromisoformat(workout.get('startTimeGmt', '').replace('Z', '+00:00'))
            if workout_date >= cutoff_date:
                recent_workouts.append(workout)
        
        if not recent_workouts:
            return {"error": "No recent workouts found"}
        
        # Calculate metrics for each workout
        metrics_list = [self.calculate_workout_metrics(w) for w in recent_workouts]
        
        # Calculate trends
        avg_speed_trend = [m.avg_speed_kmh for m in metrics_list]
        avg_hr_trend = [m.avg_hr for m in metrics_list if m.avg_hr]
        avg_power_trend = [m.avg_power for m in metrics_list if m.avg_power]
        
        return {
            "period_days": days,
            "total_workouts": len(recent_workouts),
            "avg_speed": {
                "current": round(sum(avg_speed_trend) / len(avg_speed_trend), 1),
                "max": round(max(avg_speed_trend), 1),
                "min": round(min(avg_speed_trend), 1),
                "trend": "improving" if len(avg_speed_trend) > 1 and avg_speed_trend[-1] > avg_speed_trend[0] else "stable"
            },
            "avg_heart_rate": {
                "current": round(sum(avg_hr_trend) / len(avg_hr_trend), 1) if avg_hr_trend else None,
                "trend": "improving" if len(avg_hr_trend) > 1 and avg_hr_trend[-1] < avg_hr_trend[0] else "stable"
            } if avg_hr_trend else None,
            "power_data_available": len(avg_power_trend) > 0,
            "estimated_fitness_change": self._calculate_fitness_change(metrics_list)
        }
    
    def _calculate_fitness_change(self, metrics_list: List[WorkoutMetrics]) -> str:
        """Calculate estimated fitness change"""
        if len(metrics_list) < 3:
            return "insufficient_data"
        
        # Look at speed and HR efficiency
        recent_metrics = metrics_list[-3:]  # Last 3 workouts
        older_metrics = metrics_list[:3] if len(metrics_list) >= 6 else metrics_list[:-3]
        
        if not older_metrics:
            return "insufficient_data"
        
        recent_speed_avg = sum(m.avg_speed_kmh for m in recent_metrics) / len(recent_metrics)
        older_speed_avg = sum(m.avg_speed_kmh for m in older_metrics) / len(older_metrics)
        
        speed_improvement = (recent_speed_avg - older_speed_avg) / older_speed_avg * 100
        
        if speed_improvement > 5:
            return "improving"
        elif speed_improvement < -5:
            return "declining"
        else:
            return "stable"

# Deterministic analysis helper
def generate_standardized_assessment(metrics: WorkoutMetrics, 
                                   training_load: TrainingLoad = None) -> Dict[str, Any]:
    """Generate standardized, deterministic workout assessment"""
    assessment = {
        "workout_classification": classify_workout(metrics),
        "intensity_rating": rate_intensity(metrics),
        "efficiency_score": calculate_efficiency_score(metrics),
        "recovery_recommendation": recommend_recovery(metrics, training_load),
        "key_metrics_summary": summarize_key_metrics(metrics)
    }
    
    return assessment

def classify_workout(metrics: WorkoutMetrics) -> str:
    """Classify workout type based on metrics"""
    duration = metrics.duration_minutes
    avg_speed = metrics.avg_speed_kmh
    elevation_gain = metrics.elevation_gain_m / metrics.distance_km if metrics.distance_km > 0 else 0
    
    if duration < 30:
        return "short_intensity"
    elif duration > 180:
        return "long_endurance"
    elif elevation_gain > 10:  # >10m elevation per km
        return "climbing_focused"
    elif avg_speed > 35:
        return "high_speed"
    elif avg_speed < 20:
        return "recovery_easy"
    else:
        return "moderate_endurance"

def rate_intensity(metrics: WorkoutMetrics) -> int:
    """Rate workout intensity 1-10"""
    factors = []
    
    # Speed factor
    if metrics.avg_speed_kmh > 40:
        factors.append(9)
    elif metrics.avg_speed_kmh > 35:
        factors.append(7)
    elif metrics.avg_speed_kmh > 25:
        factors.append(5)
    else:
        factors.append(3)
    
    # Duration factor
    duration_intensity = min(metrics.duration_minutes / 60 * 2, 6)
    factors.append(duration_intensity)
    
    # Elevation factor
    if metrics.distance_km > 0:
        elevation_per_km = metrics.elevation_gain_m / metrics.distance_km
        if elevation_per_km > 15:
            factors.append(8)
        elif elevation_per_km > 10:
            factors.append(6)
        elif elevation_per_km > 5:
            factors.append(4)
        else:
            factors.append(2)
    
    # HR factor (if available)
    if metrics.avg_hr and metrics.max_hr:
        hr_ratio = metrics.avg_hr / metrics.max_hr
        if hr_ratio > 0.85:
            factors.append(9)
        elif hr_ratio > 0.75:
            factors.append(7)
        elif hr_ratio > 0.65:
            factors.append(5)
        else:
            factors.append(3)
    
    return min(int(sum(factors) / len(factors)), 10)

def calculate_efficiency_score(metrics: WorkoutMetrics) -> float:
    """Calculate efficiency score (higher = more efficient)"""
    # Speed per heart rate beat (if HR available)
    if metrics.avg_hr and metrics.avg_hr > 0:
        speed_hr_efficiency = metrics.avg_speed_kmh / metrics.avg_hr * 100
        return round(speed_hr_efficiency, 2)
    else:
        # Fallback: speed per elevation gain
        if metrics.elevation_gain_m > 0:
            speed_elevation_efficiency = metrics.avg_speed_kmh / (metrics.elevation_gain_m / 100)
            return round(speed_elevation_efficiency, 2)
        else:
            return metrics.avg_speed_kmh  # Just speed as efficiency

def recommend_recovery(metrics: WorkoutMetrics, training_load: TrainingLoad = None) -> str:
    """Recommend recovery based on workout intensity"""
    intensity = rate_intensity(metrics)
    
    if training_load and training_load.training_stress_balance < -10:
        return "high_fatigue_rest_recommended"
    elif intensity >= 8:
        return "24_48_hours_easy"
    elif intensity >= 6:
        return "24_hours_easy"
    elif intensity >= 4:
        return "active_recovery_optional"
    else:
        return "ready_for_next_workout"

def summarize_key_metrics(metrics: WorkoutMetrics) -> Dict[str, str]:
    """Summarize key metrics in human readable format"""
    summary = {
        "duration": f"{metrics.duration_minutes:.0f} minutes",
        "distance": f"{metrics.distance_km:.1f} km",
        "avg_speed": f"{metrics.avg_speed_kmh:.1f} km/h",
        "elevation_gain": f"{metrics.elevation_gain_m:.0f} m"
    }
    
    if metrics.avg_hr:
        summary["avg_heart_rate"] = f"{metrics.avg_hr:.0f} bpm"
    
    if metrics.avg_power:
        summary["avg_power"] = f"{metrics.avg_power:.0f} W"
    
    if metrics.estimated_ftp:
        summary["estimated_ftp"] = f"{metrics.estimated_ftp:.0f} W"
    
    if metrics.estimated_gear_ratio:
        summary["estimated_gear"] = f"{metrics.estimated_chainring}x{metrics.estimated_cog} ({metrics.estimated_gear_ratio:.1f} ratio)"
    
    return summary