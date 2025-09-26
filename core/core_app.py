#!/usr/bin/env python3
"""
Core Application - Unified orchestrator with enhanced metrics and analysis
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ..config import Config, load_config
from ..llm.llm_client import LLMClient
from ..mcp.mcp_client import MCPClient
from .cache_manager import CacheManager
from .template_engine import TemplateEngine
from ..analysis.cycling_metrics import CyclingMetricsCalculator, generate_standardized_assessment, WorkoutMetrics, TrainingLoad, PerformanceTrend

logger = logging.getLogger(__name__)

class CyclingAnalyzerApp:
    """Unified main application class - orchestrates all components with enhanced metrics support"""
    
    def __init__(self, config: Config, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode
        self.llm_client = LLMClient(config)
        self.mcp_client = MCPClient(config)
        
        # Use unified cache manager with metrics tracking
        self.cache_manager = CacheManager(
            default_ttl=config.cache_ttl if hasattr(config, 'cache_ttl') else 300,
            metrics_file="workout_metrics.json"
        )
        
        self.template_engine = TemplateEngine(config.templates_dir)
        
        # User settings for accurate calculations
        self.user_ftp = None
        self.user_max_hr = None
        
        logger.info("DEBUG: Cache contents after init:")
        for key in ["user_profile", "last_cycling_details"]:
            data = self.cache_manager.get(key, {})
            logger.info(f"  {key}: keys={list(data.keys()) if data else 'EMPTY'}, length={len(data) if data else 0}")
    
    async def initialize(self):
        """Initialize all components with metrics support"""
        logger.info("Initializing application components...")
        
        await self.llm_client.initialize()
        await self.mcp_client.initialize()
        await self._setup_user_metrics()
        await self._preload_cache_with_metrics()
        
        logger.info("Application initialization complete")
    
    async def cleanup(self):
        """Cleanup all components"""
        # Save metrics before cleanup
        self.cache_manager.save_metrics_history()
        
        await self.mcp_client.cleanup()
        await self.llm_client.cleanup()
    
    async def _setup_user_metrics(self):
        """Setup user profile for accurate metric calculations"""
        try:
            # Try to get user profile from MCP
            if await self.mcp_client.has_tool("user_profile"):
                profile = await self.mcp_client.call_tool("user_profile", {})
                
                # Extract FTP and max HR if available
                self.user_ftp = profile.get("ftp") or profile.get("functionalThresholdPower")
                self.user_max_hr = profile.get("maxHR") or profile.get("maxHeartRate")
                
                # Also try user settings
                if await self.mcp_client.has_tool("user_settings"):
                    settings = await self.mcp_client.call_tool("user_settings", {})
                    if not self.user_ftp:
                        self.user_ftp = settings.get("ftp")
                    if not self.user_max_hr:
                        self.user_max_hr = settings.get("maxHeartRate")
                
                logger.info(f"User metrics configured: FTP={self.user_ftp}W, Max HR={self.user_max_hr}bpm")
            
            # Set up cache manager with user profile
            self.cache_manager.set_user_profile(ftp=self.user_ftp, max_hr=self.user_max_hr)
            
        except Exception as e:
            logger.warning(f"Could not setup user metrics: {e}")
            # Initialize with defaults
            self.cache_manager.set_user_profile()
    
    async def _preload_cache_with_metrics(self):
        """Pre-load cache with calculated metrics (enhanced version of _preload_cache)"""
        logger.info("Pre-loading cache with metrics calculation...")
        
        try:
            # Cache user profile (from base)
            if await self.mcp_client.has_tool("user_profile"):
                profile = await self.mcp_client.call_tool("user_profile", {})
                self.cache_manager.set("user_profile", profile)
            
            # Cache recent activities
            if await self.mcp_client.has_tool("get_activities"):
                activities = await self.mcp_client.call_tool("get_activities", {"limit": 15})
                self.cache_manager.set("recent_activities", activities)
                
                # Process cycling activities with metrics
                cycling_count = 0
                for activity in activities:
                    activity_type = activity.get("activityType", {})
                    if isinstance(activity_type, dict):
                        type_key = activity_type.get("typeKey", "").lower()
                    else:
                        type_key = str(activity_type).lower()
                    
                    if "cycling" in type_key or "bike" in type_key:
                        activity_id = activity.get("activityId")
                        if activity_id and cycling_count < 5:  # Limit to 5 recent cycling activities
                            try:
                                # Get detailed activity data
                                if await self.mcp_client.has_tool("get_activity_details"):
                                    details = await self.mcp_client.call_tool(
                                        "get_activity_details", 
                                        {"activity_id": str(activity_id)}
                                    )
                                    
                                    # Calculate and cache metrics
                                    metrics = self.cache_manager.cache_workout_with_metrics(
                                        str(activity_id), details
                                    )
                                    
                                    logger.info(f"Processed activity {activity_id}: {metrics.workout_classification}")
                                    cycling_count += 1
                                    
                            except Exception as e:
                                logger.warning(f"Could not process activity {activity_id}: {e}")
                
                logger.info(f"Processed {cycling_count} cycling activities with metrics")
                
        except Exception as e:
            logger.error(f"Error preloading cache with metrics: {e}")
    
    def _find_last_cycling_activity(self, activities: list) -> Optional[Dict[str, Any]]:
        """Find the most recent cycling activity from activities list (from base)"""
        cycling_activities = [
            act for act in activities
            if "cycling" in act.get("activityType", {}).get("typeKey", "").lower()
        ]
        return max(cycling_activities, key=lambda x: x.get("startTimeGmt", 0)) if cycling_activities else None
    
    # Core functionality methods (enhanced)
    
    async def analyze_workout(self, analysis_type: str = "last_workout", **kwargs) -> str:
        """Analyze workout with deterministic metrics (enhanced version)"""
        if analysis_type == "deterministic":
            return await self.analyze_workout_deterministic(**kwargs)
        # Fallback to base logic if needed
        template_name = f"workflows/{analysis_type}.txt"
        
        # Prepare enhanced context with data quality assessment
        context = self._prepare_analysis_context(**kwargs)
        
        # Load and render template
        logger.info(f"Rendering template {template_name} with context keys: {list(context.keys())}")
        prompt = self.template_engine.render(template_name, **context)
        
        if self.test_mode:
            logger.info("Test mode: Printing rendered prompt instead of calling LLM")
            print("\n" + "="*60)
            print("RENDERED PROMPT FOR LLM:")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")
            return f"TEST MODE: Prompt rendered (length: {len(prompt)} characters)"
        
        # Call LLM
        return await self.llm_client.generate(prompt)
    
    def _prepare_analysis_context(self, **kwargs) -> Dict[str, Any]:
        """Prepare analysis context with data quality assessment (from base)"""
        user_info = self.cache_manager.get("user_profile", {})
        activity_summary = self.cache_manager.get("last_cycling_details", {})
        
        logger.info(f"DEBUG: user_info keys: {list(user_info.keys()) if user_info else 'EMPTY'}, length: {len(user_info) if user_info else 0}")
        logger.info(f"DEBUG: activity_summary keys: {list(activity_summary.keys()) if activity_summary else 'EMPTY'}, length: {len(activity_summary) if activity_summary else 0}")
        
        # Assess data quality
        data_quality = self._assess_data_quality(activity_summary)
        logger.info(f"DEBUG: data_quality: {data_quality}")
        
        context = {
            "user_info": user_info,
            "activity_summary": activity_summary,
            "data_quality": data_quality,
            "missing_metrics": data_quality.get("missing", []),
            **kwargs
        }
        
        logger.debug(f"Prepared context with data quality: {data_quality.get('overall', 'N/A')}")
        return context
    
    def _assess_data_quality(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality and completeness of activity data (from base)"""
        summary_dto = activity_data.get('summaryDTO', {})
        is_indoor = activity_data.get('is_indoor', False)
        
        missing = []
        overall = "complete"
        
        # Key metrics for outdoor cycling
        outdoor_metrics = ['averageSpeed', 'maxSpeed', 'elevationGain', 'elevationLoss']
        # Key metrics for indoor cycling
        indoor_metrics = ['averagePower', 'maxPower', 'averageHR', 'maxHR']
        
        if is_indoor:
            expected = indoor_metrics
            note = "Indoor activity - focus on power and heart rate metrics"
        else:
            expected = outdoor_metrics
            note = "Outdoor activity - full metrics expected"
        
        for metric in expected:
            if summary_dto.get(metric) is None:
                missing.append(metric)
        
        if missing:
            overall = "incomplete"
            note += f" | Missing: {', '.join(missing)}"
        
        return {
            "overall": overall,
            "is_indoor": is_indoor,
            "missing": missing,
            "note": note,
            "available_metrics": [k for k, v in summary_dto.items() if v is not None]
        }
    
    async def suggest_next_workout(self, **kwargs) -> str:
        """Generate data-driven workout suggestion (enhanced version)"""
        return await self.suggest_next_workout_data_driven(**kwargs)
    
    async def enhanced_analysis(self, analysis_type: str, **kwargs) -> str:
        """Perform enhanced analysis based on type (enhanced version)"""
        if analysis_type == "ftp_estimation":
            return await self.estimate_ftp_without_power(**kwargs)
        elif analysis_type == "gear_analysis":
            return await self.analyze_single_speed_gears(**kwargs)
        elif analysis_type == "training_load":
            return await self.get_training_load_analysis(**kwargs)
        else:
            # Fallback to deterministic analysis
            return await self.analyze_workout_deterministic(**kwargs)
    
    # Enhanced analysis methods from enhanced_core_app
    async def analyze_workout_deterministic(self, activity_id: str = None, **kwargs) -> str:
        """Analyze workout using deterministic metrics"""
        if not activity_id:
            activity_id = self._get_last_cycling_activity_id()
        
        if not activity_id:
            return "No cycling activity found for analysis"
        
        # Get deterministic analysis data
        analysis_data = self.cache_manager.get_workout_summary_for_llm(activity_id)
        
        if "error" in analysis_data:
            return f"Error: {analysis_data['error']}"
        
        # Get performance trends
        performance_trends = self.cache_manager.get_performance_trends(30)
        
        # Use enhanced template
        template_name = "workflows/analyze_workout_with_metrics.txt"
        
        context = {
            "workout_summary": analysis_data,
            "performance_trends": [
                {
                    "metric_name": trend.metric_name,
                    "current_value": trend.current_value,
                    "trend_direction": trend.trend_direction,
                    "trend_7day": trend.trend_7day
                }
                for trend in performance_trends
            ],
            "training_rules": kwargs.get("training_rules", ""),
            **kwargs
        }
        
        prompt = self.template_engine.render(template_name, **context)
        return await self.llm_client.generate(prompt)
    
    async def estimate_ftp_without_power(self, activity_id: str = None, **kwargs) -> str:
        """Estimate FTP for workouts without power meter"""
        if not activity_id:
            activity_id = self._get_last_cycling_activity_id()
        
        if not activity_id:
            return "No cycling activity found for FTP estimation"
        
        # Get workout metrics
        metrics = self.cache_manager.get_workout_metrics(activity_id)
        if not metrics:
            return "No metrics available for FTP estimation"
        
        # Get FTP estimation history
        ftp_history = self.cache_manager.get_ftp_estimates_history()
        
        # Calculate additional metrics for FTP estimation
        hr_intensity = 0
        if metrics.avg_hr and self.user_max_hr:
            hr_intensity = metrics.avg_hr / self.user_max_hr
        elif metrics.avg_hr:
            # Estimate max HR if not provided
            estimated_max_hr = 220 - 30  # Assume 30 years old, should be configurable
            hr_intensity = metrics.avg_hr / estimated_max_hr
        
        # Estimate power from speed
        avg_speed_ms = metrics.avg_speed_kmh / 3.6
        estimated_power_from_speed = (avg_speed_ms ** 2.5) * 3.5
        
        # Adjust for elevation
        elevation_per_km = metrics.elevation_gain_m / metrics.distance_km if metrics.distance_km > 0 else 0
        elevation_factor = 1 + (elevation_per_km / 1000) * 0.1
        elevation_adjusted_power = estimated_power_from_speed * elevation_factor
        
        template_name = "workflows/estimate_ftp_no_power.txt"
        
        context = {
            "duration_minutes": metrics.duration_minutes,
            "distance_km": metrics.distance_km,
            "avg_speed_kmh": metrics.avg_speed_kmh,
            "elevation_gain_m": metrics.elevation_gain_m,
            "avg_hr": metrics.avg_hr,
            "max_hr": metrics.max_hr,
            "hr_intensity": hr_intensity,
            "estimated_power_from_speed": round(estimated_power_from_speed, 0),
            "elevation_adjusted_power": round(elevation_adjusted_power, 0),
            "estimated_ftp": metrics.estimated_ftp,
            "elevation_per_km": round(elevation_per_km, 1),
            "elevation_factor": elevation_factor,
            "ftp_history": ftp_history[:10],  # Last 10 estimates
            **kwargs
        }
        
        prompt = self.template_engine.render(template_name, **context)
        return await self.llm_client.generate(prompt)
    
    async def analyze_single_speed_gears(self, activity_id: str = None, **kwargs) -> str:
        """Analyze single speed gear selection and optimization"""
        if not activity_id:
            activity_id = self._get_last_cycling_activity_id()
        
        if not activity_id:
            return "No cycling activity found for gear analysis"
        
        # Get workout metrics
        metrics = self.cache_manager.get_workout_metrics(activity_id)
        if not metrics:
            return "No metrics available for gear analysis"
        
        # Get gear usage analysis
        gear_analysis = self.cache_manager.get_gear_usage_analysis()
        
        # Calculate additional gear metrics
        chainrings = [46, 38]
        cogs = [14, 15, 16, 17, 18, 19, 20]
        wheel_circumference = 2.096  # meters
        
        available_gears = []
        for chainring in chainrings:
            for cog in cogs:
                ratio = chainring / cog
                gear_inches = ratio * 27  # 700c wheel ≈ 27" diameter
                development = ratio * wheel_circumference
                available_gears.append({
                    "chainring": chainring,
                    "cog": cog,
                    "ratio": round(ratio, 2),
                    "gear_inches": round(gear_inches, 1),
                    "development": round(development, 1)
                })
        
        # Estimate cadence
        if metrics.avg_speed_kmh > 0 and metrics.estimated_gear_ratio:
            speed_ms = metrics.avg_speed_kmh / 3.6
            estimated_cadence = (speed_ms / (metrics.estimated_gear_ratio * wheel_circumference)) * 60
        else:
            estimated_cadence = 85  # Default assumption
        
        # Classify terrain
        elevation_per_km = metrics.elevation_gain_m / metrics.distance_km if metrics.distance_km > 0 else 0
        if elevation_per_km > 15:
            terrain_type = "steep_climbing"
        elif elevation_per_km > 8:
            terrain_type = "moderate_climbing"  
        elif elevation_per_km > 3:
            terrain_type = "rolling_hills"
        else:
            terrain_type = "flat_terrain"
        
        template_name = "workflows/single_speed_gear_analysis.txt"
        
        context = {
            "avg_speed_kmh": metrics.avg_speed_kmh,
            "duration_minutes": metrics.duration_minutes,
            "elevation_gain_m": metrics.elevation_gain_m,
            "terrain_type": terrain_type,
            "estimated_chainring": metrics.estimated_chainring,
            "estimated_cog": metrics.estimated_cog,
            "estimated_gear_ratio": metrics.estimated_gear_ratio,
            "gear_inches": round((metrics.estimated_gear_ratio or 2.5) * 27, 1),
            "development_meters": round((metrics.estimated_gear_ratio or 2.5) * wheel_circumference, 1),
            "available_gears": available_gears,
            "gear_usage_by_terrain": gear_analysis.get("gear_by_terrain", {}),
            "best_flat_gear": "46x16",  # Example, should be calculated
            "best_climbing_gear": "38x20",  # Example, should be calculated
            "most_versatile_gear": gear_analysis.get("most_common_gear", {}).get("gear", "46x17"),
            "efficiency_rating": 7,  # Should be calculated based on speed/effort
            "estimated_cadence": round(estimated_cadence, 0),
            "elevation_per_km": round(elevation_per_km, 1),
            **kwargs
        }
        
        prompt = self.template_engine.render(template_name, **context)
        return await self.llm_client.generate(prompt)
    
    async def get_training_load_analysis(self, **kwargs) -> str:
        """Analyze training load and recovery status"""
        training_load = self.cache_manager.get_training_load()
        if not training_load:
            return "Insufficient workout history for training load analysis"
        
        # Get performance trends
        performance_trends = self.cache_manager.get_performance_trends(42)  # 6 weeks
        
        # Classify training load status
        if training_load.training_stress_balance > 5:
            form_status = "fresh_and_ready"
        elif training_load.training_stress_balance > -5:
            form_status = "maintaining_fitness"
        elif training_load.training_stress_balance > -15:
            form_status = "building_fitness"
        else:
            form_status = "high_fatigue_risk"
        
        template_name = "workflows/training_load_analysis.txt"
        
        context = {
            "training_load": {
                "fitness": training_load.fitness,
                "fatigue": training_load.fatigue,
                "form": training_load.form,
                "acute_load": training_load.acute_training_load,
                "chronic_load": training_load.chronic_training_load
            },
            "form_status": form_status,
            "performance_trends": [
                {
                    "metric": trend.metric_name,
                    "trend_direction": trend.trend_direction,
                    "trend_7day": trend.trend_7day,
                    "trend_30day": trend.trend_30day,
                    "confidence": trend.confidence
                }
                for trend in performance_trends
            ],
            "training_rules": kwargs.get("training_rules", ""),
            **kwargs
        }
        
        prompt = self.template_engine.render(template_name, **context)
        return await self.llm_client.generate(prompt)
    
    async def suggest_next_workout_data_driven(self, **kwargs) -> str:
        """Generate data-driven workout suggestions"""
        # Get training load status
        training_load = self.cache_manager.get_training_load()
        performance_trends = self.cache_manager.get_performance_trends(14)  # 2 weeks
        
        # Get recent workout pattern
        recent_activities = self.cache_manager.get("recent_activities", [])
        recent_cycling = [act for act in recent_activities 
                         if "cycling" in act.get("activityType", {}).get("typeKey", "").lower()]
        
        # Analyze recent workout pattern
        recent_intensities = []
        recent_durations = []
        recent_types = []
        
        for activity in recent_cycling[:7]:  # Last 7 cycling activities
            activity_id = str(activity.get("activityId"))
            metrics = self.cache_manager.get_workout_metrics(activity_id)
            if metrics:
                recent_intensities.append(self._rate_intensity(metrics))
                recent_durations.append(metrics.duration_minutes)
                recent_types.append(self._classify_workout(metrics))
        
        # Calculate training pattern analysis
        avg_intensity = sum(recent_intensities) / len(recent_intensities) if recent_intensities else 5
        avg_duration = sum(recent_durations) / len(recent_durations) if recent_durations else 60
        
        # Determine workout recommendation based on data
        if training_load and training_load.form < -10:
            recommendation_type = "recovery_focus"
        elif avg_intensity > 7:
            recommendation_type = "endurance_focus"
        elif avg_intensity < 4:
            recommendation_type = "intensity_focus"
        else:
            recommendation_type = "balanced_progression"
        
        template_name = "workflows/suggest_next_workout_data_driven.txt"
        
        context = {
            "training_load": training_load,
            "performance_trends": performance_trends,
            "recent_workout_analysis": {
                "avg_intensity": round(avg_intensity, 1),
                "avg_duration": round(avg_duration, 0),
                "workout_types": recent_types,
                "pattern_analysis": self._analyze_workout_pattern(recent_types)
            },
            "recommendation_type": recommendation_type,
            "user_ftp": self.user_ftp,
            "training_rules": kwargs.get("training_rules", ""),
            **kwargs
        }
        
        prompt = self.template_engine.render(template_name, **context)
        return await self.llm_client.generate(prompt)
    
    # Utility methods from enhanced
    def _get_last_cycling_activity_id(self) -> Optional[str]:
        """Get the ID of the most recent cycling activity"""
        activities = self.cache_manager.get("recent_activities", [])
        for activity in activities:
            activity_type = activity.get("activityType", {})
            if isinstance(activity_type, dict):
                type_key = activity_type.get("typeKey", "").lower()
            else:
                type_key = str(activity_type).lower()
            
            if "cycling" in type_key or "bike" in type_key:
                return str(activity.get("activityId"))
        return None
    
    def _rate_intensity(self, metrics) -> int:
        """Rate workout intensity 1-10 based on metrics"""
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
        
        return min(int(sum(factors) / len(factors)), 10)
    
    def _classify_workout(self, metrics) -> str:
        """Classify workout type"""
        duration = metrics.duration_minutes
        avg_speed = metrics.avg_speed_kmh
        elevation_gain = metrics.elevation_gain_m / metrics.distance_km if metrics.distance_km > 0 else 0
        
        if duration < 30:
            return "short_intensity"
        elif duration > 180:
            return "long_endurance"
        elif elevation_gain > 10:
            return "climbing_focused"
        elif avg_speed > 35:
            return "high_speed"
        elif avg_speed < 20:
            return "recovery_easy"
        else:
            return "moderate_endurance"
    
    def _analyze_workout_pattern(self, recent_types: list) -> str:
        """Analyze recent workout pattern"""
        if not recent_types:
            return "insufficient_data"
        
        type_counts = {}
        for workout_type in recent_types:
            type_counts[workout_type] = type_counts.get(workout_type, 0) + 1
        
        total_workouts = len(recent_types)
        intensity_workouts = sum(1 for t in recent_types if "intensity" in t or "speed" in t)
        endurance_workouts = sum(1 for t in recent_types if "endurance" in t)
        recovery_workouts = sum(1 for t in recent_types if "recovery" in t)
        
        intensity_ratio = intensity_workouts / total_workouts
        endurance_ratio = endurance_workouts / total_workouts
        
        if intensity_ratio > 0.5:
            return "high_intensity_bias"
        elif recovery_workouts > total_workouts * 0.4:
            return "recovery_heavy"
        elif endurance_ratio > 0.6:
            return "endurance_focused"
        else:
            return "balanced_training"
    
    # Utility methods (from base and enhanced)
    async def list_available_tools(self) -> list:
        """Get list of available MCP tools"""
        return await self.mcp_client.list_tools()
    
    def list_templates(self) -> list:
        """Get list of available templates"""
        return self.template_engine.list_templates()
    
    def get_cached_data(self, key: str = None) -> Any:
        """Get cached data by key, or all if no key provided"""
        return self.cache_manager.get(key) if key else self.cache_manager.get_all()
    
    # New deterministic data access methods from enhanced
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        performance_trends = self.cache_manager.get_performance_trends(30)
        training_load = self.cache_manager.get_training_load()
        ftp_history = self.cache_manager.get_ftp_estimates_history()
        gear_analysis = self.cache_manager.get_gear_usage_analysis()
        
        return {
            "performance_trends": [
                {
                    "metric": trend.metric_name,
                    "current": trend.current_value,
                    "trend_7d": f"{trend.trend_7day:+.1f}%",
                    "trend_30d": f"{trend.trend_30day:+.1f}%",
                    "direction": trend.trend_direction,
                    "confidence": trend.confidence
                }
                for trend in performance_trends
            ],
            "training_load": {
                "fitness": training_load.fitness if training_load else None,
                "fatigue": training_load.fatigue if training_load else None,
                "form": training_load.form if training_load else None
            },
            "ftp_estimates": {
                "latest": ftp_history[0]["estimated_ftp"] if ftp_history else None,
                "trend": "improving" if len(ftp_history) > 1 and ftp_history[0]["estimated_ftp"] > ftp_history[1]["estimated_ftp"] else "stable",
                "history_count": len(ftp_history)
            },
            "gear_usage": {
                "most_common": gear_analysis.get("most_common_gear", {}),
                "total_analyzed": gear_analysis.get("total_workouts_analyzed", 0)
            }
        }
    
    def get_metrics_for_activity(self, activity_id: str) -> Optional[Dict[str, Any]]:
        """Get all calculated metrics for a specific activity"""
        return self.cache_manager.get_deterministic_analysis_data(activity_id)

async def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        config = load_config()
        app = CyclingAnalyzerApp(config)
        
        await app.initialize()
        
        # Example usage
        print("Available tools:", len(await app.list_available_tools()))
        print("Available templates:", len(app.list_templates()))
        
        # Run analysis
        # analysis = await app.analyze_workout("analyze_last_workout",
        #                                     training_rules="Sample rules")
        # print("Analysis:", analysis[:200] + "...")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if 'app' in locals():
            await app.cleanup()

if __name__ == "__main__":
    asyncio.run(main())