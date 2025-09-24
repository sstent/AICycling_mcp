#!/usr/bin/env python3
"""
Cache Manager - Handles caching of MCP responses and other data
"""

import time
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds

class CacheManager:
    """Manages caching of data with TTL support"""
    
    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
    
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

# Specialized cache for common cycling data patterns
class CyclingDataCache(CacheManager):
    """Specialized cache for cycling data with helper methods"""
    
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