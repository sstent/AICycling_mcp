#!/usr/bin/env python3
"""
Configuration management
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration"""
    # LLM settings
    openrouter_api_key: str
    openrouter_model: str = "deepseek/deepseek-r1-0528:free"
    
    # MCP settings
    garth_token: str = ""
    garth_mcp_server_path: str = "uvx"
    
    # Application settings
    templates_dir: str = "templates"
    rules_file: str = "rules.yaml"
    cache_ttl: int = 300  # Cache TTL in seconds
    
    # Logging
    log_level: str = "INFO"

def load_config(config_file: str = "config.yaml") -> Config:
    """Load configuration from file or environment variables"""
    config_path = Path(config_file)
    
    # Load from file if exists
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        return Config(**config_data)
    
    # Load from environment variables
    return Config(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free"),
        garth_token=os.getenv("GARTH_TOKEN", ""),
        garth_mcp_server_path=os.getenv("GARTH_MCP_SERVER_PATH", "uvx"),
        templates_dir=os.getenv("TEMPLATES_DIR", "templates"),
        rules_file=os.getenv("RULES_FILE", "rules.yaml"),
        cache_ttl=int(os.getenv("CACHE_TTL", "300")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

def create_sample_config(config_file: str = "config.yaml") -> None:
    """Create a sample configuration file"""
    config_path = Path(config_file)
    if config_path.exists():
        return
    
    sample_config = {
        "openrouter_api_key": "your_openrouter_api_key_here",
        "openrouter_model": "deepseek/deepseek-r1-0528:free",
        "garth_token": "your_garth_token_here",
        "garth_mcp_server_path": "uvx",
        "templates_dir": "templates",
        "rules_file": "rules.yaml",
        "cache_ttl": 300,
        "log_level": "INFO"
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    print(f"Created sample config file: {config_file}")
    print("Please edit with your actual API keys and settings.")