#!/usr/bin/env python
"""Setup script that initializes iron templates."""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Add scripts directory to path so we can import the initialization script
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import and run the initialization
try:
    from init_fe_templates import VestergaardWilkes2001
    print("Initializing iron templates...")
    VestergaardWilkes2001.initialise()
    print("Iron templates initialized successfully!")
except Exception as e:
    print(f"Warning: Failed to initialize iron templates: {e}")

# Standard setup configuration (reads from pyproject.toml)
setup()
