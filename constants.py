"""
Path configuration for FrontierScience environment.

Automatically uses /orwd_data when available (production ORS deployment),
otherwise falls back to local directory (development).
"""

import os
from pathlib import Path

# Production: use /orwd_data if it exists
# Development: use local directory
if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent
