"""crazyflow-gate-ppo: end-to-end JAX PPO expert for Crazyflie gate racing (UAVLA Phase 1)."""

import os

# crazyflow requires scipy's array-API mode and refuses to import if scipy was imported without it.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

__version__ = "0.1.0"
