"""
DWL-FallGuard: Uncertainty-aware fall detection, diagnosis, and recovery for humanoid robots.
"""

# Version
__version__ = "0.1.0"

# Public API — what users get when they do `from dwl_fallguard import ...`
from .detection import FallDetector
from .diagnosis import DiagnosisEngine
from .recovery import RecoveryAgent
from .world_model import DWLExtended
from .utils import load_model, save_model