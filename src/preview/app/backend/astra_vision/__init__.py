"""Image-space armor recognition and tracking for RoboMaster simulation."""

from .detector import ArmorDetector, DetectorConfig
from .tracking import MultiObjectTracker, TrackerConfig
from .types import Detection, TrackSnapshot
from .fusion import FusionArmorDetector, FusionConfig, FusionResult

__all__ = ["ArmorDetector", "DetectorConfig", "Detection", "MultiObjectTracker", "TrackerConfig", "TrackSnapshot",
           "FusionArmorDetector", "FusionConfig", "FusionResult"]
