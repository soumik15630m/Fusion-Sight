"""Wire models for the standalone fusion service. Kept separate from the
detector's app/schemas.py so this service has no dependency on the detector
package -- it runs on its own laptop."""
from typing import Optional

from pydantic import BaseModel, Field


class PoseUpdate(BaseModel):
    """One GPS/IMU sample from a phone, arriving on WS /ws/pose/{source_id}."""
    lat: float
    lon: float
    alt: float = 0.0
    heading_deg: float = 0.0
    tilt_deg: float = 0.0
    accuracy_m: Optional[float] = None
    fov_deg: Optional[float] = None
    name: Optional[str] = Field(None, description="Human-readable feed label for the operator page")
