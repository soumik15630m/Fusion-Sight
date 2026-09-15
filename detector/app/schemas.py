"""Pydantic models — these define the JSON contract between the backend and
the AR helmet client."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_id: int = Field(..., description="0=personnel 1=two_wheeler 2=light_vehicle "
                                             "3=heavy_vehicle -1=moving_object (coherent motion "
                                             "the trained classes didn't recognize, e.g. a UAV)")
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    # Normalised xyxy so the AR client can scale to its own viewport
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = Field(None, description="Stable across frames when tracking")
    moving: Optional[bool] = Field(None, description="True if centroid displaced beyond threshold")
    # Fusion fields (see fusion/engine.py) -- absent/false unless FUSION_ENABLED
    # and this feed has a pose. A ghost is an object this feed's own model
    # missed but another feed detected nearby (geolocation match); source_feed
    # is that other feed's source_id.
    is_ghost: bool = Field(False, description="True if backfilled from another feed rather than detected locally")
    source_feed: Optional[str] = Field(None, description="Originating source_id, set only when is_ghost")
    world_lat: Optional[float] = None
    world_lon: Optional[float] = None
    bearing_deg: Optional[float] = Field(
        None, description="Compass bearing from this feed to the object; ghosts use this for an off-screen edge indicator when not in_frame"
    )
    in_frame: Optional[bool] = Field(None, description="Ghosts only: whether the projected position falls inside this feed's current view")


class DetectionResponse(BaseModel):
    source_id: str
    frame_width: int
    frame_height: int
    inference_ms: float
    detections: List[Detection]


class PoseUpdate(BaseModel):
    """One GPS/IMU sample from a feed's phone, sent over WS /ws/pose/{source_id}
    (app/routers/pose.py), independent of the video/detection WS."""
    lat: float
    lon: float
    name: Optional[str] = Field(None, description="Human-readable feed label (the wearer's name), shown on the operator page instead of the raw source_id")
    alt: float = 0.0
    heading_deg: float = Field(0.0, description="Compass heading, 0=north, clockwise")
    tilt_deg: float = Field(0.0, description="Camera pitch below horizontal; positive = looking down")
    accuracy_m: Optional[float] = Field(None, description="Reported GPS accuracy, widens the fusion match radius")
    fov_deg: Optional[float] = Field(None, description="Camera horizontal FOV if known; falls back to FUSION_DEFAULT_FOV_DEG")


class TrainRequest(BaseModel):
    model: str = "yolo26s.pt"
    data: str = "data/battlesight.yaml"
    epochs: int = Field(60, ge=1, le=1000)
    imgsz: int = Field(640, ge=320, le=1536)
    batch: int = Field(8, ge=1, le=64)
    name: str = "battlesight_v2"


class TrainJob(BaseModel):
    job_id: str
    status: str               # queued | running | completed | failed | cancelled
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    run_name: str
    save_dir: Optional[str] = Field(
        None, description="Actual output dir; ultralytics increments a reused name"
    )
    current_epoch: Optional[int] = None
    total_epochs: int
    log_tail: List[str] = []
    best_weights: Optional[str] = None
