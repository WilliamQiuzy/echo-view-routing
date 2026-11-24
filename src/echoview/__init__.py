"""EchoRouter: echocardiography view recognition and view segmentation for video.

    import echoview
    system = echoview.load()                          # the three published seeds as one ensemble
    recording = echoview.read_recording("clip.mp4")   # DICOM cine, video file or folder of frames
    print(system.classify(recording.frames))          # one of PLAX, PSAX, A4C, A5C, SC4C
    print(system.segment(recording.frames).segments)  # the pieces of a longer stream
"""
from __future__ import annotations

from pathlib import Path

from . import uncertain
from .frames import ReadError, Recording, find_recordings, read_recording
from .model import CLASSES, WindowModel
from .pipeline import Classification, Ensemble, Segmentation
from .reference import NAME as MODEL_NAME
from .reference import VERSION as MODEL_VERSION
from .weights import WeightsError

__version__ = "1.3.0"
__all__ = ["CLASSES", "MODEL_NAME", "MODEL_VERSION", "Classification", "Ensemble", "ReadError", "Recording", "Segmentation", "WeightsError",
           "WindowModel", "find_recordings", "load", "read_recording", "uncertain"]


def load(weights: str | Path | None = None, device: str = "auto", seeds: tuple = (0, 1, 2), batch: int = 0) -> Ensemble:
    """The reference ensemble. Without `weights` the folder is $ECHOVIEW_WEIGHTS, else a download from the Hub."""
    return Ensemble(weights, seeds=seeds, device=device, batch=batch)
