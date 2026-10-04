"""Planes through a 3D echo probe, written in the probe frame (mm): x along the central beam, y elevation, z azimuth.

This is EchoSlicer's frame: a point at range R, azimuth P and elevation T sits at (R cos T cos P, R sin T, R cos T sin P).
An apical 2D view is modelled as a plane through the probe. The zero-elevation plane (x-z) is rotated about the beam by
psi, then fanned about its own lateral axis by alpha (the motion that turns A4C into A5C), then optionally shifted along
its normal. Image coordinates are (s, t): s across the plane (columns run along minus the lateral axis, the standard
display orientation), t along the fanned beam, with the probe at t = 0 at the top of the image, as in a real apical
frame.

At psi = 0 the plane's normal equals EchoSlicer's normal for phi = 0 and theta = 90 + alpha, so a positive alpha fans
the way EchoSlicer searches for A5C. At psi = 90 the plane is EchoSlicer's A2C start plane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbePlane:
    psi_deg: float
    alpha_deg: float
    offset_mm: float = 0.0

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit vectors (depth b, lateral l, normal n) of the plane in the probe frame."""
        psi, alpha = np.radians(self.psi_deg), np.radians(self.alpha_deg)
        lateral = np.array([0.0, np.sin(psi), np.cos(psi)])
        normal0 = np.array([0.0, np.cos(psi), -np.sin(psi)])             # normal before fanning
        depth = np.cos(alpha) * np.array([1.0, 0.0, 0.0]) + np.sin(alpha) * normal0
        normal = -np.sin(alpha) * np.array([1.0, 0.0, 0.0]) + np.cos(alpha) * normal0
        return depth, lateral, normal

    def points(self, s_mm: np.ndarray, t_mm: np.ndarray) -> np.ndarray:
        """(len(t), len(s), 3) probe-frame points: rows run down the depth axis; columns run along -l, the standard
        display orientation. EchoSlicer's A4C image runs along -azimuth too. With +l, every image would be mirrored,
        and EchoSlicer's own A4C planes scored 0.01-0.05 on its classifier instead of about 0.99."""
        b, l, n = self.basis()
        S, T = np.meshgrid(np.asarray(s_mm, float), np.asarray(t_mm, float))
        return self.offset_mm * n - S[..., None] * l + T[..., None] * b


def tilt_between(p1: ProbePlane, p2: ProbePlane) -> float:
    """Angle (degrees) between the two planes' normals, folded to 0-90."""
    c = abs(float(p1.basis()[2] @ p2.basis()[2]))
    return float(np.degrees(np.arccos(min(1.0, c))))


def from_echoslicer(d_mm: float, phi_deg: float, theta_deg: float) -> ProbePlane:
    """The ProbePlane with EchoSlicer's plane (normal (cos t cos p, sin t, cos t sin p), through d * normal).

    Matching n(psi, alpha) = -sin(alpha) x + cos(alpha) (cos(psi) y - sin(psi) z) to EchoSlicer's normal gives alpha and
    psi; the offset is d. The in-plane axes differ from EchoSlicer's rotation and flips, so images match up to an
    in-plane rotation or mirror."""
    p, t = np.radians(phi_deg), np.radians(theta_deg)
    n = np.array([np.cos(t) * np.cos(p), np.sin(t), np.cos(t) * np.sin(p)])
    alpha = -np.arcsin(np.clip(n[0], -1.0, 1.0)); psi = np.arctan2(-n[2], n[1])
    return ProbePlane(float(np.degrees(psi)), float(np.degrees(alpha)), float(d_mm))
