"""Convenience re-exports of all metric families."""
from echo_routing.evaluate.metrics_boundary import BoundaryResult, boundary_f1, match_boundaries
from echo_routing.evaluate.metrics_classification import classification_summary
from echo_routing.evaluate.metrics_routing import coverage_and_risk, risk_coverage_curve
from echo_routing.evaluate.metrics_segmentation import segment_f1, temporal_iou
from echo_routing.evaluate.metrics_stability import fragments_per_minute

__all__ = ["BoundaryResult", "boundary_f1", "match_boundaries", "classification_summary", "coverage_and_risk",
           "risk_coverage_curve", "segment_f1", "temporal_iou", "fragments_per_minute"]
