from __future__ import annotations

import copy

import numpy as np


class BBox:
    """Lightweight 2D/metadata box used by legacy V2X-Reg++ utilities."""

    def __init__(
        self,
        bbox_type,
        bbox_4=[0, 0, 0, 0],
        occluded_state=0,
        truncated_state=0,
        alpha=0.0,
        confidence=1.0,
    ):
        self.bbox_type = bbox_type
        self.occluded_state = occluded_state
        self.truncated_state = truncated_state
        self.alpha = alpha
        self.bbox2d_4 = bbox_4
        self.confidence = confidence

    def __eq__(self, other):
        if not isinstance(other, BBox):
            return False
        return (
            self.bbox_type == other.bbox_type
            and self.occluded_state == other.occluded_state
            and self.truncated_state == other.truncated_state
        )

    def get_bbox_type(self):
        return self.bbox_type.lower()

    def get_bbox2d_4(self):
        return self.bbox2d_4

    def get_occluded_state(self):
        return self.occluded_state

    def get_truncated_state(self):
        return self.truncated_state

    def get_alpha(self):
        return self.alpha

    def get_confidence(self):
        return self.confidence

    def set_confidence(self, confidence):
        self.confidence = confidence

    def copy(self):
        return BBox(
            self.bbox_type,
            self.bbox2d_4,
            self.occluded_state,
            self.truncated_state,
            self.alpha,
            self.confidence,
        )


class BBox3d(BBox):
    """3D box with 8 corners plus optional descriptor."""

    def __init__(
        self,
        bbox_type,
        bbox_8_3,
        bbox_4=[0, 0, 0, 0],
        occluded_state=0,
        truncated_state=0,
        alpha=0.0,
        confidence=1.0,
        descriptor=None,
    ):
        super().__init__(bbox_type, bbox_4, occluded_state, truncated_state, alpha, confidence)
        self.bbox3d_8_3 = bbox_8_3
        self.descriptor = descriptor

    def __eq__(self, other):
        if not isinstance(other, BBox3d):
            return False
        return super().__eq__(other) and np.array_equal(self.bbox3d_8_3, other.bbox3d_8_3)

    def get_bbox3d_8_3(self):
        return self.bbox3d_8_3

    def get_descriptor(self):
        return self.descriptor

    def copy(self):
        return BBox3d(
            bbox_type=self.bbox_type,
            bbox_8_3=copy.deepcopy(self.bbox3d_8_3),
            bbox_4=copy.deepcopy(self.bbox2d_4),
            occluded_state=self.occluded_state,
            truncated_state=self.truncated_state,
            alpha=self.alpha,
            confidence=self.confidence,
            descriptor=copy.deepcopy(self.descriptor),
        )


__all__ = ["BBox", "BBox3d"]
