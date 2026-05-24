"""Compatibility wrapper for moved image-depth pseudo-LiDAR stage1 exporter."""

from opencood.detection.stage1_exporters.export_stage1_boxes_image_depth_pseudolidar import *  # noqa: F401,F403
from opencood.detection.stage1_exporters.export_stage1_boxes_image_depth_pseudolidar import main


if __name__ == "__main__":
    main()
