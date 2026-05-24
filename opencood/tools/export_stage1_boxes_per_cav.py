"""Compatibility wrapper for moved stage1 exporter."""

from opencood.detection.stage1_exporters.export_stage1_boxes_per_cav import *  # noqa: F401,F403
from opencood.detection.stage1_exporters.export_stage1_boxes_per_cav import main


if __name__ == "__main__":
    main()
