# -*- coding: utf-8 -*-

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset


class V2VLOCBaseDataset(OPV2VBaseDataset):
    """
    V2VLoc uses the same OpenCDA/OpenCOOD folder convention as OPV2V.

    This class is an alias for OPV2VBaseDataset to keep dataset selection
    explicit in yaml configs.
    """

    pass
