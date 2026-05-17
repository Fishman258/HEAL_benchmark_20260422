import numpy as np


def get_xyz_from_bbox3d_8_3(bbox3d_8_3):
    return np.mean(bbox3d_8_3, axis=0)

def get_lwh_from_bbox3d_8_3(bbox3d_8_3):
    """
    Estimate box edge lengths (l, w, h) from 8 corners.

    NOTE:
    The original implementation assumed a fixed corner ordering. Detector exports
    (e.g., HEAL/PointPillars/SECOND caches) can shuffle corner order, which makes
    index-based edge extraction invalid and breaks volume-based filtering.

    This implementation is order-invariant and avoids "short diagonal" traps:
    it recovers the three side lengths via PCA on the 8 corners. For an ideal
    cuboid with side lengths (l,w,h), the variance of its corners along each
    principal axis equals (side/2)^2, so side = 2*sqrt(eigval).
    """
    pts = np.asarray(bbox3d_8_3, dtype=np.float64)
    if pts.shape != (8, 3):
        pts = pts.reshape(-1, 3)
    if pts.shape[0] != 8:
        # Fallback: axis-aligned extents in the observed frame.
        ext = np.ptp(pts, axis=0) if pts.size else np.zeros(3, dtype=np.float64)
        return float(ext[0]), float(ext[1]), float(ext[2])
    center = np.mean(pts, axis=0)
    rel = pts - center
    cov = (rel.T @ rel) / 8.0
    if not np.all(np.isfinite(cov)):
        ext = np.ptp(pts, axis=0)
        return float(ext[0]), float(ext[1]), float(ext[2])
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    lengths = 2.0 * np.sqrt(eigvals)
    l, w, h = sorted((float(lengths[0]), float(lengths[1]), float(lengths[2])), reverse=True)
    return l, w, h

def get_bbox3d_8_3_from_xyz_lwh_yaw(xyz, lwh, yaw):
    l, w, h = lwh
    r = np.matrix(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    center = [xyz[0], xyz[1], xyz[2] - h / 2]
    corners_3d = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d = r * corners_3d + np.matrix(center).T
    return np.array(corners_3d.T)

def get_bbox3d_8_3_from_xyz_lwh(xyz, lwh):
    return get_bbox3d_8_3_from_xyz_lwh_yaw(xyz, lwh, 0)

def get_bbox3d_n_8_3_from_bbox_object_list(bbox_object_list):
    return np.array([bbox_object.get_bbox3d_8_3() for bbox_object in bbox_object_list])
    

def get_vector_between_bbox3d_8_3(bbox3d_8_3_1, bbox3d_8_3_2):
    xyz1 = get_xyz_from_bbox3d_8_3(bbox3d_8_3_1)
    xyz2 = get_xyz_from_bbox3d_8_3(bbox3d_8_3_2)
    vector = xyz2 - xyz1
    return vector

def get_length_between_bbox3d_8_3(bbox3d_8_3_one, bbox3d_8_3_two):
    vector = get_vector_between_bbox3d_8_3(bbox3d_8_3_one, bbox3d_8_3_two)
    length = np.linalg.norm(vector)
    return length

def get_volume_from_bbox3d_8_3(bbox3d_8_3):
    l, w, h = get_lwh_from_bbox3d_8_3(bbox3d_8_3)
    volume = l * w * h
    return volume
