from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np

from opencood.extrinsics.types import ExtrinsicEstimate, ExtrinsicInit, MethodContext


@dataclass(frozen=True)
class ImageMatchingConfig:
    matcher: str = "orb"  # orb | sift | loftr | disk | lightglue
    max_features: int = 4000
    ratio_test: float = 0.75
    cross_check: bool = False
    ransac_thresh_px: float = 1.0
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 2000
    min_matches: int = 20
    min_inliers: int = 15
    max_report_matches: int = 200
    resize_max_dim: int = 1024
    allow_no_intrinsics: bool = False
    scale_from_init: bool = True
    t_scale: Optional[float] = None
    device: str = "cpu"  # for LoFTR


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _resize_if_needed(image: np.ndarray, *, max_dim: int) -> Tuple[np.ndarray, float]:
    import cv2

    if max_dim <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    cur_max = max(h, w)
    if cur_max <= max_dim:
        return image, 1.0
    scale = float(max_dim) / float(cur_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _normalize_intrinsics(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (3, 4):
        K = K[:, :3]
    elif K.shape == (4, 4):
        K = K[:3, :3]
    return K.reshape(3, 3).copy()


def _scale_intrinsics(K: np.ndarray, scale: float) -> np.ndarray:
    K = _normalize_intrinsics(K)
    K[0, :] *= float(scale)
    K[1, :] *= float(scale)
    return K


class ImageMatchingEstimator:
    """
    Image-based pose estimation using raw camera images.

    Notes:
      - Uses feature matching + essential matrix (R,t) recovery.
      - Translation scale is ambiguous; when possible it is scaled to match
        the init translation norm (or a user-provided t_scale).
      - Returns a source->target transform in camera coordinates.
    """

    def __init__(self, *, cfg: Optional[ImageMatchingConfig] = None) -> None:
        self._cfg = cfg or ImageMatchingConfig()
        self._loftr = None
        self._disk = None
        self._lightglue = None

    @property
    def config(self) -> ImageMatchingConfig:
        return self._cfg

    def _load_image(self, image: Any) -> np.ndarray:
        import cv2

        if isinstance(image, np.ndarray):
            arr = image
        else:
            arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError(f"Failed to read image: {image}")
        if arr.ndim != 3:
            raise ValueError(f"Expected HxWxC image, got {arr.shape}")
        return arr

    def _match_orb_sift(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        import cv2

        matcher = str(self._cfg.matcher).lower().strip()
        if matcher == "sift":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("SIFT is not available in this OpenCV build")
            detector = cv2.SIFT_create(nfeatures=int(self._cfg.max_features))
            norm = cv2.NORM_L2
        else:
            detector = cv2.ORB_create(nfeatures=int(self._cfg.max_features))
            norm = cv2.NORM_HAMMING

        k0, d0 = detector.detectAndCompute(img0, None)
        k1, d1 = detector.detectAndCompute(img1, None)
        if d0 is None or d1 is None or not k0 or not k1:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {
                "reason": "no_descriptors",
                "num_keypoints0": 0 if not k0 else int(len(k0)),
                "num_keypoints1": 0 if not k1 else int(len(k1)),
            }

        if self._cfg.cross_check:
            bf = cv2.BFMatcher(norm, crossCheck=True)
            matches = bf.match(d0, d1)
            matches = sorted(matches, key=lambda m: m.distance)
        else:
            bf = cv2.BFMatcher(norm)
            raw = bf.knnMatch(d0, d1, k=2)
            matches = []
            ratio = float(self._cfg.ratio_test)
            for m, n in raw:
                if m.distance < ratio * n.distance:
                    matches.append(m)
            matches = sorted(matches, key=lambda m: m.distance)

        if not matches:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {
                "reason": "no_matches",
                "num_keypoints0": int(len(k0)),
                "num_keypoints1": int(len(k1)),
            }

        if self._cfg.max_report_matches > 0 and len(matches) > int(self._cfg.max_report_matches):
            matches = matches[: int(self._cfg.max_report_matches)]

        pts0 = np.array([k0[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts1 = np.array([k1[m.trainIdx].pt for m in matches], dtype=np.float32)
        meta = {
            "num_keypoints0": int(len(k0)),
            "num_keypoints1": int(len(k1)),
            "num_matches": int(len(matches)),
        }
        return pts0, pts1, meta

    def _match_loftr(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        try:
            import torch
            from kornia.feature import LoFTR
        except Exception as exc:
            raise RuntimeError("LoFTR requires kornia+torch; please install them first") from exc

        device = torch.device(str(self._cfg.device))
        matcher = self._loftr
        if matcher is None:
            matcher = LoFTR(pretrained="outdoor").to(device).eval()
            self._loftr = matcher

        img0_t = torch.from_numpy(img0).float() / 255.0
        img1_t = torch.from_numpy(img1).float() / 255.0
        if img0_t.ndim == 2:
            img0_t = img0_t[None, None]
        else:
            img0_t = img0_t.permute(2, 0, 1)[None]
        if img1_t.ndim == 2:
            img1_t = img1_t[None, None]
        else:
            img1_t = img1_t.permute(2, 0, 1)[None]
        if img0_t.shape[1] == 3:
            img0_t = img0_t[:, :1]
        if img1_t.shape[1] == 3:
            img1_t = img1_t[:, :1]

        batch = {"image0": img0_t.to(device), "image1": img1_t.to(device)}
        with torch.no_grad():
            out = matcher(batch)
        mkpts0 = out["keypoints0"].cpu().numpy()
        mkpts1 = out["keypoints1"].cpu().numpy()
        meta = {
            "num_matches": int(mkpts0.shape[0]),
        }
        return mkpts0.astype(np.float32), mkpts1.astype(np.float32), meta

    def _match_disk(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        try:
            import torch
            import kornia.feature as KF
        except Exception as exc:
            raise RuntimeError("DISK requires kornia+torch; please install them first") from exc

        feats0 = self._disk_features(img0)
        feats1 = self._disk_features(img1)

        if feats0.keypoints.numel() == 0 or feats1.keypoints.numel() == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {
                "reason": "no_keypoints",
                "num_keypoints0": int(feats0.keypoints.shape[0]),
                "num_keypoints1": int(feats1.keypoints.shape[0]),
            }

        distances, matches = KF.match_smnn(
            feats0.descriptors,
            feats1.descriptors,
            th=float(self._cfg.ratio_test),
        )
        if matches.numel() == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {
                "reason": "no_matches",
                "num_keypoints0": int(feats0.keypoints.shape[0]),
                "num_keypoints1": int(feats1.keypoints.shape[0]),
            }

        idx0 = matches[:, 0].long()
        idx1 = matches[:, 1].long()
        pts0 = feats0.keypoints[idx0].cpu().numpy()
        pts1 = feats1.keypoints[idx1].cpu().numpy()
        meta = {
            "num_keypoints0": int(feats0.keypoints.shape[0]),
            "num_keypoints1": int(feats1.keypoints.shape[0]),
            "num_matches": int(matches.shape[0]),
        }
        return pts0.astype(np.float32), pts1.astype(np.float32), meta

    def _disk_features(self, img: np.ndarray):
        try:
            import torch
            import kornia.feature as KF
        except Exception as exc:
            raise RuntimeError("DISK requires kornia+torch; please install them first") from exc

        device = torch.device(str(self._cfg.device))
        disk = self._disk
        if disk is None:
            disk = KF.DISK.from_pretrained("depth", device=device)
            disk = disk.eval()
            self._disk = disk

        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim != 3:
            raise ValueError(f"Unsupported image shape for DISK: {arr.shape}")
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]

        arr = arr.astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            feats = disk(t.to(device), n=int(self._cfg.max_features))[0]
        return feats

    def _match_lightglue(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        try:
            import torch
            import kornia.feature as KF
        except Exception as exc:
            raise RuntimeError("LightGlue requires kornia+torch; please install them first") from exc

        feats0 = self._disk_features(img0)
        feats1 = self._disk_features(img1)
        if feats0.keypoints.numel() == 0 or feats1.keypoints.numel() == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {
                "reason": "no_keypoints",
                "num_keypoints0": int(feats0.keypoints.shape[0]),
                "num_keypoints1": int(feats1.keypoints.shape[0]),
            }

        device = torch.device(str(self._cfg.device))
        matcher = self._lightglue
        if matcher is None:
            matcher = KF.LightGlueMatcher("disk").to(device).eval()
            self._lightglue = matcher

        kpts0 = feats0.keypoints.to(device)
        kpts1 = feats1.keypoints.to(device)
        lafs0 = KF.laf_from_center_scale_ori(kpts0.unsqueeze(0))
        lafs1 = KF.laf_from_center_scale_ori(kpts1.unsqueeze(0))
        desc0 = feats0.descriptors.to(device)
        desc1 = feats1.descriptors.to(device)

        with torch.no_grad():
            scores, matches = matcher(
                desc0,
                desc1,
                lafs0,
                lafs1,
                hw1=img0.shape[:2],
                hw2=img1.shape[:2],
            )

        if matches.numel() == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), {
                "reason": "no_matches",
                "num_keypoints0": int(feats0.keypoints.shape[0]),
                "num_keypoints1": int(feats1.keypoints.shape[0]),
            }

        idx0 = matches[:, 0].long()
        idx1 = matches[:, 1].long()
        pts0 = kpts0[idx0].cpu().numpy()
        pts1 = kpts1[idx1].cpu().numpy()
        meta = {
            "num_keypoints0": int(feats0.keypoints.shape[0]),
            "num_keypoints1": int(feats1.keypoints.shape[0]),
            "num_matches": int(matches.shape[0]),
        }
        return pts0.astype(np.float32), pts1.astype(np.float32), meta

    def _compute_matches(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        matcher = str(self._cfg.matcher).lower().strip()
        if matcher == "loftr":
            return self._match_loftr(img0, img1)
        if matcher == "disk":
            return self._match_disk(img0, img1)
        if matcher == "lightglue":
            return self._match_lightglue(img0, img1)
        return self._match_orb_sift(img0, img1)

    def estimate_from_images(
        self,
        src_image: Any,
        dst_image: Any,
        *,
        K_src: Optional[np.ndarray] = None,
        K_dst: Optional[np.ndarray] = None,
        init: Optional[ExtrinsicInit] = None,
        ctx: Optional[MethodContext] = None,
    ) -> ExtrinsicEstimate:
        import cv2

        ctx = ctx or MethodContext()
        start = perf_counter()
        try:
            img0 = self._load_image(src_image)
            img1 = self._load_image(dst_image)
        except Exception as exc:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="image_match",
                time_sec=float(perf_counter() - start),
                extra={"reason": "image_load_failed", "error": str(exc)},
            )

        img0, scale0 = _resize_if_needed(img0, max_dim=int(self._cfg.resize_max_dim))
        img1, scale1 = _resize_if_needed(img1, max_dim=int(self._cfg.resize_max_dim))

        gray0 = _as_gray(img0).astype(np.uint8)
        gray1 = _as_gray(img1).astype(np.uint8)
        matcher_name = str(self._cfg.matcher).lower().strip()
        if matcher_name in {"disk", "lightglue"}:
            match_img0 = img0
            match_img1 = img1
        else:
            match_img0 = gray0
            match_img1 = gray1

        try:
            pts0, pts1, match_meta = self._compute_matches(match_img0, match_img1)
        except Exception as exc:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="image_match",
                time_sec=float(perf_counter() - start),
                extra={"reason": "match_failed", "error": str(exc)},
            )

        if pts0.shape[0] < int(self._cfg.min_matches):
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="image_match",
                time_sec=float(perf_counter() - start),
                extra={"reason": "too_few_matches", **match_meta},
            )

        if K_src is None or K_dst is None:
            if not self._cfg.allow_no_intrinsics:
                return ExtrinsicEstimate(
                    T=None,
                    success=False,
                    method="image_match",
                    time_sec=float(perf_counter() - start),
                    extra={"reason": "missing_intrinsics", **match_meta},
                )
            K_src = np.eye(3, dtype=np.float64)
            K_dst = np.eye(3, dtype=np.float64)
        else:
            K_src = _scale_intrinsics(K_src, scale0)
            K_dst = _scale_intrinsics(K_dst, scale1)

        pts0_px = pts0.reshape(-1, 2)
        pts1_px = pts1.reshape(-1, 2)

        try:
            E, inliers = cv2.findEssentialMat(
                pts0_px,
                pts1_px,
                K_src,
                method=cv2.RANSAC,
                prob=float(self._cfg.ransac_confidence),
                threshold=float(self._cfg.ransac_thresh_px),
                maxIters=int(self._cfg.ransac_max_iters),
            )
        except TypeError:
            E, inliers = cv2.findEssentialMat(
                pts0_px,
                pts1_px,
                K_src,
                method=cv2.RANSAC,
                prob=float(self._cfg.ransac_confidence),
                threshold=float(self._cfg.ransac_thresh_px),
            )
        if E is None:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="image_match",
                time_sec=float(perf_counter() - start),
                extra={"reason": "essential_failed", **match_meta},
            )

        inlier_mask = None
        if inliers is not None:
            inlier_mask = np.asarray(inliers).reshape(-1) > 0
        inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if inlier_count < int(self._cfg.min_inliers):
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="image_match",
                time_sec=float(perf_counter() - start),
                extra={"reason": "too_few_inliers", "inliers": inlier_count, **match_meta},
            )

        try:
            _, R, t, pose_mask = cv2.recoverPose(
                E,
                pts0_px,
                pts1_px,
                K_src,
            )
        except TypeError:
            dist = np.zeros((4, 1), dtype=np.float64)
            _, R, t, pose_mask = cv2.recoverPose(
                E,
                pts0_px,
                pts1_px,
                K_src,
                dist,
            )
        t = t.reshape(3)
        scale = None
        if self._cfg.t_scale is not None:
            scale = float(self._cfg.t_scale)
        elif self._cfg.scale_from_init and init is not None:
            scale = float(np.linalg.norm(np.asarray(init.T_init, dtype=np.float64)[:3, 3]))
        if scale is None or scale <= 0:
            scale = 1.0
        t = t * float(scale)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R, dtype=np.float64)
        T[:3, 3] = t.astype(np.float64)

        RE = TE = None
        if ctx.T_true is not None:
            try:
                from v2x_calib.utils import convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true

                RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(
                    convert_T_to_6DOF(T), convert_T_to_6DOF(ctx.T_true)
                )
            except Exception:
                RE = TE = None

        if pose_mask is None:
            pose_inlier_count = int(inlier_count)
        else:
            pose_inlier_count = int(np.count_nonzero(np.asarray(pose_mask)))
        stability = float(pose_inlier_count) / float(max(1, int(pts0.shape[0])))

        time_sec = perf_counter() - start
        return ExtrinsicEstimate(
            T=T,
            success=True,
            method="image_match",
            stability=stability,
            matches=[],
            RE=None if RE is None else float(RE),
            TE=None if TE is None else float(TE),
            time_sec=float(time_sec),
            extra={
                **match_meta,
                "inliers": int(inlier_count),
                "pose_inliers": int(pose_inlier_count),
                "scale": float(scale),
            },
        )


__all__ = ["ImageMatchingConfig", "ImageMatchingEstimator"]
