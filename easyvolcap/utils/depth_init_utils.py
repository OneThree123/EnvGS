import numpy as np
import torch
from scipy.spatial import cKDTree


def estimate_surfel_parameters(xyz: torch.Tensor, reference_normals: torch.Tensor = None, knn: int = 16, chunk_size: int = 32768):
    """Estimate 2D Gaussian scales and oriented local-z rotations from a point cloud."""
    points = xyz.detach().cpu().numpy().astype(np.float64, copy=False)
    if len(points) < 3:
        raise ValueError('At least three points are required for depth-point-cloud initialization.')

    k = min(max(int(knn), 3), len(points))
    tree = cKDTree(points)
    normals = np.empty_like(points, dtype=np.float32)
    scales = np.empty((len(points), 2), dtype=np.float32)
    reference = None if reference_normals is None else reference_normals.detach().cpu().numpy().astype(np.float32, copy=False)

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        distances, indices = tree.query(points[start:end], k=k, workers=-1)
        neighbours = points[indices]
        centered = neighbours - neighbours.mean(axis=1, keepdims=True)
        covariance = np.matmul(centered.transpose(0, 2, 1), centered) / (k - 1)
        _, vectors = np.linalg.eigh(covariance)
        chunk_normals = vectors[:, :, 0].astype(np.float32)
        if reference is not None:
            signs = np.sign((chunk_normals * reference[start:end]).sum(axis=1, keepdims=True))
            chunk_normals *= np.where(signs == 0, 1.0, signs)
        normals[start:end] = chunk_normals
        scales[start:end] = np.maximum(distances[:, 1:2], 1e-6).astype(np.float32) * 2.0

    normals_t = torch.from_numpy(normals).to(device=xyz.device, dtype=xyz.dtype)
    scales_t = torch.from_numpy(np.log(scales)).to(device=xyz.device, dtype=xyz.dtype)
    rotations_t = normals_to_quaternions(normals_t)
    return scales_t, rotations_t


def normals_to_quaternions(normals: torch.Tensor):
    """Return (w, x, y, z) rotations whose local z-axis follows each normal."""
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=normals.dtype, device=normals.device).expand_as(normals)
    normal = torch.nn.functional.normalize(normals, dim=-1)
    cross = torch.cross(z_axis, normal, dim=-1)
    dot = (z_axis * normal).sum(dim=-1, keepdim=True)
    quaternion = torch.cat([1.0 + dot, cross], dim=-1)

    opposite = dot[:, 0] < -0.999999
    quaternion[opposite] = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=normals.dtype, device=normals.device)
    return torch.nn.functional.normalize(quaternion, dim=-1)
