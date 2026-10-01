"""Stations along a path, and cross-sections of the field on their normal planes.

Ported from ``research/vessels/straighten.py`` (stations, frames, pixel-count areas) and extended
for tubes that are not round.

**Stations.** :func:`stations` smooths a centerline with a cubic spline whose per-point weights
1 / (0.15 r + 0.05) hold the deviation to ~15 % of the local radius (one global budget let a
19 mm trunk's allowance cut distal corners), resamples it every ``step`` mm of arc length, and
carries a parallel-transport frame (n1, n2 rotated along the tangent without twist).

**Sections.** :func:`section_image` samples the margin on a station's normal disk (a square of
half-width 1.6 r + 2 mm at 0.1 mm). On it:

- :func:`pixel_areas`: the reference's measure - the area of the pixel component holding the
  center, at several margin levels (the model's own interval: levels -2..+2 logits);
- :func:`describe`: the sub-pixel contour of that component at one level (marching squares) and
  what a non-round lumen needs: area, perimeter, equivalent diameter, minimum and maximum
  caliper width (Feret), the second-moment aspect ratio (minor / major axis, 1 = round), the
  centroid's offset from the station,
  whether the contour is closed inside the window, and the contour itself.

Level 0 is the model's boundary. A section whose center is outside the structure at a level
reports area 0 there (the reference's convention) and no contour.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .field import sample
from .geometry import SmoothPath, Stations, transport_frames  # noqa: F401 (re-exported)

LEVELS = (-2.0, -1.0, 0.0, 1.0, 2.0)
PIXEL = 0.1


def stations(points: np.ndarray, radius: np.ndarray, step: float = 0.5, relative: float = 0.15,
             floor: float = 0.05, samples: int = 4000) -> Stations:
    """Smoothed, resampled stations with parallel-transport frames (see the module docstring)."""
    return SmoothPath(points, radius, relative, floor, samples).stations(step)


def half_width(radius: float) -> float:
    return 1.6 * radius + 2.0


def section_image(margin: np.ndarray, geometry, center, n1, n2, half: float, pixel: float = PIXEL,
                  cval: float = -8.0):
    """The margin sampled on the normal square around ``center``: (image, coordinates u = v).
    Axis 0 runs along n1, axis 1 along n2; the center is pixel (c, c), c = len(u) // 2."""
    g = np.arange(-half, half + 1e-9, pixel)
    U, V = np.meshgrid(g, g, indexing="ij")
    P = center + U[..., None] * n1 + V[..., None] * n2
    return sample(margin, geometry, P.reshape(-1, 3), cval=cval).reshape(U.shape), g


def pixel_areas(image: np.ndarray, pixel: float = PIXEL, levels=LEVELS) -> np.ndarray:
    """Area (mm^2) of the pixel component holding the center at each level (0 if the center is out)."""
    c = len(image) // 2
    out = np.zeros(len(levels))
    for j, lv in enumerate(levels):
        lab, _ = ndi.label(image > lv)
        if lab[c, c]:
            out[j] = (lab == lab[c, c]).sum() * pixel * pixel
    return out


def _polygon_area_centroid(xy):
    x, y = xy[:, 0], xy[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cross = x * ys - xs * y
    a = 0.5 * cross.sum()
    if a == 0:
        return 0.0, xy.mean(0)
    cx = ((x + xs) * cross).sum() / (6 * a)
    cy = ((y + ys) * cross).sum() / (6 * a)
    return a, np.array([cx, cy])


def _area_covariance(xy, area, centroid):
    """Second central moments of a polygon's AREA (Green's theorem), / area: a 2 x 2 covariance."""
    x, y = xy[:, 0], xy[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cross = x * ys - xs * y
    ixx = (cross * (x * x + x * xs + xs * xs)).sum() / 12.0
    iyy = (cross * (y * y + y * ys + ys * ys)).sum() / 12.0
    ixy = (cross * (x * ys + 2 * x * y + 2 * xs * ys + xs * y)).sum() / 24.0
    cx, cy = centroid
    return np.array([[ixx / area - cx * cx, ixy / area - cx * cy],
                     [ixy / area - cx * cy, iyy / area - cy * cy]])


def _inside(xy, p):
    """Point-in-polygon (even-odd) for a closed contour."""
    x, y = xy[:, 0], xy[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    hit = ((y > p[1]) != (ys > p[1])) & (p[0] < (xs - x) * (p[1] - y) / (ys - y + 1e-300) + x)
    return bool(hit.sum() % 2)


def _feret(xy):
    """(min, max) caliper width of a polygon, via its convex hull (rotating calipers over edges)."""
    from scipy.spatial import ConvexHull, QhullError
    try:
        h = xy[ConvexHull(xy).vertices]
    except (QhullError, ValueError):
        return 0.0, 0.0
    d = h[:, None] - h[None]
    fmax = float(np.sqrt((d ** 2).sum(-1)).max())
    e = np.roll(h, -1, axis=0) - h
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    nrm = np.stack([-e[:, 1], e[:, 0]], 1)
    proj = h @ nrm.T                                         # (vertices, edges)
    fmin = float((proj.max(0) - proj.min(0)).min())
    return fmin, fmax


def center_contour(image: np.ndarray, coords: np.ndarray, level: float = 0.0):
    """The innermost closed contour at ``level`` around the image's center pixel, as (contour
    (k, 2) in mm, closed: last point = first; signed area; area centroid), or None if the center
    is outside. The image is padded with the outside value, so a component reaching the window's
    edge still yields a closed contour (``describe`` reports whether it did)."""
    from skimage.measure import find_contours
    c = len(image) // 2
    if not image[c, c] > level:
        return None
    pixel = float(coords[1] - coords[0])
    pad = np.pad(image, 1, constant_values=min(float(image.min()), level) - 1.0)
    center = np.array([coords[c], coords[c]])
    best = None
    for cont in find_contours(pad, level):
        xy = coords[0] + (cont - 1.0) * pixel
        if len(xy) < 4 or not np.allclose(xy[0], xy[-1]):
            continue
        a, cen = _polygon_area_centroid(xy[:-1])
        if abs(a) > 0 and _inside(xy[:-1], center) and (best is None or abs(a) < abs(best[1])):
            best = (xy, a, cen)                               # the innermost contour around the center
    return best


def describe(image: np.ndarray, coords: np.ndarray, level: float = 0.0) -> dict:
    """Shape of the component holding the center, from its sub-pixel contour at ``level``.

    Returns area, perimeter (mm), equivalent_diameter (diameter of the circle of equal area),
    min_feret and max_feret (caliper widths, mm), aspect_ratio (minor / major axis of the region's
    second moments of area, exact b / a for an ellipse; 1 = round, 0.5 = twice as wide as deep - not
    eccentricity, which reads 0.4 for an 8 % flattening), centroid_offset (mm, from the station), closed
    (the contour does not reach the window's edge) and contour ((k, 2) in (n1, n2) mm). A center
    outside the structure gives area 0 and contour None."""
    empty = dict(area=0.0, perimeter=0.0, equivalent_diameter=0.0, min_feret=0.0, max_feret=0.0,
                 aspect_ratio=None, centroid_offset=None, closed=False, contour=None)
    best = center_contour(image, coords, level)
    if best is None:
        return empty
    xy, a, cen = best
    c = len(image) // 2
    center = np.array([coords[c], coords[c]])
    pixel = float(coords[1] - coords[0])
    area = abs(a)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    perimeter = float(seg.sum())
    fmin, fmax = _feret(xy[:-1])
    ev = np.sort(np.linalg.eigvalsh(_area_covariance(xy[:-1], a, cen)))
    aspect = float(np.sqrt(max(ev[0], 0.0) / ev[1])) if ev[1] > 0 else 1.0
    edge = coords[0] + 0.5 * pixel, coords[-1] - 0.5 * pixel
    closed = bool((xy.min() > edge[0]) and (xy.max() < edge[1]))
    return dict(area=float(area), perimeter=perimeter, equivalent_diameter=float(2 * np.sqrt(area / np.pi)),
                min_feret=fmin, max_feret=fmax, aspect_ratio=aspect,
                centroid_offset=float(np.linalg.norm(cen - center)), closed=closed, contour=xy)
