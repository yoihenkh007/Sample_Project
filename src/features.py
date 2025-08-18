# src/features.py
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

def _band_mask(H, W, y_top, y_bot, band_idx, n_bands):
    """
    Returns a boolean mask for a given band between y_top (ILM) and y_bot (RPE).
    band_idx in [0, n_bands-1], from top to bottom.
    """
    y0 = y_top + (y_bot - y_top) * (band_idx / n_bands)
    y1 = y_top + (y_bot - y_top) * ((band_idx + 1) / n_bands)
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    return (rows >= y0[None, :]) & (rows < y1[None, :])

def _glcm_feats(patch, distances=(1,2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    # clip and quantize to 8 levels to stabilize GLCM on OCT
    q = np.clip(patch, 0, 255).astype(np.uint8) // 32
    glcm = graycomatrix(q, distances=distances, angles=angles, levels=8, symmetric=True, normed=True)
    feats = {}
    for prop in ("contrast","homogeneity","energy","correlation"):
        vals = graycoprops(glcm, prop).ravel()
        feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"glcm_{prop}_std"]  = float(np.std(vals))
    return feats

def _basic_stats(patch):
    hist, _ = np.histogram(patch, bins=32, range=(0,255), density=True)
    ent = entropy(hist + 1e-12)
    return {
        "mean": float(np.mean(patch)),
        "std": float(np.std(patch)),
        "median": float(np.median(patch)),
        "entropy": float(ent),
    }

def _grad_energy(img, mask):
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    g = np.sqrt(gx**2 + gy**2)
    vals = g[mask]
    return {"grad_energy_mean": float(np.mean(vals)), "grad_energy_std": float(np.std(vals))}

def find_fovea(y_ilm):
    y = cv2.GaussianBlur(y_ilm.reshape(1,-1).astype(np.float32), (51,1), 0).ravel()
    x = np.arange(len(y))
    # curvature proxy: second derivative magnitude; fovea ~ max positive curvature & deepest
    dy = np.gradient(y)
    d2y = np.gradient(dy)
    # weight depth
    score = (y - y.min()) * (d2y > 0) * np.abs(d2y)
    xf = int(np.argmax(score))
    depth = float(y[xf] - np.min(y))
    curv  = float(np.max(d2y[max(0,xf-5):min(len(y),xf+6)]))
    return xf, depth, curv

def extract_tabular(img, y_ilm, y_rpe, n_upper=5, n_lower=8):
    H, W = img.shape
    n_bands = n_upper + n_lower

    feats = {}
    thickness = y_rpe - y_ilm
    feats["retina_thick_mean"] = float(np.mean(thickness))
    feats["retina_thick_std"]  = float(np.std(thickness))
    feats["retina_thick_p5"]   = float(np.percentile(thickness, 5))
    feats["retina_thick_p95"]  = float(np.percentile(thickness, 95))

    # Fovea features
    xf, f_depth, f_curv = find_fovea(y_ilm.astype(np.float32))
    feats["fovea_x_rel"]   = float(xf / W)
    feats["fovea_depth"]   = f_depth
    feats["fovea_curv"]    = f_curv

    # Per-band features
    for b in range(n_bands):
        mask = _band_mask(H, W, y_ilm, y_rpe, b, n_bands)
        band = img[mask]
        region = f"upper_{b+1}" if b < n_upper else f"lower_{b-n_upper+1}"
        feats.update({f"{region}_{k}": v for k, v in _basic_stats(band).items()})
        feats.update({f"{region}_{k}": v for k, v in _grad_energy(img, mask).items()})
        feats.update({f"{region}_{k}": v for k, v in _glcm_feats(band.reshape(-1,1)).items()})

    # global SNR-ish proxy
    feats["global_mean"] = float(np.mean(img))
    feats["global_std"]  = float(np.std(img))
    return feats
