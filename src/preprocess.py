# src/preprocess.py
import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def resize(img, size=(512, 256)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def denoise(img):
    # fast path: median to kill salt/pepper; optional NLM for extra smoothness
    med = cv2.medianBlur(img, 3)
    # non-local means (skimage expects float in [0,1])
    f = med.astype(np.float32) / 255.0
    sigma = estimate_sigma(f, channel_axis=None)
    nlm = denoise_nl_means(
        f, h=1.2 * sigma, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=None
    )
    out = (np.clip(nlm, 0, 1) * 255).astype(np.uint8)
    return out

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def find_ilm_rpe(img):
    """
    Heuristic: for each column, find two strongest vertical gradient sign-changes
    in plausible bands -> approximate ILM (top surface) and RPE (bright band).
    Returns y_ilm, y_rpe arrays (length = width).
    """
    # smooth vertically to suppress speckle and emphasize layers
    k = cv2.GaussianBlur(img, (3,3), 0)
    gy = cv2.Sobel(k, cv2.CV_32F, 0, 1, ksize=3)  # vertical gradient
    H, W = img.shape
    y_ilm = np.zeros(W, dtype=np.int32)
    y_rpe = np.zeros(W, dtype=np.int32)

    for x in range(W):
        col = gy[:, x]
        # search ILM near top 20–40% (first strong negative-to-positive edge)
        top_band = col[:int(0.4*H)]
        y_ilm[x] = int(np.argmax(np.abs(top_band)))

        # search RPE near lower half (strong gradient magnitude)
        bot_band = col[int(0.4*H):]
        y_rpe[x] = int(0.4*H) + int(np.argmax(np.abs(bot_band)))

    # smooth boundaries
    y_ilm = cv2.GaussianBlur(y_ilm.reshape(1,-1).astype(np.float32), (21,1), 0).ravel().astype(int)
    y_rpe = cv2.GaussianBlur(y_rpe.reshape(1,-1).astype(np.float32), (21,1), 0).ravel().astype(int)
    # enforce order
    y_rpe = np.maximum(y_rpe, y_ilm + 5)
    return y_ilm, y_rpe

def vertical_realign(img, y_rpe_target=220):
    """
    Shift the image so the mean RPE sits near y_rpe_target.
    """
    y_ilm, y_rpe = find_ilm_rpe(img)
    shift = int(y_rpe_target - np.median(y_rpe))
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned, y_ilm + shift, y_rpe + shift

def preprocess_bscan(bgr_or_gray, size=(512,256)):
    g = to_gray(bgr_or_gray)
    g = resize(g, size)
    g = denoise(g)
    g = clahe(g)
    g_al, y_ilm, y_rpe = vertical_realign(g)
    return g_al, y_ilm, y_rpe
