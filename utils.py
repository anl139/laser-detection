import cv2
import numpy as np


def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)

    b, g, r = cv2.split(img)

    b_mean = np.mean(b) + 1e-6
    g_mean = np.mean(g) + 1e-6
    r_mean = np.mean(r) + 1e-6

    gray = (b_mean + g_mean + r_mean) / 3.0

    b *= gray / b_mean
    g *= gray / g_mean
    r *= gray / r_mean

    out = cv2.merge([b, g, r])

    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_enhance(img_bgr):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def underwater_enhance(img_bgr):
    img = gray_world_white_balance(img_bgr)
    img = clahe_enhance(img)

    return img


def underwater_augment(img_rgb):
    img = img_rgb.astype(np.float32)

    # Blue shift
    img[:, :, 0] *= np.random.uniform(1.0, 1.5)

    # Green shift
    img[:, :, 1] *= np.random.uniform(0.9, 1.3)

    # Red attenuation
    img[:, :, 2] *= np.random.uniform(0.4, 1.0)

    # Blur
    if np.random.rand() < 0.5:
        k = np.random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Noise
    if np.random.rand() < 0.7:
        noise = np.random.normal(
            0,
            np.random.uniform(4, 12),
            img.shape
        )
        img += noise

    # Brightness
    if np.random.rand() < 0.8:
        img *= np.random.uniform(0.8, 1.2)

    return np.clip(img, 0, 255).astype(np.uint8)


def make_heatmap(h, w, cx, cy, sigma=6.0):
    heatmap = np.zeros((h, w), dtype=np.float32)

    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    heatmap[cy, cx] = 1.0

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)

    m = heatmap.max()

    if m > 1e-8:
        heatmap /= m

    return heatmap


def crop_around_point(img, cx, cy, crop_size):
    h, w = img.shape[:2]

    half = crop_size // 2

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)

    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    crop = img[y1:y2, x1:x2]

    return crop, x1, y1


def peak_to_xy(heatmap):
    y, x = np.unravel_index(
        np.argmax(heatmap),
        heatmap.shape
    )

    return int(x), int(y), float(heatmap[y, x])