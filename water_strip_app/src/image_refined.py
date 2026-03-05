import cv2
import numpy as np

def crop_strip_simple(raw_bgr):
    """
    Simple strip crop like your notebook:
    threshold -> biggest contour -> bounding box -> rotate if needed
    """
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contour found. Try different lighting/background.")

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    strip = raw_bgr[y:y+h, x:x+w]

    if w > h:
        strip = cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE)
    return strip

def get_pad_data_refined(strip_bgr, num_pads):
    """
    Returns:
      labs_std: list of [L*, a*, b*] (standard CIELAB already!)
      vis_bgr : visualization image
    """
    H, W, _ = strip_bgr.shape
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Signal processing
    row_mean = gray.mean(axis=1)
    row_mean = (row_mean - row_mean.min()) / (row_mean.max() - row_mean.min() + 1e-6)
    signal = 1 - row_mean
    signal = cv2.GaussianBlur(signal.reshape(-1, 1), (1, 21), 0).flatten()

    # Peak detection
    pad_centers = []
    min_dist = max(5, H // (num_pads + 10))

    for i in range(15, len(signal) - 15):
        if signal[i] == max(signal[i-5:i+6]) and signal[i] > 0.02:
            if (not pad_centers) or (i - pad_centers[-1] > min_dist):
                pad_centers.append(i)

    pad_centers = pad_centers[:num_pads]

    vis = strip_bgr.copy()
    labs_std = []

    PAD_H_HALF = int(H * 0.02)

    for i, c in enumerate(pad_centers):
        y1 = max(0, c - PAD_H_HALF)
        y2 = min(H, c + PAD_H_HALF)

        # ROI focus: 30-70% height region inside pad, and 35-65% width
        cy1 = int(y1 + 0.3 * (y2 - y1))
        cy2 = int(y1 + 0.7 * (y2 - y1))
        cx1 = int(W * 0.35)
        cx2 = int(W * 0.65)

        roi = strip_bgr[cy1:cy2, cx1:cx2]
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        # Convert OpenCV LAB -> Standard CIELAB
        L = (roi_lab[:, :, 0].mean() * 100.0) / 255.0
        a = roi_lab[:, :, 1].mean() - 128.0
        b = roi_lab[:, :, 2].mean() - 128.0

        labs_std.append([float(L), float(a), float(b)])

        cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
        cv2.putText(vis, str(i+1), (5, cy1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return labs_std, vis