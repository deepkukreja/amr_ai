"""Pack a bounded REP-118 depth preview in little-endian millimetres."""
import base64
import math

import numpy as np


def depth_preview(message):
    layout = {"32FC1": "f4", "16UC1": "u2"}.get(message.encoding)
    width, height = message.width, message.height
    if not layout or not width or not height or width * height > 9_000_000:
        return None
    dtype = np.dtype((">" if message.is_bigendian else "<") + layout)
    if message.step < width * dtype.itemsize or len(message.data) != message.step * height:
        return None
    image = np.ndarray((height, width), dtype=dtype, buffer=message.data,
                       strides=(message.step, dtype.itemsize))
    stride = math.ceil(max(width, height) / 160)
    sample = image[::stride, ::stride]
    if layout == "f4":
        # Out-of-range and non-finite values remain unobserved in the preview.
        valid = np.isfinite(sample) & (sample > 0) & (sample <= 65.535)
        sample = np.rint(np.where(valid, sample, 0) * 1000)
    millimetres = sample.astype("<u2", copy=False)
    rows, columns = millimetres.shape
    return {"width": columns, "height": rows,
            "mm16": base64.b64encode(millimetres.tobytes()).decode("ascii")}
