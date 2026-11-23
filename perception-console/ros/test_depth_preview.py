import base64
import math
import struct
import unittest

from sensor_msgs.msg import Image

from depth_preview import depth_preview


class DepthPreviewTest(unittest.TestCase):
    def test_metres_convert_to_millimetres_with_invalid_values_preserved(self):
        message = Image(width=7, height=1, encoding="32FC1", step=32, is_bigendian=1)
        message.data = struct.pack(">7f", 0.5, 2.125, math.nan, math.inf, -math.inf, -1, 100) + b"pad!"
        result = depth_preview(message)
        self.assertEqual((result["width"], result["height"]), (7, 1))
        self.assertEqual(struct.unpack("<7H", base64.b64decode(result["mm16"])), (500, 2125, 0, 0, 0, 0, 0))

    def test_unsigned_depth_honors_padding_and_downsampling(self):
        message = Image(width=320, height=2, encoding="16UC1", step=644)
        message.data = (struct.pack("<320H", *range(320)) + b"pad!") * 2
        result = depth_preview(message)
        self.assertEqual((result["width"], result["height"]), (160, 1))
        self.assertEqual(struct.unpack("<160H", base64.b64decode(result["mm16"])), tuple(range(0, 320, 2)))

    def test_rejects_inconsistent_or_unrecognized_image_layout(self):
        for encoding, step in [("32FC1", 4), ("16UC1", 3), ("mono16", 4), ("16UC1", 8)]:
            message = Image(width=2, height=1, encoding=encoding, step=step)
            message.data = b"1234"
            self.assertIsNone(depth_preview(message))


if __name__ == "__main__":
    unittest.main()
