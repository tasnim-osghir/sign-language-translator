"""Vision module: camera -> MediaPipe (hands + pose) -> (67, 3) landmarks.

Heavy imports (cv2, mediapipe) are done lazily so that this package can be
imported in CI, where no camera or GPU is available.
"""

__version__ = "0.1.0"