
import cv2


class Camera:
    """Manage webcam initialization, frame capture, and cleanup."""

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        """Open the webcam and configure its resolution."""
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                f"Cannot open camera with index {self.camera_index}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print("Camera opened successfully.")
        print(
            f"Resolution: "
            f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )

    def read(self):
        """Capture one frame and return it in OpenCV BGR format."""
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self):
        """Release the webcam resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def main():
    """Run a simple webcam preview to test the camera module."""
    try:
        with Camera() as camera:
            while True:
                frame = camera.read()

                if frame is None:
                    print("Failed to capture frame.")
                    break

                cv2.imshow("Computer Vision - Camera", frame)

                # Press Q to quit.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except RuntimeError as error:
        print(f"Camera error: {error}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
