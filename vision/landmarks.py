"""Extraction des landmarks d'une image webcam pour models/predictor.py.

Format de sortie : tableau numpy (67, 3) float32, valeurs brutes MediaPipe
(x et y entre 0 et 1, z relatif) :
- lignes 0-20  : main gauche (21 points)
- lignes 21-41 : main droite (21 points)
- lignes 42-66 : les 25 premiers points de la pose
Une partie non detectee est remplie de NaN.

mediapipe et cv2 sont importes dans LandmarkExtractor et main() seulement :
build_points() et ce module s'importent sans eux (la CI ne les installe pas).
"""
import os
from pathlib import Path

import numpy as np

N_HAND = 21
N_POSE = 25
# La pose MediaPipe complete a 33 points ; on garde les 25 premiers.
N_POSE_MEDIAPIPE = 33
N_POINTS = 2 * N_HAND + N_POSE  # 67
LEFT = slice(0, N_HAND)
RIGHT = slice(N_HAND, 2 * N_HAND)
POSE = slice(2 * N_HAND, N_POINTS)

# A verifier empiriquement : True echange les mains gauche et droite.
SWAP_HANDS = False

DEFAULT_MODEL = Path.home() / "Projects" / "asl-data" / "holistic_landmarker.task"
MODEL_ENV_VAR = "SIGN_HOLISTIC_MODEL"


def _to_array(landmarks, expected):
    """Convertit une liste de landmarks en (n, 3), ou None si absente/mauvaise taille."""
    if not landmarks or len(landmarks) != expected:
        return None
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def build_points(left, right, pose) -> np.ndarray:
    """Assemble mains et pose en un tableau (67, 3) float32.

    left, right : listes de 21 landmarks (.x .y .z), ou None / vide.
    pose : liste de 33 landmarks MediaPipe (seuls les 25 premiers sont gardes),
    ou None / vide. Une liste de mauvaise taille compte comme non detectee.
    """
    if SWAP_HANDS:
        left, right = right, left

    points = np.full((N_POINTS, 3), np.nan, dtype=np.float32)
    left_arr = _to_array(left, N_HAND)
    right_arr = _to_array(right, N_HAND)
    pose_arr = _to_array(pose, N_POSE_MEDIAPIPE)
    if left_arr is not None:
        points[LEFT] = left_arr
    if right_arr is not None:
        points[RIGHT] = right_arr
    if pose_arr is not None:
        points[POSE] = pose_arr[:N_POSE]
    return points


def _model_path(model_path):
    if model_path is None:
        model_path = os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL
    return Path(model_path).expanduser()


class LandmarkExtractor:
    """Detecte mains et pose avec le HolisticLandmarker de mediapipe 1.1.1.

    Le modele vient de model_path, sinon de la variable d'environnement
    SIGN_HOLISTIC_MODEL, sinon de ~/Projects/asl-data/holistic_landmarker.task.
    Aucun autre fichier modele n'est charge.
    """

    def __init__(self, model_path=None):
        path = _model_path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Holistic model not found: {path}. Download "
                f"holistic_landmarker.task and set {MODEL_ENV_VAR} to its path."
            )

        # Imports ici : le module reste importable sans mediapipe ni cv2.
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        self._cv2 = cv2
        self._mp = mp
        self.model_path = path
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=VisionTaskRunningMode.IMAGE,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(options)

    def extract(self, frame_bgr) -> np.ndarray:
        """Renvoie les landmarks (67, 3) float32 d'une image BGR.

        frame_bgr doit etre l'image ORIGINALE de la camera, pas la copie
        retournee (miroir) affichee a l'ecran : sinon gauche et droite, et
        les coordonnees x, ne correspondent plus a ce qu'attend le modele.
        """
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError(
                f"frame must be a numpy array, got {type(frame_bgr).__name__}"
            )
        if frame_bgr.dtype != np.uint8:
            raise ValueError(f"frame must be uint8, got {frame_bgr.dtype}")
        if (
            frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.shape[0] == 0
            or frame_bgr.shape[1] == 0
        ):
            raise ValueError(f"frame must have shape (H, W, 3), got {frame_bgr.shape}")

        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        return build_points(
            result.left_hand_landmarks,
            result.right_hand_landmarks,
            result.pose_landmarks,
        )

    def close(self) -> None:
        """Libere le detecteur."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _detected(points, part):
    return not np.isnan(points[part]).any()


def main():
    """Apercu webcam : affiche les parties detectees. Q pour quitter."""
    import cv2

    from vision.camera import Camera

    try:
        with LandmarkExtractor() as extractor, Camera() as camera:
            while True:
                frame = camera.read()
                if frame is None:
                    print("Failed to capture frame.")
                    break

                # Image originale pour l'extraction ; copie miroir pour l'affichage.
                points = extractor.extract(frame)
                rows = int((~np.isnan(points).any(axis=1)).sum())
                print(
                    f"left hand: {_detected(points, LEFT)!s:5}  "
                    f"right hand: {_detected(points, RIGHT)!s:5}  "
                    f"pose: {_detected(points, POSE)!s:5}  "
                    f"rows: {rows}/{N_POINTS}"
                )

                cv2.imshow("Landmarks", cv2.flip(frame, 1))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
