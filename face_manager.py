import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "known_faces"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "face_lbph.yml"
MAP_PATH = MODEL_DIR / "face_lbph.txt"

CASCADE_PATH = str(
    Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
)


def face_detector():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(
            "Could not load Haar face cascade. Check your OpenCV installation."
        )
    return cascade


def detect_faces(image, cascade=None):
    """Return face rectangles (x, y, width, height) from a BGR or grayscale image."""
    if image is None or image.size == 0:
        return [], None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.equalizeHist(gray)
    detector = cascade if cascade is not None else face_detector()

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return list(faces), gray


def train_faces():
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []
    label_to_id = {}
    cascade = face_detector()
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for path in sorted(FACES_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        person = path.parent.name if path.parent != FACES_DIR else path.stem
        image = cv2.imread(str(path))
        if image is None:
            continue

        faces, gray = detect_faces(image, cascade)
        if not faces:
            continue

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

        if person not in label_to_id:
            label_to_id[person] = len(label_to_id)

        images.append(face)
        labels.append(label_to_id[person])

    if not images:
        raise RuntimeError(
            "No faces were found in known_faces. Use clear, front-facing "
            "photos with the face large and well lit."
        )

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is missing. Install opencv-contrib-python."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))

    id_to_name = {value: key for key, value in label_to_id.items()}
    MAP_PATH.write_text(
        "\n".join(f"{key}={value}" for key, value in id_to_name.items()),
        encoding="utf-8"
    )
    return recognizer, id_to_name, len(images)


def load_face_model():
    if not MODEL_PATH.exists() or not MAP_PATH.exists():
        return None, {}

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is missing. Install opencv-contrib-python."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    names = {}
    for line in MAP_PATH.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            names[int(key)] = value

    return recognizer, names


def identify_face(face_image, recognizer, names, threshold=85):
    if recognizer is None or face_image is None or face_image.size == 0:
        return "Unknown", None

    if len(face_image.shape) == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    face = cv2.resize(face_image, (200, 200))
    label_id, distance = recognizer.predict(face)

    if distance <= threshold and label_id in names:
        return names[label_id], float(distance)

    return "Unknown", float(distance)
