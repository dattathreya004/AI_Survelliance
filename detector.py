import cv2
import easyocr
import re
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from database import add_detection, is_registered_record
from face_manager import face_detector, load_face_model, identify_face


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CAPTURE_DIR = BASE_DIR / "captured"

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


class SurveillanceProcessor:
    def __init__(self, confidence=0.30, face_threshold=75, cooldown_seconds=8):
        MODEL_DIR.mkdir(exist_ok=True)
        CAPTURE_DIR.mkdir(exist_ok=True)

        yolo_path = MODEL_DIR / "yolov8n.pt"
        self.yolo = YOLO(str(yolo_path)) if yolo_path.exists() else YOLO("yolov8n.pt")

        # OCR is initialized once, not for every video frame.
        self.reader = easyocr.Reader(["en"], gpu=False)

        self.face_cascade = face_detector()
        self.face_model, self.face_names = load_face_model()

        self.confidence = confidence
        self.face_threshold = face_threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_logged = {}

        self.frame_count = 0
        self.cached_faces = []
        self.cached_vehicles = []
        self.face_interval = 2       # Haar face detection every 2nd frame
        self.vehicle_interval = 12   # YOLO every 12th frame
        self.ocr_interval = 90       # OCR less frequently; it is CPU intensive

    def reload_face_model(self):
        self.face_model, self.face_names = load_face_model()

    def allowed_to_log(self, key):
        now = datetime.now().timestamp()
        last = self.last_logged.get(key, 0)
        if now - last >= self.cooldown_seconds:
            self.last_logged[key] = now
            return True
        return False

    def save_crop(self, crop, prefix):
        if crop is None or crop.size == 0:
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = CAPTURE_DIR / f"{prefix}_{stamp}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path.relative_to(BASE_DIR))

    def read_plate(self, vehicle_crop):
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        # Keep OCR input reasonably small to reduce CPU load.
        height, width = vehicle_crop.shape[:2]
        if width > 640:
            scale = 640 / width
            vehicle_crop = cv2.resize(
                vehicle_crop,
                (640, max(1, int(height * scale)))
            )

        results = self.reader.readtext(
            vehicle_crop,
            detail=1,
            paragraph=False
        )

        candidates = []
        for _, text, score in results:
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            if len(cleaned) >= 6 and float(score) >= 0.20:
                candidates.append((cleaned, float(score)))

        if not candidates:
            return None

        indian_plate = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")
        valid = [item for item in candidates if indian_plate.match(item[0])]
        return max(valid or candidates, key=lambda item: item[1])

    def _update_faces(self, frame, gray):
        found = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        updated = []
        for (x, y, w, h) in found:
            face_crop = gray[y:y + h, x:x + w]
            name, distance = identify_face(
                face_crop,
                self.face_model,
                self.face_names,
                self.face_threshold
            )

            registered = (
                name != "Unknown"
                and is_registered_record("Face", name)
            )

            if registered:
                label = f"{name} - REGISTERED"
            elif name != "Unknown":
                label = f"{name} - NOT REGISTERED"
            else:
                label = "Unknown Face"
            # Always use green for every detected face, as requested.
            color = (0, 255, 0)

            updated.append((x, y, w, h, label, color))

            log_key = f"Face:{name}"
            if self.allowed_to_log(log_key):
                image_path = self.save_crop(frame[y:y + h, x:x + w], "face")
                add_detection(
                    "Face",
                    name,
                    distance,
                    image_path,
                    details=f"Face recognition; registered={registered}",
                    registered=registered
                )

        self.cached_faces = updated

    def _update_vehicles(self, frame, run_ocr):
        result = self.yolo.predict(
            frame,
            conf=self.confidence,
            imgsz=320,
            verbose=False,
            device="cpu"
        )[0]

        updated = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.yolo.names[class_id]
            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            vehicle_conf = float(box.conf[0])
            plate_result = self.read_plate(vehicle_crop) if run_ocr else None
            plate = plate_result[0] if plate_result else "Plate not readable"

            registered = (
                plate != "Plate not readable"
                and is_registered_record("Vehicle", plate)
            )

            if registered:
                label, color = f"{class_name} | {plate} | REGISTERED", (0, 180, 0)
            elif plate != "Plate not readable":
                label, color = f"{class_name} | {plate} | NEW", (0, 0, 255)
            else:
                label, color = f"{class_name} | PLATE NOT READABLE", (0, 165, 255)

            updated.append((x1, y1, x2, y2, label, color))

            identity = plate if plate != "Plate not readable" else class_name
            log_key = f"Vehicle:{class_name}:{identity}"

            if self.allowed_to_log(log_key):
                image_path = self.save_crop(vehicle_crop, "vehicle")
                add_detection(
                    "Vehicle",
                    identity,
                    vehicle_conf,
                    image_path,
                    details=(
                        f"Vehicle type={class_name}; "
                        f"plate={plate}; registered={registered}"
                    ),
                    registered=registered
                )

                if plate != "Plate not readable":
                    plate_key = f"Number Plate:{plate}"
                    if self.allowed_to_log(plate_key):
                        add_detection(
                            "Number Plate",
                            plate,
                            vehicle_conf,
                            image_path,
                            details=f"Vehicle type={class_name}",
                            registered=registered
                        )

        self.cached_vehicles = updated

    def process(self, frame):
        output = frame.copy()
        self.frame_count += 1

        # Run Haar detection less often and reuse the most recent boxes between
        # inference frames. This reduces CPU load while keeping boxes visible.
        if self.frame_count % self.face_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            self._update_faces(frame, gray)

        # YOLO and EasyOCR are the heaviest operations. Run them less often.
        if self.frame_count % self.vehicle_interval == 0:
            run_ocr = (self.frame_count % self.ocr_interval == 0)
            self._update_vehicles(frame, run_ocr)

        for x, y, w, h, label, color in self.cached_faces:
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                output, label, (x, max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

        for x1, y1, x2, y2, label, color in self.cached_vehicles:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output, label, (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2
            )

        return output
