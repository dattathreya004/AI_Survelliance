# AI-Based Intelligent Surveillance System

Features:
- Face detection and LBPH face recognition
- YOLOv8 vehicle detection (car, motorcycle, bus, truck)
- EasyOCR number-plate text extraction from detected vehicle crops
- Automatic SQLite event storage and previous-event checking
- Streamlit live dashboard, history, search, and face enrollment

## Windows setup

Use Python 3.10 or 3.11 in a fresh virtual environment:

```powershell
cd "C:\Users\admin\OneDrive\Desktop\DCSCC19\AI_Surveillance"
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads YOLO weights (`yolov8n.pt`) and EasyOCR language data. An internet connection is needed for the initial download.

## Run

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, usually http://localhost:8501.

## Face enrollment

Put images in folders named after the person:

```text
known_faces/
  Ravi/
    1.jpg
    2.jpg
  Sita/
    1.jpg
```

Flat files such as `known_faces/person1.jpg` are also supported. In the dashboard, open **Register Faces** and click **Train / update face recognition model**.

For better accuracy, use 5–10 clear face images per person. Use only images you have permission to process.

## DroidCam / camera

Start DroidCam on the Android phone and the Windows DroidCam Client. In the browser, grant camera permission and choose the DroidCam virtual camera as the browser camera device. `streamlit-webrtc` uses the browser's selected camera, not OpenCV camera index 0.

## Database behavior

Before each accepted event is inserted, the application checks whether the same detection type and identity already exists in `database/surveillance.db`. The first event is stored as a first detection; later matches are stored with `record_exists=1`.

The processor applies an 8-second per-identity cooldown to prevent logging the same object on every frame. Change `cooldown_seconds` in `app.py`'s processor construction if needed.

## Notes

- YOLOv8n COCO detects vehicle classes; it is not a license-plate detector.
- EasyOCR attempts to read text from each vehicle crop. A dedicated plate detector is recommended for reliable plate localization.
- Face recognition uses OpenCV LBPH. It is a basic local recognition approach, not a high-security identity verification system.
- Keep the camera feed and database secured. Obtain consent and follow applicable privacy laws.
