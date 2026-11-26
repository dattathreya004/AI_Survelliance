import re
import threading
from pathlib import Path

import pandas as pd
import cv2
import streamlit as st
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from database import (
    init_db,
    get_counts,
    get_detections,
    search_records,
    register_record,
    get_registered_records
)
from face_manager import train_faces
from detector import SurveillanceProcessor


BASE_DIR = Path(__file__).resolve().parent
PROCESSOR_LOCK = threading.Lock()

st.set_page_config(
    page_title="AI Intelligent Surveillance",
    page_icon="📹",
    layout="wide"
)

init_db()

st.title("AI-Based Intelligent Surveillance System")
st.markdown(
    "**Face Recognition | Vehicle Detection | Number Plate Recognition**\n\n"
    "Live Registration | Previous Record Verification | Detection History"
)


@st.cache_resource
def get_processor():
    return SurveillanceProcessor(
        confidence=0.30,
        face_threshold=75,
        cooldown_seconds=8
    )


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.processor = get_processor()
        self.lock = PROCESSOR_LOCK

    def recv(self, video_frame):
        frame = video_frame.to_ndarray(format="bgr24")

        # Limit AI input size to reduce CPU usage, then restore display size.
        original_height, original_width = frame.shape[:2]
        max_width = 480
        if original_width > max_width:
            scale = max_width / original_width
            frame = cv2.resize(
                frame,
                (max_width, max(1, int(original_height * scale)))
            )

        with self.lock:
            processed = self.processor.process(frame)

        if processed.shape[1] != original_width or processed.shape[0] != original_height:
            processed = cv2.resize(
                processed,
                (original_width, original_height)
            )

        return VideoFrame.from_ndarray(processed, format="bgr24")


@st.fragment(run_every="2s")
def show_detection_statistics():
    counts = get_counts()
    st.subheader("Detection Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Detections", counts["total"])
    c2.metric("Face Events", counts["faces"])
    c3.metric("Vehicle Events", counts["vehicles"])
    c4.metric("Number Plate Events", counts["plates"])


@st.fragment(run_every="2s")
def show_detection_history():
    # Re-query SQLite on every fragment refresh so new events appear automatically.
    records = get_detections(1000)
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Detection History",
            df.to_csv(index=False).encode("utf-8"),
            "detection_history.csv",
            "text/csv",
            key="download_detection_history"
        )
        st.caption(f"Showing latest {len(df)} detection record(s). Refreshes every 2 seconds.")
    else:
        st.info("No detection records available yet.")


page = st.sidebar.radio(
    "Select Module",
    [
        "Live Dashboard",
        "Detection History",
        "Search Previous Records",
        "Register Faces",
        "Registered Records",
        "About Project"
    ]
)
st.sidebar.title("Navigation")
st.sidebar.caption("AI Intelligent Surveillance System")


if page == "Live Dashboard":
    st.header("Live Surveillance Dashboard")

    st.subheader("Register New Records")
    tab_person, tab_vehicle = st.tabs(["Register Person", "Register Vehicle"])

    with tab_person:
        with st.form("person_registration"):
            person_name = st.text_input("Person Name")
            face_image = st.file_uploader(
                "Upload Face Photograph",
                type=["jpg", "jpeg", "png"]
            )
            person_details = st.text_input("Additional Details")
            submit_person = st.form_submit_button("Register New Person")

            if submit_person:
                name = person_name.strip()
                if not name or face_image is None:
                    st.error("Enter a name and upload a photograph.")
                elif not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                    st.error("Use letters, numbers, underscore or hyphen only.")
                else:
                    person_folder = BASE_DIR / "known_faces" / name
                    person_folder.mkdir(parents=True, exist_ok=True)
                    image_name = Path(face_image.name).name
                    image_path = person_folder / image_name
                    image_path.write_bytes(face_image.getbuffer())

                    success, message = register_record("Face", name, person_details)
                    try:
                        with st.spinner("Training face recognition model..."):
                            _, names, image_count = train_faces()
                        get_processor.clear()
                        st.success(f"Face training completed using {image_count} image(s).")
                        st.write("Recognized labels:", list(names.values()))
                        st.success(message) if success else st.info(message)
                    except Exception as exc:
                        st.error(f"Face image saved, but training failed: {exc}")

    with tab_vehicle:
        with st.form("vehicle_registration"):
            vehicle_number = st.text_input("Vehicle Registration Number")
            vehicle_type = st.selectbox(
                "Vehicle Type",
                ["Car", "Motorcycle", "Bus", "Truck"]
            )
            vehicle_details = st.text_input("Additional Details")
            submit_vehicle = st.form_submit_button("Register New Vehicle")

            if submit_vehicle:
                plate = re.sub(r"[^A-Z0-9]", "", vehicle_number.upper())
                if not plate:
                    st.error("Enter a vehicle registration number.")
                else:
                    success, message = register_record(
                        "Vehicle",
                        plate,
                        f"{vehicle_type} - {vehicle_details}"
                    )
                    st.success(message) if success else st.warning(message)

    st.divider()
    show_detection_statistics()
    st.divider()

    st.subheader("Live Camera Surveillance")
    st.info(
        "Allow camera access in your browser. For DroidCam, start the "
        "Android application and Windows DroidCam Client."
    )

    try:
        with st.spinner("Loading YOLO, EasyOCR and Face Recognition..."):
            get_processor()
        st.success("AI models loaded successfully!")
    except Exception as exc:
        st.error(f"AI model loading failed: {exc}")
        st.stop()

    # Keep the existing WebRTC camera key and browser-camera connection.
    webrtc_streamer(
        key="ai-surveillance-camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    st.caption(
        "Detection events are stored automatically. "
        "An 8-second cooldown prevents excessive repeated entries."
    )


elif page == "Detection History":
    st.header("Detection History")
    st.caption("New detections are loaded automatically every 2 seconds.")
    show_detection_history()


elif page == "Search Previous Records":
    st.header("Search Previous Records")
    search = st.text_input("Enter Person Name or Vehicle Number")
    if search:
        records = search_records(search)
        if records:
            st.success(f"{len(records)} previous detection event(s) found.")
            st.dataframe(
                pd.DataFrame(records),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No previous detection history found.")


elif page == "Register Faces":
    st.header("Face Recognition Training")
    st.write(
        "Add photographs to known_faces/<person_name>/, for example "
        "known_faces/Ravi/1.jpg and known_faces/Ravi/2.jpg."
    )
    if st.button("Train / Update Face Model"):
        try:
            with st.spinner("Training face recognition..."):
                _, names, image_count = train_faces()
            get_processor.clear()
            st.success("Face model training completed!")
            st.write("Total training images:", image_count)
            st.write("Registered face labels:", list(names.values()))
        except Exception as exc:
            st.error(str(exc))


elif page == "Registered Records":
    st.header("Registered People and Vehicles")
    records = get_registered_records()
    if records:
        st.dataframe(
            pd.DataFrame(records),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No registered records available.")


elif page == "About Project":
    st.header("AI-Based Intelligent Surveillance System")
    st.markdown(
        """
### Technologies Used
- Python
- Streamlit
- OpenCV
- YOLOv8
- EasyOCR
- SQLite Database
- WebRTC

### Main Features
1. Real-time face detection.
2. Face recognition.
3. Vehicle detection.
4. Number plate text recognition.
5. New person registration.
6. New vehicle registration.
7. Previous record verification.
8. Automatic detection history.
9. Database management.
10. Live surveillance dashboard.

### Database
All registered identities and detection events are stored in the SQLite database.
"""
    )
