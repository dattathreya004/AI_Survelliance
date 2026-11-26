
import sqlite3
import csv
import re

from pathlib import Path
from datetime import datetime
from contextlib import contextmanager


# =====================================================
# DATABASE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "database" / "surveillance.db"
CSV_PATH = BASE_DIR / "logs" / "detection_log.csv"


# =====================================================
# DATABASE CONNECTION
# =====================================================

@contextmanager
def db_connection():

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30
    )

    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# =====================================================
# NORMALIZE IDENTITY
# =====================================================

def normalize(record_type, identity):

    identity = (identity or "").strip()

    if record_type == "Vehicle":
        return re.sub(
            r"[^A-Z0-9]",
            "",
            identity.upper()
        )

    return identity.casefold()


# =====================================================
# DATABASE INITIALIZATION AND MIGRATION
# =====================================================

def init_db():

    with db_connection() as conn:

        # Registered records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS registered_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_type TEXT NOT NULL,
                identity TEXT NOT NULL,
                details TEXT,
                created_at TEXT,
                UNIQUE(record_type, identity)
            )
        """)

        # Detection history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TEXT,
                detection_type TEXT,
                identity TEXT,
                confidence REAL,
                record_exists INTEGER DEFAULT 0,
                registered INTEGER DEFAULT 0,
                image_path TEXT,
                details TEXT
            )
        """)

        # Automatically upgrade older databases.
        # Existing records are preserved.

        detection_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(detections)"
            ).fetchall()
        }

        required_columns = {
            "detected_at": "TEXT",
            "detection_type": "TEXT",
            "identity": "TEXT",
            "confidence": "REAL",
            "record_exists": "INTEGER DEFAULT 0",
            "registered": "INTEGER DEFAULT 0",
            "image_path": "TEXT",
            "details": "TEXT"
        }

        for column_name, column_type in required_columns.items():

            if column_name not in detection_columns:

                conn.execute(
                    f"""
                    ALTER TABLE detections
                    ADD COLUMN {column_name} {column_type}
                    """
                )

        # Upgrade registered records table if necessary.

        registered_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(registered_records)"
            ).fetchall()
        }

        registered_required = {
            "record_type": "TEXT",
            "identity": "TEXT",
            "details": "TEXT",
            "created_at": "TEXT"
        }

        for column_name, column_type in registered_required.items():

            if column_name not in registered_columns:

                conn.execute(
                    f"""
                    ALTER TABLE registered_records
                    ADD COLUMN {column_name} {column_type}
                    """
                )


# =====================================================
# REGISTER PERSON OR VEHICLE
# =====================================================

def register_record(record_type, identity, details=""):

    identity = (identity or "").strip()
    record_type = (record_type or "").strip().title()

    if record_type not in ("Face", "Vehicle"):
        return False, "Invalid record type."

    if not identity:
        return False, "Identity cannot be empty."

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:

        with db_connection() as conn:

            conn.execute("""
                INSERT INTO registered_records
                (record_type, identity, details, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                record_type,
                identity,
                details,
                now
            ))

        return True, "Record registered successfully!"

    except sqlite3.IntegrityError:

        return False, "This record is already registered."


# =====================================================
# CHECK REGISTERED RECORD
# =====================================================

def is_registered_record(record_type, identity):

    if not identity:
        return False

    if identity in ("Unknown", "Plate not readable"):
        return False

    target = normalize(record_type, identity)

    with db_connection() as conn:

        rows = conn.execute("""
            SELECT identity
            FROM registered_records
            WHERE record_type = ?
        """, (record_type,)).fetchall()

    return any(
        normalize(record_type, row["identity"]) == target
        for row in rows
    )


# =====================================================
# CHECK PREVIOUS DETECTION
# =====================================================

def check_previous_record(detection_type, identity):

    if not identity:
        return False

    with db_connection() as conn:

        result = conn.execute("""
            SELECT id
            FROM detections
            WHERE detection_type = ?
            AND identity = ?
            LIMIT 1
        """, (
            detection_type,
            identity
        )).fetchone()

    return result is not None


# =====================================================
# GET REGISTERED RECORDS
# =====================================================

def get_registered_records():

    with db_connection() as conn:

        rows = conn.execute("""
            SELECT *
            FROM registered_records
            ORDER BY id DESC
        """).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# =====================================================
# ADD NEW DETECTION
# =====================================================

def add_detection(
    detection_type,
    identity,
    confidence=None,
    image_path=None,
    details="",
    registered=False
):

    identity = (identity or "Unknown").strip()

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Check whether the identity appeared previously.

    with db_connection() as conn:

        previous = conn.execute("""
            SELECT id
            FROM detections
            WHERE detection_type = ?
            AND identity = ?
            LIMIT 1
        """, (
            detection_type,
            identity
        )).fetchone()

        previous_exists = previous is not None

        # Insert new detection.

        conn.execute("""
            INSERT INTO detections (
                detected_at,
                detection_type,
                identity,
                confidence,
                record_exists,
                registered,
                image_path,
                details
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            detection_type,
            identity,
            confidence,
            int(previous_exists),
            int(registered),
            image_path,
            details
        ))

    # Also save detection in CSV.

    CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    write_header = (
        not CSV_PATH.exists()
        or CSV_PATH.stat().st_size == 0
    )

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if write_header:

            writer.writerow([
                "Date",
                "Detection Type",
                "Identity",
                "Confidence",
                "Previous Record",
                "Registered",
                "Image Path",
                "Details"
            ])

        writer.writerow([
            now,
            detection_type,
            identity,
            confidence,
            previous_exists,
            registered,
            image_path,
            details
        ])

    return previous_exists


# =====================================================
# GET DETECTION HISTORY
# =====================================================

def get_detections(limit=500):

    with db_connection() as conn:

        rows = conn.execute("""
            SELECT *
            FROM detections
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# =====================================================
# GET DASHBOARD COUNTS
# =====================================================

def get_counts():

    with db_connection() as conn:

        total = conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]

        faces = conn.execute("""
            SELECT COUNT(*)
            FROM detections
            WHERE detection_type = 'Face'
        """).fetchone()[0]

        vehicles = conn.execute("""
            SELECT COUNT(*)
            FROM detections
            WHERE detection_type = 'Vehicle'
        """).fetchone()[0]

        plates = conn.execute("""
            SELECT COUNT(*)
            FROM detections
            WHERE detection_type = 'Number Plate'
        """).fetchone()[0]

    return {
        "total": total,
        "faces": faces,
        "vehicles": vehicles,
        "plates": plates
    }


# =====================================================
# SEARCH PREVIOUS RECORDS
# =====================================================

def search_records(term):

    term = f"%{term.strip()}%"

    with db_connection() as conn:

        rows = conn.execute("""
            SELECT *
            FROM detections
            WHERE identity LIKE ?
            OR details LIKE ?
            ORDER BY id DESC
            LIMIT 300
        """, (
            term,
            term
        )).fetchall()

    return [
        dict(row)
        for row in rows
    ]
