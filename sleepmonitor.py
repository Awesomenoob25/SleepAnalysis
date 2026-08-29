import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from datetime import datetime
import threading
import csv
import os
import re


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = "pose_landmarker_full.task"
EXPORT_FOLDER = "exports"

# Classification defaults
BODY_VISIBILITY_THRESHOLD = 0.45
SIDE_SHOULDER_WIDTH = 0.12
SIDE_HIP_WIDTH = 0.10
BACK_FACE_VISIBILITY = 0.60
STOMACH_FACE_VISIBILITY = 0.35
LANDMARK_VISIBILITY_THRESHOLD = 0.40

# Current session/export information
current_session_name = "Sleep_Session"
current_export_file = None




# --------------------------------------------------
# CONFIGURABLE EXPORT FOLDER
# --------------------------------------------------

def set_export_folder(export_folder):
    """Update where sleep-position CSV data is saved."""

    global EXPORT_FOLDER

    if export_folder:
        EXPORT_FOLDER = os.path.abspath(
            os.path.expanduser(str(export_folder))
        )

    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    print("Sleep data export folder:", EXPORT_FOLDER)

    return EXPORT_FOLDER



# --------------------------------------------------
# CONFIGURABLE CLASSIFICATION SETTINGS
# --------------------------------------------------

def set_classification_settings(settings):
    """
    Update sleep-position classification thresholds
    from config.json.
    """

    global BODY_VISIBILITY_THRESHOLD
    global SIDE_SHOULDER_WIDTH
    global SIDE_HIP_WIDTH
    global BACK_FACE_VISIBILITY
    global STOMACH_FACE_VISIBILITY
    global LANDMARK_VISIBILITY_THRESHOLD

    if not isinstance(settings, dict):
        return

    def get_float(name, current_value):
        try:
            value = float(settings.get(name, current_value))
        except (TypeError, ValueError):
            return current_value

        # Keep thresholds in a sensible normalised range.
        if value < 0.0:
            return 0.0

        if value > 1.0:
            return 1.0

        return value

    BODY_VISIBILITY_THRESHOLD = get_float(
        "body_visibility_threshold",
        BODY_VISIBILITY_THRESHOLD
    )

    SIDE_SHOULDER_WIDTH = get_float(
        "side_shoulder_width",
        SIDE_SHOULDER_WIDTH
    )

    SIDE_HIP_WIDTH = get_float(
        "side_hip_width",
        SIDE_HIP_WIDTH
    )

    BACK_FACE_VISIBILITY = get_float(
        "back_face_visibility",
        BACK_FACE_VISIBILITY
    )

    STOMACH_FACE_VISIBILITY = get_float(
        "stomach_face_visibility",
        STOMACH_FACE_VISIBILITY
    )

    LANDMARK_VISIBILITY_THRESHOLD = get_float(
        "landmark_visibility_threshold",
        LANDMARK_VISIBILITY_THRESHOLD
    )

    print("Classification settings updated:")
    print("  Body visibility:", BODY_VISIBILITY_THRESHOLD)
    print("  Side shoulder width:", SIDE_SHOULDER_WIDTH)
    print("  Side hip width:", SIDE_HIP_WIDTH)
    print("  Back face visibility:", BACK_FACE_VISIBILITY)
    print("  Stomach face visibility:", STOMACH_FACE_VISIBILITY)
    print("  Landmark visibility:", LANDMARK_VISIBILITY_THRESHOLD)

# --------------------------------------------------
# MEDIAPIPE SETUP
# --------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(
    options
)

mediapipe_start_time = time.perf_counter()


# --------------------------------------------------
# LIVE STATISTICS
# --------------------------------------------------

stats_lock = threading.Lock()

current_position = "No Person Detected"
previous_position = None

position_times = {
    "Back": 0.0,
    "Left Side": 0.0,
    "Right Side": 0.0,
    "Stomach": 0.0,
    "Unknown": 0.0,
    "No Person Detected": 0.0
}

session_start_time = None
last_frame_time = time.time()

position_changes = 0
last_position_change = None

session_running = False


# --------------------------------------------------
# SESSION EXPORT
# --------------------------------------------------

LANDMARKS_TO_LOG = {
    "LEFT_EYE": 2,
    "RIGHT_EYE": 5,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24
}


def clean_session_name(session_name):

    if not session_name:
        session_name = "Sleep_Session"

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(session_name).strip()
    )

    safe_name = safe_name.strip("_")

    if not safe_name:
        safe_name = "Sleep_Session"

    return safe_name


def create_session_export(session_name):

    global current_session_name
    global current_export_file

    current_session_name = (
        str(session_name).strip()
        if session_name
        else "Sleep_Session"
    )

    os.makedirs(
        EXPORT_FOLDER,
        exist_ok=True
    )

    safe_session_name = clean_session_name(
        current_session_name
    )

    file_timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H%M%S"
    )

    filename = (
        f"{safe_session_name}_"
        f"{file_timestamp}.csv"
    )

    current_export_file = os.path.join(
        EXPORT_FOLDER,
        filename
    )

    header = [
        "Timestamp",
        "Session_Name",
        "Position"
    ]

    # Flat-file columns for selected landmarks only.
    for landmark_name in LANDMARKS_TO_LOG:

        header.extend([
            f"{landmark_name}_X",
            f"{landmark_name}_Y",
            f"{landmark_name}_Z",
            f"{landmark_name}_Visibility"
        ])

    with open(
        current_export_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as export_file:

        writer = csv.writer(
            export_file
        )

        writer.writerow(
            header
        )

    print(
        "Export file created:",
        current_export_file
    )

    return current_export_file


def log_position_and_landmarks(
    timestamp,
    position,
    landmarks
):

    if current_export_file is None:
        return

    row = [
        timestamp,
        current_session_name,
        position
    ]

    if landmarks is not None:

        for landmark_name, landmark_index in LANDMARKS_TO_LOG.items():

            landmark = landmarks[
                landmark_index
            ]

            row.extend([
                round(landmark.x, 6),
                round(landmark.y, 6),
                round(landmark.z, 6),
                round(landmark.visibility, 6)
            ])

    else:

        # 8 landmarks x 4 values each
        row.extend(
            [""] * (8 * 4)
        )

    try:

        with open(
            current_export_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as export_file:

            writer = csv.writer(
                export_file
            )

            writer.writerow(
                row
            )

    except Exception as error:

        print(
            "CSV logging error:",
            error
        )


# --------------------------------------------------
# START SESSION STATISTICS
# --------------------------------------------------

def start_session_statistics(session_name="Sleep_Session"):

    global current_position
    global previous_position
    global session_start_time
    global last_frame_time
    global position_changes
    global last_position_change
    global session_running

    with stats_lock:

        current_position = "No Person Detected"
        previous_position = None

        for position in position_times:
            position_times[position] = 0.0

        session_start_time = time.time()
        last_frame_time = time.time()

        position_changes = 0
        last_position_change = None

        session_running = True

    export_file = create_session_export(
        session_name
    )

    print("Sleep statistics session started.")
    print(
        "Session name:",
        current_session_name
    )
    print(
        "Export file:",
        export_file
    )

    return export_file


# --------------------------------------------------
# STOP SESSION STATISTICS
# --------------------------------------------------

def stop_session_statistics():

    global session_running

    with stats_lock:

        was_running = session_running
        final_position = current_position

        session_running = False

    # Write one final timestamp to the CSV.
    #
    # This gives the Analysis page the true end time
    # of the final sleep-position segment.
    #
    # The position is intentionally repeated, so it
    # does NOT count as another position change.
    if (
        was_running
        and current_export_file is not None
    ):

        final_timestamp = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        log_position_and_landmarks(
            final_timestamp,
            final_position,
            None
        )

        print(
            f"Logged session end: "
            f"{final_timestamp}, "
            f"{current_session_name}, "
            f"{final_position}"
        )

    print("Sleep statistics session stopped.")


# --------------------------------------------------
# SLEEP POSITION CLASSIFICATION
# --------------------------------------------------

def classify_sleep_position(landmarks):

    # MediaPipe pose landmark numbers
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    nose = landmarks[NOSE]

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]

    left_ear = landmarks[LEFT_EAR]
    right_ear = landmarks[RIGHT_EAR]

    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]

    left_hip = landmarks[LEFT_HIP]
    right_hip = landmarks[RIGHT_HIP]

    # --------------------------------------------------
    # BODY WIDTH
    # --------------------------------------------------

    shoulder_width = abs(
        left_shoulder.x - right_shoulder.x
    )

    hip_width = abs(
        left_hip.x - right_hip.x
    )

    # --------------------------------------------------
    # FACE VISIBILITY
    # --------------------------------------------------

    face_visibility = (
        nose.visibility
        + left_eye.visibility
        + right_eye.visibility
    ) / 3

    # --------------------------------------------------
    # BODY VISIBILITY
    # --------------------------------------------------

    body_visibility = (
        left_shoulder.visibility
        + right_shoulder.visibility
        + left_hip.visibility
        + right_hip.visibility
    ) / 4

    # --------------------------------------------------
    # NOT ENOUGH LANDMARKS
    # --------------------------------------------------

    if body_visibility < BODY_VISIBILITY_THRESHOLD:
        return "Unknown"

    # --------------------------------------------------
    # SIDE SLEEPING
    # --------------------------------------------------

    if (
        shoulder_width < SIDE_SHOULDER_WIDTH
        or hip_width < SIDE_HIP_WIDTH
    ):

        left_visibility = (
            left_shoulder.visibility
            + left_hip.visibility
            + left_ear.visibility
        ) / 3

        right_visibility = (
            right_shoulder.visibility
            + right_hip.visibility
            + right_ear.visibility
        ) / 3

        # ----------------------------------------------
        # SLEEPER PERSPECTIVE
        # ----------------------------------------------
        #
        # If the sleeper is lying on their LEFT side,
        # their RIGHT side is generally more exposed.
        #
        # If the sleeper is lying on their RIGHT side,
        # their LEFT side is generally more exposed.
        # ----------------------------------------------

        if right_visibility > left_visibility:
            return "Left Side"

        if left_visibility > right_visibility:
            return "Right Side"

        return "Unknown"

    # --------------------------------------------------
    # BACK
    # --------------------------------------------------

    if face_visibility > BACK_FACE_VISIBILITY:
        return "Back"

    # --------------------------------------------------
    # STOMACH
    # --------------------------------------------------

    if face_visibility < STOMACH_FACE_VISIBILITY:
        return "Stomach"

    # --------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------

    return "Unknown"


# --------------------------------------------------
# PROCESS FRAME
# --------------------------------------------------

def process_frame(frame):

    global current_position
    global previous_position
    global last_frame_time
    global position_changes
    global last_position_change

    now = time.time()

    elapsed = now - last_frame_time
    last_frame_time = now

    # --------------------------------------------------
    # CONVERT FRAME TO RGB
    # --------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------------------
    # MEDIAPIPE TIMESTAMP
    # --------------------------------------------------

    timestamp = int(
        (
            time.perf_counter()
            - mediapipe_start_time
        ) * 1000
    )

    # --------------------------------------------------
    # DETECT POSE
    # --------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    position = "No Person Detected"
    landmarks = None

    # --------------------------------------------------
    # PERSON DETECTED
    # --------------------------------------------------

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        position = classify_sleep_position(
            landmarks
        )

        # --------------------------------------------------
        # DRAW LANDMARKS
        # --------------------------------------------------

        for landmark in landmarks:

            if landmark.visibility > LANDMARK_VISIBILITY_THRESHOLD:

                x = int(
                    landmark.x
                    * frame.shape[1]
                )

                y = int(
                    landmark.y
                    * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

    # --------------------------------------------------
    # UPDATE STATISTICS
    # --------------------------------------------------

    with stats_lock:

        current_position = position

        # Only record stats while session is running
        if session_running:

            if position in position_times:
                position_times[position] += elapsed

            # --------------------------------------------------
            # POSITION CHANGED
            # --------------------------------------------------

            if (
                previous_position is not None
                and position != previous_position
            ):

                position_changes += 1

                last_position_change = (
                    datetime.now().strftime(
                        "%H:%M:%S"
                    )
                )

            # --------------------------------------------------
            # LOG POSITION CHANGE
            # --------------------------------------------------

            if position != previous_position:

                current_datetime = (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

                log_position_and_landmarks(
                    current_datetime,
                    position,
                    landmarks
                )

                print(
                    f"Logged: "
                    f"{current_datetime}, "
                    f"{current_session_name}, "
                    f"{position}"
                )

            previous_position = position

    # --------------------------------------------------
    # DISPLAY POSITION ON VIDEO
    # --------------------------------------------------

    cv2.putText(
        frame,
        f"Position: {position}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return frame


# --------------------------------------------------
# GET CURRENT POSITION
# --------------------------------------------------

def get_current_position():

    with stats_lock:
        return current_position


# --------------------------------------------------
# GET LIVE STATISTICS
# --------------------------------------------------

def get_position_counts():

    with stats_lock:

        statistics = {}

        total_time = sum(
            position_times.values()
        )

        for position, seconds in position_times.items():

            if total_time > 0:

                percentage = (
                    seconds / total_time
                ) * 100

            else:

                percentage = 0.0

            statistics[position] = {

                "seconds":
                    round(seconds, 1),

                "percentage":
                    round(percentage, 1)
            }

        # --------------------------------------------------
        # SESSION TIMER
        # --------------------------------------------------

        if (
            session_running
            and session_start_time is not None
        ):

            monitoring_seconds = int(
                time.time()
                - session_start_time
            )

        elif session_start_time is not None:

            monitoring_seconds = int(
                sum(
                    position_times.values()
                )
            )

        else:

            monitoring_seconds = 0

        return {

            "current_position":
                current_position,

            "monitoring_seconds":
                monitoring_seconds,

            "position_changes":
                position_changes,

            "last_position_change":
                last_position_change,

            "session_running":
                session_running,

            "positions":
                statistics
        }


# --------------------------------------------------
# CLOSE MEDIAPIPE
# --------------------------------------------------

def close_landmarker():

    try:
        landmarker.close()

    except Exception:
        pass