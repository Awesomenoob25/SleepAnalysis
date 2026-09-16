import cv2
import os
import threading
import time
from datetime import datetime

from sleepmonitor import process_frame


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SNAPSHOT_INTERVAL_SECONDS = 180

RECORDINGS_FOLDER = "recordings"
SNAPSHOTS_FOLDER = "snapshots"




# --------------------------------------------------
# CONFIGURABLE OUTPUT FOLDERS
# --------------------------------------------------

def set_output_folders(
    recordings_folder=None,
    snapshots_folder=None
):
    """Update where recordings and snapshots are saved."""

    global RECORDINGS_FOLDER
    global SNAPSHOTS_FOLDER

    if recordings_folder:
        RECORDINGS_FOLDER = os.path.abspath(
            os.path.expanduser(str(recordings_folder))
        )

    if snapshots_folder:
        SNAPSHOTS_FOLDER = os.path.abspath(
            os.path.expanduser(str(snapshots_folder))
        )

    os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
    os.makedirs(SNAPSHOTS_FOLDER, exist_ok=True)

    print("Recordings folder:", RECORDINGS_FOLDER)
    print("Snapshots folder:", SNAPSHOTS_FOLDER)


# --------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------

camera = None
current_camera_index = None

video_writer = None
recording = False

current_filename = None
current_session_name = None

last_snapshot_time = 0.0
current_snapshot_folder = None


# RLock prevents locking when
# set_camera is called from get_frame
camera_lock = threading.RLock()


# --------------------------------------------------
# CLEAN SESSION NAME
# --------------------------------------------------

def clean_session_name(
    session_name
):

    if not session_name:

        session_name = (
            "Night_001"
        )

    safe_name = ""

    for character in str(
        session_name
    ).strip():

        if (
            character.isalnum()
            or character in "-_"
        ):

            safe_name += character

        elif character == " ":

            safe_name += "_"

    if not safe_name:

        safe_name = (
            "Night_001"
        )

    return safe_name


# --------------------------------------------------
# SET CAMERA
# --------------------------------------------------

def set_camera(
    camera_index=0
):

    global camera
    global current_camera_index


    camera_index = int(
        camera_index
    )


    with camera_lock:

        if (
            camera is not None

            and camera.isOpened()

            and current_camera_index
            == camera_index
        ):

            return True


        # Close old camera

        if camera is not None:

            try:

                camera.release()

            except Exception:

                pass

            camera = None


        print(

            f"Opening camera "
            f"{camera_index}..."
        )


        camera = cv2.VideoCapture(

            camera_index,

            cv2.CAP_DSHOW
        )


        if not camera.isOpened():

            camera = None

            print(

                f"Unable to open "
                f"camera {camera_index}."
            )

            return False


        current_camera_index = (
            camera_index
        )


        print(

            f"Camera {camera_index} "
            f"opened."
        )


        return True


# --------------------------------------------------
# SAVE SNAPSHOT
# --------------------------------------------------

def save_snapshot(
    frame
):

    global last_snapshot_time


    if (
        current_snapshot_folder
        is None
    ):

        return


    now = time.time()


    if (
        last_snapshot_time > 0

        and (
            now
            - last_snapshot_time
        )
        < SNAPSHOT_INTERVAL_SECONDS
    ):

        return


    timestamp = (
        datetime.now().strftime(
            "%Y-%m-%d_%H%M%S"
        )
    )


    snapshot_filename = (
        os.path.join(
            current_snapshot_folder,
            (
                f"snapshot_"
                f"{timestamp}.jpg"
            )
        )
    )


    success = cv2.imwrite(
        snapshot_filename,
        frame
    )


    if success:

        last_snapshot_time = (
            now
        )

        print(
            "Snapshot saved:",
            snapshot_filename
        )


# --------------------------------------------------
# GET FRAME
# --------------------------------------------------

def get_frame():

    global camera


    with camera_lock:

        if (
            camera is None

            or not camera.isOpened()
        ):

            if not set_camera(0):

                return None


        success, frame = (
            camera.read()
        )


    if (
        not success
        or frame is None
    ):

        return None


    # Keep a clean frame for
    # recording and snapshots

    clean_frame = (
        frame.copy()
    )


    # MediaPipe processing.
    # This draws green landmarks
    # and the detected position.

    display_frame = (
        process_frame(
            frame.copy()
        )
    )


    # --------------------------------------------------
    # RECORDING
    # --------------------------------------------------

    if recording:

        save_snapshot(
            clean_frame
        )


        if (
            video_writer
            is not None
        ):

            try:

                video_writer.write(
                    clean_frame
                )

            except Exception as error:

                print(
                    "Video writing error:",
                    error
                )


    # Browser receives the
    # processed frame with landmarks

    return display_frame


# --------------------------------------------------
# START RECORDING
# --------------------------------------------------

def start_recording(
    camera_index=0,
    session_name="Night_001",
    snapshot_interval=180
):

    global recording
    global video_writer
    global current_filename
    global current_session_name
    global last_snapshot_time
    global current_snapshot_folder
    global SNAPSHOT_INTERVAL_SECONDS


    if recording:

        raise RuntimeError(
            "A recording session "
            "is already running."
        )


    if not set_camera(
        camera_index
    ):

        raise RuntimeError(
            "Unable to open "
            "selected camera."
        )


    current_session_name = (
        str(session_name).strip()
    )


    if not current_session_name:

        current_session_name = (
            "Night_001"
        )


    allowed_snapshot_intervals = {
        60,
        180,
        300,
        600,
        900
    }

    snapshot_interval = int(
        snapshot_interval
    )

    if snapshot_interval not in allowed_snapshot_intervals:
        snapshot_interval = 180

    SNAPSHOT_INTERVAL_SECONDS = (
        snapshot_interval
    )


    safe_session_name = (
        clean_session_name(
            current_session_name
        )
    )


    session_timestamp = (
        datetime.now().strftime(
            "%Y-%m-%d_%H%M%S"
        )
    )


    # --------------------------------------------------
    # CREATE FOLDERS
    # --------------------------------------------------

    os.makedirs(
        RECORDINGS_FOLDER,
        exist_ok=True
    )


    os.makedirs(
        SNAPSHOTS_FOLDER,
        exist_ok=True
    )


    current_snapshot_folder = (
        os.path.join(
            SNAPSHOTS_FOLDER,
            (
                f"{safe_session_name}_"
                f"{session_timestamp}"
            )
        )
    )


    os.makedirs(
        current_snapshot_folder,
        exist_ok=True
    )


    # --------------------------------------------------
    # VIDEO FILE
    # --------------------------------------------------

    current_filename = (
        os.path.join(
            RECORDINGS_FOLDER,
            (
                f"{safe_session_name}_"
                f"{session_timestamp}.avi"
            )
        )
    )


    # --------------------------------------------------
    # CAMERA PROPERTIES
    # --------------------------------------------------

    with camera_lock:

        width = int(
            camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = camera.get(
            cv2.CAP_PROP_FPS
        )


    if width <= 0:

        width = 640


    if height <= 0:

        height = 480


    if (
        fps <= 0
        or fps > 120
    ):

        fps = 20.0


    # --------------------------------------------------
    # CREATE VIDEO WRITER
    # --------------------------------------------------

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"XVID"
        )
    )


    video_writer = (
        cv2.VideoWriter(
            current_filename,
            fourcc,
            fps,
            (
                width,
                height
            )
        )
    )


    if not video_writer.isOpened():

        video_writer = None

        raise RuntimeError(
            "Unable to create "
            "video recording file."
        )


    last_snapshot_time = 0.0

    recording = True


    print(
        "Recording started."
    )

    print(
        "Session:",
        current_session_name
    )

    print(
        "Snapshot interval:",
        f"{SNAPSHOT_INTERVAL_SECONDS // 60} minute(s)"
    )

    print(
        "Video file:",
        current_filename
    )

    print(
        "Snapshot folder:",
        current_snapshot_folder
    )


    return current_filename


# --------------------------------------------------
# STOP RECORDING
# --------------------------------------------------

def stop_recording():

    global recording
    global video_writer


    recording = False


    if video_writer is not None:

        try:

            video_writer.release()

        except Exception:

            pass


        video_writer = None


    print(
        "Recording stopped."
    )


    return current_filename


# --------------------------------------------------
# CLOSE CAMERA
# --------------------------------------------------

def close_camera():

    global camera
    global current_camera_index


    with camera_lock:

        if camera is not None:

            try:

                camera.release()

            except Exception:

                pass


            camera = None


        current_camera_index = (
            None
        )
