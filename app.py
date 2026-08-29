from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request,
    send_from_directory,
    redirect,
    url_for
)

import cv2
import json
import os
import subprocess

from analysis import analyse_csv

from record_webcam import (
    get_frame,
    set_camera,
    start_recording,
    stop_recording,
    set_output_folders
)

from sleepmonitor import (
    start_session_statistics,
    stop_session_statistics,
    get_position_counts,
    set_export_folder,
    set_classification_settings
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    "camera_index": 0,
    "recordings_folder": os.path.join(BASE_DIR, "recordings"),
    "exports_folder": os.path.join(BASE_DIR, "exports"),
    "snapshots_folder": os.path.join(BASE_DIR, "snapshots"),
    "snapshot_interval": 180,
    "classification": {
        "body_visibility_threshold": 0.45,
        "side_shoulder_width": 0.12,
        "side_hip_width": 0.10,
        "back_face_visibility": 0.60,
        "stomach_face_visibility": 0.35,
        "landmark_visibility_threshold": 0.40
    }
}


def normalise_folder(path_value, fallback):
    value = str(path_value or "").strip()
    if not value:
        value = fallback
    value = os.path.expanduser(value)
    if not os.path.isabs(value):
        value = os.path.join(BASE_DIR, value)
    return os.path.abspath(value)


def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
                saved = json.load(config_file)
            if isinstance(saved, dict):
                config.update(saved)
        except (OSError, json.JSONDecodeError) as error:
            print("Configuration read error:", error)

    try:
        config["camera_index"] = int(config.get("camera_index", 0))
    except (TypeError, ValueError):
        config["camera_index"] = 0

    try:
        config["snapshot_interval"] = int(config.get("snapshot_interval", 180))
    except (TypeError, ValueError):
        config["snapshot_interval"] = 180

    if config["snapshot_interval"] not in {60, 180, 300, 600, 900}:
        config["snapshot_interval"] = 180

    classification_defaults = DEFAULT_CONFIG["classification"].copy()
    saved_classification = config.get("classification")

    if isinstance(saved_classification, dict):
        classification_defaults.update(saved_classification)

    config["classification"] = classification_defaults

    config["recordings_folder"] = normalise_folder(
        config.get("recordings_folder"), DEFAULT_CONFIG["recordings_folder"]
    )
    config["exports_folder"] = normalise_folder(
        config.get("exports_folder"), DEFAULT_CONFIG["exports_folder"]
    )
    config["snapshots_folder"] = normalise_folder(
        config.get("snapshots_folder"), DEFAULT_CONFIG["snapshots_folder"]
    )
    return config


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=4)


def apply_config(config=None):
    if config is None:
        config = load_config()
    for folder_key in ["recordings_folder", "exports_folder", "snapshots_folder"]:
        os.makedirs(config[folder_key], exist_ok=True)
    set_output_folders(
        recordings_folder=config["recordings_folder"],
        snapshots_folder=config["snapshots_folder"]
    )
    set_export_folder(config["exports_folder"])

    set_classification_settings(
        config["classification"]
    )

    return config


apply_config()


def detect_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            success, frame = cap.read()
            if success and frame is not None:
                available_cameras.append(index)
        cap.release()
    return available_cameras


@app.route("/")
def capture():
    config = load_config()
    cameras = detect_cameras()
    return render_template(
        "capture.html",
        cameras=cameras,
        default_camera_index=config["camera_index"],
        default_snapshot_interval=config["snapshot_interval"]
    )


@app.route("/configuration", methods=["GET", "POST"])
def configuration():
    config = load_config()
    cameras = detect_cameras()
    saved = request.args.get("saved") == "1"
    error = None

    if request.method == "POST":
        try:
            camera_index = int(request.form.get("camera_index", config["camera_index"]))
            snapshot_interval = int(
                request.form.get("snapshot_interval", config["snapshot_interval"])
            )
            if snapshot_interval not in {60, 180, 300, 600, 900}:
                snapshot_interval = 180

            def form_float(name, default_value):
                try:
                    value = float(request.form.get(name, default_value))
                except (TypeError, ValueError):
                    value = default_value

                return max(0.0, min(1.0, value))

            new_config = {
                "camera_index": camera_index,
                "recordings_folder": normalise_folder(
                    request.form.get("recordings_folder"),
                    DEFAULT_CONFIG["recordings_folder"]
                ),
                "exports_folder": normalise_folder(
                    request.form.get("exports_folder"),
                    DEFAULT_CONFIG["exports_folder"]
                ),
                "snapshots_folder": normalise_folder(
                    request.form.get("snapshots_folder"),
                    DEFAULT_CONFIG["snapshots_folder"]
                ),
                "snapshot_interval": snapshot_interval,
                "classification": {
                    "body_visibility_threshold": form_float(
                        "body_visibility_threshold",
                        0.45
                    ),
                    "side_shoulder_width": form_float(
                        "side_shoulder_width",
                        0.12
                    ),
                    "side_hip_width": form_float(
                        "side_hip_width",
                        0.10
                    ),
                    "back_face_visibility": form_float(
                        "back_face_visibility",
                        0.60
                    ),
                    "stomach_face_visibility": form_float(
                        "stomach_face_visibility",
                        0.35
                    ),
                    "landmark_visibility_threshold": form_float(
                        "landmark_visibility_threshold",
                        0.40
                    )
                }
            }

            for folder_key in ["recordings_folder", "exports_folder", "snapshots_folder"]:
                os.makedirs(new_config[folder_key], exist_ok=True)

            save_config(new_config)
            apply_config(new_config)
            return redirect(url_for("configuration", saved=1))

        except Exception as exception:
            error = str(exception)

    return render_template(
        "configuration.html",
        config=config,
        cameras=cameras,
        saved=saved,
        error=error
    )


@app.route("/browse_folder", methods=["POST"])
def browse_folder():
    data = request.get_json(silent=True) or {}
    initial_folder = str(data.get("current_path", "")).strip()

    if os.name != "nt":
        return jsonify({
            "status": "error",
            "message": "The Browse button currently supports Windows. You can still type the folder path manually."
        }), 400

    powershell_script = r'''Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = "Select a Sleep Monitor folder"
$dialog.ShowNewFolderButton = $true
if ($env:SLEEP_MONITOR_INITIAL -and (Test-Path $env:SLEEP_MONITOR_INITIAL)) {
    $dialog.SelectedPath = $env:SLEEP_MONITOR_INITIAL
}
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dialog.SelectedPath
}
'''

    environment = os.environ.copy()
    environment["SLEEP_MONITOR_INITIAL"] = initial_folder

    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", powershell_script],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120
        )
        selected_folder = result.stdout.strip()
        if selected_folder:
            return jsonify({"status": "success", "path": selected_folder})
        return jsonify({"status": "cancelled"})
    except Exception as exception:
        return jsonify({"status": "error", "message": str(exception)}), 500


def get_saved_sessions():
    config = load_config()
    exports_folder = config["exports_folder"]
    recordings_folder = config["recordings_folder"]
    os.makedirs(exports_folder, exist_ok=True)
    os.makedirs(recordings_folder, exist_ok=True)
    sessions = []

    try:
        csv_files = os.listdir(exports_folder)
    except OSError:
        csv_files = []

    for csv_filename in csv_files:
        if not csv_filename.lower().endswith(".csv"):
            continue

        session_name = os.path.splitext(csv_filename)[0]
        csv_path = os.path.join(exports_folder, csv_filename)
        video_filename = None
        video_path = None

        for extension in [".mp4", ".avi", ".mov", ".webm"]:
            possible_video = session_name + extension
            possible_video_path = os.path.join(recordings_folder, possible_video)
            if os.path.exists(possible_video_path):
                video_filename = possible_video
                video_path = possible_video_path
                break

        if video_filename is None:
            try:
                video_files = os.listdir(recordings_folder)
            except OSError:
                video_files = []

            prefix_parts = session_name.rsplit("_", 2)
            session_prefix = prefix_parts[0] if len(prefix_parts) == 3 else session_name
            matching_videos = []

            for filename in video_files:
                if not filename.lower().endswith((".avi", ".mp4", ".mov", ".webm")):
                    continue
                if filename.startswith(session_prefix + "_"):
                    full_path = os.path.join(recordings_folder, filename)
                    matching_videos.append((
                        abs(os.path.getmtime(full_path) - os.path.getmtime(csv_path)),
                        filename,
                        full_path
                    ))

            if matching_videos:
                matching_videos.sort(key=lambda item: item[0])
                _, video_filename, video_path = matching_videos[0]

        try:
            modified_time = os.path.getmtime(csv_path)
        except OSError:
            modified_time = 0

        sessions.append({
            "name": session_name,
            "csv_filename": csv_filename,
            "csv_path": csv_path,
            "video_filename": video_filename,
            "video_path": video_path,
            "modified_time": modified_time
        })

    sessions.sort(key=lambda session: session["modified_time"], reverse=True)
    return sessions



@app.route("/help")
def help_page():
    return render_template(
        "help.html"
    )


@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    config = load_config()
    sessions = get_saved_sessions()
    results = None
    comparison_results = []
    error = None
    selected_sessions = []

    if request.method == "POST":
        try:
            selected_sessions = request.form.getlist("selected_sessions")
            if not selected_sessions:
                raise ValueError("Please select at least one sleep session.")

            available_sessions = {session["name"]: session for session in sessions}
            for session_name in selected_sessions:
                session = available_sessions.get(session_name)
                if session is None:
                    continue
                session_results = analyse_csv(session["csv_path"])
                comparison_results.append({
                    "session_name": session_name,
                    "csv_filename": session["csv_filename"],
                    "video_filename": session["video_filename"],
                    "results": session_results
                })

            if not comparison_results:
                raise ValueError("The selected session files could not be analysed.")

            if len(comparison_results) == 1:
                results = comparison_results[0]["results"]

        except Exception as exception:
            error = str(exception)

    return render_template(
        "analysis.html",
        sessions=sessions,
        selected_sessions=selected_sessions,
        results=results,
        comparison_results=comparison_results,
        error=error,
        exports_folder=config["exports_folder"],
        recordings_folder=config["recordings_folder"]
    )


@app.route("/recordings/<path:filename>")
def recording_file(filename):
    config = load_config()
    return send_from_directory(config["recordings_folder"], filename)


def generate_frames(camera_index):
    set_camera(camera_index)
    while True:
        frame = get_frame()
        if frame is None:
            continue
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    config = load_config()
    camera_index = request.args.get(
        "camera_index",
        default=config["camera_index"],
        type=int
    )
    return Response(
        generate_frames(camera_index),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/select_camera", methods=["POST"])
def select_camera():
    config = load_config()
    data = request.get_json(silent=True) or {}
    camera_index = int(data.get("camera_index", config["camera_index"]))
    success = set_camera(camera_index)
    if success:
        return jsonify({"status": "success", "camera_index": camera_index})
    return jsonify({
        "status": "error",
        "message": "Unable to open selected camera."
    }), 400


@app.route("/start_recording", methods=["POST"])
def start():
    config = apply_config(load_config())
    data = request.get_json(silent=True) or {}
    camera_index = int(data.get("camera_index", config["camera_index"]))
    session_name = str(data.get("session_name", "Night_001")).strip()
    if not session_name:
        session_name = "Night_001"

    snapshot_interval = int(
        data.get("snapshot_interval", config["snapshot_interval"])
    )
    if snapshot_interval not in {60, 180, 300, 600, 900}:
        snapshot_interval = config["snapshot_interval"]

    try:
        filename = start_recording(
            camera_index=camera_index,
            session_name=session_name,
            snapshot_interval=snapshot_interval
        )
        export_file = start_session_statistics(session_name)

        print("--------------------------------")
        print("Recording started")
        print("Session:", session_name)
        print("Camera:", camera_index)
        print("Video:", filename)
        print("Sleep data:", export_file)
        print("--------------------------------")

        return jsonify({
            "status": "recording",
            "camera_index": camera_index,
            "session_name": session_name,
            "snapshot_interval": snapshot_interval,
            "filename": filename,
            "export_file": export_file
        })
    except Exception as error:
        print("Start recording error:", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@app.route("/stop_recording", methods=["POST"])
def stop():
    try:
        filename = stop_recording()
        stop_session_statistics()
        return jsonify({"status": "stopped", "filename": filename})
    except Exception as error:
        return jsonify({"status": "error", "message": str(error)}), 500


@app.route("/position_stats")
def position_stats():
    return jsonify(get_position_counts())


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
