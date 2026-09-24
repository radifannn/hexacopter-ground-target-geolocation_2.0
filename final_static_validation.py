import csv
import math
import re
import subprocess
import threading
import time

import cv2
import numpy as np
from pymavlink import mavutil

import geolocation_geometry as geo


# ============================================================
# Configuration
# ============================================================

CAMERA_PIPELINE = (
    'udpsrc port=5600 caps="application/x-rtp, media=video, '
    'encoding-name=H264, payload=96" ! '
    'rtph264depay ! h264parse ! avdec_h264 ! '
    'videoconvert ! appsink drop=true sync=false'
)

POSE_TOPIC = (
    "/world/hexacopter_runway/dynamic_pose/info"
)

MAVLINK_CONNECTION = (
    "tcp:127.0.0.1:5762"
)

TARGET_Z = 0.01

N_SAMPLES = 100

SAMPLE_INTERVAL = 0.10

OUTPUT_CSV = (
    "/home/rifqi-radifan/gz_ws/geolocation/"
    "final_static_validation.csv"
)


# ============================================================
# Detector
# ============================================================

LOWER_RED_1 = np.array([
    0, 100, 80
])

UPPER_RED_1 = np.array([
    10, 255, 255
])

LOWER_RED_2 = np.array([
    170, 100, 80
])

UPPER_RED_2 = np.array([
    180, 255, 255
])

KERNEL = np.ones(
    (3, 3),
    np.uint8
)


# ============================================================
# Shared state
# ============================================================

state_lock = threading.Lock()

state = {
    "uav_world": None,
    "camera_world": None,
    "R_world_pitch": None,

    "target_world": None,

    "pose_timestamp": None,

    "gps_lat": None,
    "gps_lon": None,

    "gps_timestamp": None,

    "pose_error": None,
    "gps_error": None
}

stop_event = threading.Event()


# ============================================================
# Pose parsing
# ============================================================

def parse_pose_block(block):

    name_match = re.search(
        r'name:\s*"([^"]+)"',
        block
    )

    if name_match is None:
        return None

    name = name_match.group(1)

    position_match = re.search(
        r'position\s*\{(.*?)\}',
        block,
        re.DOTALL
    )

    orientation_match = re.search(
        r'orientation\s*\{(.*?)\}',
        block,
        re.DOTALL
    )

    if (
        position_match is None
        or orientation_match is None
    ):
        return None

    position_text = (
        position_match.group(1)
    )

    orientation_text = (
        orientation_match.group(1)
    )

    def value(
        pattern,
        source,
        default=0.0
    ):

        match = re.search(
            pattern,
            source
        )

        if match:
            return float(
                match.group(1)
            )

        return default

    return {
        "name": name,

        "position": np.array([
            value(
                r'x:\s*([-+0-9.eE]+)',
                position_text
            ),
            value(
                r'y:\s*([-+0-9.eE]+)',
                position_text
            ),
            value(
                r'z:\s*([-+0-9.eE]+)',
                position_text
            )
        ]),

        "quaternion": np.array([
            value(
                r'x:\s*([-+0-9.eE]+)',
                orientation_text
            ),
            value(
                r'y:\s*([-+0-9.eE]+)',
                orientation_text
            ),
            value(
                r'z:\s*([-+0-9.eE]+)',
                orientation_text
            ),
            value(
                r'w:\s*([-+0-9.eE]+)',
                orientation_text,
                1.0
            )
        ])
    }


def parse_header_timestamp(block):

    sec_match = re.search(
        r'sec:\s*(\d+)',
        block
    )

    nsec_match = re.search(
        r'nsec:\s*(\d+)',
        block
    )

    if sec_match is None:
        return None

    sec = int(
        sec_match.group(1)
    )

    nsec = (
        int(nsec_match.group(1))
        if nsec_match
        else 0
    )

    return (
        sec
        + nsec * 1e-9
    )


# ============================================================
# Camera pose composition
# ============================================================

def compose_camera_state(poses):

    required = {
        "hexacopter_with_ardupilot",
        "gimbal",
        "pitch_link"
    }

    if not required.issubset(
        poses.keys()
    ):
        return None

    uav = poses[
        "hexacopter_with_ardupilot"
    ]

    gimbal = poses[
        "gimbal"
    ]

    pitch_link = poses[
        "pitch_link"
    ]

    R_world_uav = (
        geo.quaternion_to_rotation_matrix(
            uav["quaternion"]
        )
    )

    R_uav_gimbal = (
        geo.quaternion_to_rotation_matrix(
            gimbal["quaternion"]
        )
    )

    R_gimbal_pitch = (
        geo.quaternion_to_rotation_matrix(
            pitch_link["quaternion"]
        )
    )

    R_world_pitch = (
        R_world_uav
        @ R_uav_gimbal
        @ R_gimbal_pitch
    )

    camera_uav = (
        gimbal["position"]
        + R_uav_gimbal
        @ pitch_link["position"]
    )

    camera_world = (
        uav["position"]
        + R_world_uav
        @ camera_uav
    )

    return (
        uav["position"],
        camera_world,
        R_world_pitch
    )


# ============================================================
# Gazebo persistent stream
# ============================================================

def gazebo_pose_worker():

    process = None

    try:

        process = subprocess.Popen(
            [
                "stdbuf",
                "-oL",
                "gz",
                "topic",
                "-e",
                "-t",
                POSE_TOPIC
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        pose_lines = []
        pose_depth = 0
        inside_pose = False

        header_lines = []
        header_depth = 0
        inside_header = False

        pending_poses = {}

        required = {
            "hexacopter_with_ardupilot",
            "gimbal",
            "pitch_link",
            "moving_target"
        }

        for line in process.stdout:

            if stop_event.is_set():
                break

            stripped = line.strip()

            # ------------------------------------------------
            # Pose
            # ------------------------------------------------

            if (
                not inside_pose
                and stripped == "pose {"
            ):

                inside_pose = True

                pose_depth = 1

                pose_lines = [
                    stripped
                ]

                continue

            if inside_pose:

                pose_lines.append(
                    stripped
                )

                pose_depth += (
                    stripped.count("{")
                    - stripped.count("}")
                )

                if pose_depth == 0:

                    inside_pose = False

                    block = "\n".join(
                        pose_lines
                    )

                    pose = parse_pose_block(
                        block
                    )

                    if pose is not None:

                        if pose["name"] in required:

                            pending_poses[
                                pose["name"]
                            ] = pose

                continue

            # ------------------------------------------------
            # Header
            # ------------------------------------------------

            if (
                not inside_header
                and stripped == "header {"
            ):

                inside_header = True

                header_depth = 1

                header_lines = [
                    stripped
                ]

                continue

            if inside_header:

                header_lines.append(
                    stripped
                )

                header_depth += (
                    stripped.count("{")
                    - stripped.count("}")
                )

                if header_depth == 0:

                    inside_header = False

                    header_block = "\n".join(
                        header_lines
                    )

                    timestamp = (
                        parse_header_timestamp(
                            header_block
                        )
                    )

                    if (
                        timestamp is not None
                        and required.issubset(
                            pending_poses.keys()
                        )
                    ):

                        try:

                            (
                                uav_world,
                                camera_world,
                                R_world_pitch
                            ) = compose_camera_state(
                                pending_poses
                            )

                            target_world = (
                                pending_poses[
                                    "moving_target"
                                ]["position"].copy()
                            )

                            with state_lock:

                                state[
                                    "uav_world"
                                ] = (
                                    uav_world.copy()
                                )

                                state[
                                    "camera_world"
                                ] = (
                                    camera_world.copy()
                                )

                                state[
                                    "R_world_pitch"
                                ] = (
                                    R_world_pitch.copy()
                                )

                                state[
                                    "target_world"
                                ] = (
                                    target_world
                                )

                                state[
                                    "pose_timestamp"
                                ] = time.monotonic()

                                state[
                                    "pose_error"
                                ] = None

                        except Exception as exc:

                            with state_lock:

                                state[
                                    "pose_error"
                                ] = str(exc)

                    pending_poses = {}

    except Exception as exc:

        with state_lock:

            state[
                "pose_error"
            ] = str(exc)

    finally:

        if process is not None:

            try:
                process.terminate()
                process.wait(
                    timeout=1.0
                )
            except Exception:

                try:
                    process.kill()
                except Exception:
                    pass


# ============================================================
# MAVLink GPS worker
# ============================================================

def gps_worker():

    try:

        master = mavutil.mavlink_connection(
            MAVLINK_CONNECTION
        )

        master.wait_heartbeat(
            timeout=5
        )

        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            200000,
            0, 0, 0, 0, 0
        )

        while not stop_event.is_set():

            msg = master.recv_match(
                type="GLOBAL_POSITION_INT",
                blocking=True,
                timeout=0.25
            )

            if msg is None:
                continue

            with state_lock:

                state[
                    "gps_lat"
                ] = msg.lat / 1e7

                state[
                    "gps_lon"
                ] = msg.lon / 1e7

                state[
                    "gps_timestamp"
                ] = time.monotonic()

                state[
                    "gps_error"
                ] = None

    except Exception as exc:

        with state_lock:

            state[
                "gps_error"
            ] = str(exc)


# ============================================================
# Detector
# ============================================================

def detect_target(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask1 = cv2.inRange(
        hsv,
        LOWER_RED_1,
        UPPER_RED_1
    )

    mask2 = cv2.inRange(
        hsv,
        LOWER_RED_2,
        UPPER_RED_2
    )

    mask = cv2.bitwise_or(
        mask1,
        mask2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        KERNEL
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        KERNEL
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        contour
    )

    if area < 10:
        return None

    moments = cv2.moments(
        contour
    )

    if moments["m00"] == 0:
        return None

    u = (
        moments["m10"]
        / moments["m00"]
    )

    v = (
        moments["m01"]
        / moments["m00"]
    )

    x, y, w, h = (
        cv2.boundingRect(
            contour
        )
    )

    return {
        "u": float(u),
        "v": float(v),
        "bbox": (
            x, y, w, h
        ),
        "area": float(area)
    }


# ============================================================
# Pixel -> NED
# ============================================================

def pixel_to_ned_ray(
    u,
    v,
    R_world_pitch
):

    ray_camera = (
        geo.pixel_to_camera_ray(
            u,
            v
        )
    )

    ray_gazebo_camera = (
        geo.R_GAZEBO_FROM_CAMERA
        @ ray_camera
    )

    ray_pitch = (
        geo.R_SENSOR
        @ ray_gazebo_camera
    )

    ray_world = (
        R_world_pitch
        @ ray_pitch
    )

    R_ned_from_gazebo = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ], dtype=float)

    ray_ned = (
        R_ned_from_gazebo
        @ ray_world
    )

    ray_ned /= np.linalg.norm(
        ray_ned
    )

    return ray_ned


# ============================================================
# Estimate target
# ============================================================

def estimate_target(
    u,
    v,
    uav_world,
    camera_world,
    R_world_pitch,
    gps_lat,
    gps_lon
):

    ray_ned = pixel_to_ned_ray(
        u,
        v,
        R_world_pitch
    )

    if ray_ned[2] <= 0:

        raise RuntimeError(
            "Ray does not intersect target plane."
        )

    camera_delta = (
        camera_world
        - uav_world
    )

    camera_ned = np.array([
        camera_delta[1],
        camera_delta[0],
        -camera_delta[2]
    ])

    vertical_distance = (
        camera_world[2]
        - TARGET_Z
    )

    if vertical_distance <= 0:

        raise RuntimeError(
            "Target plane is not below camera."
        )

    scale = (
        vertical_distance
        / ray_ned[2]
    )

    target_ned = (
        camera_ned
        + scale * ray_ned
    )

    target_lat, target_lon = (
        geo.ned_to_gps(
            gps_lat,
            gps_lon,
            target_ned[0],
            target_ned[1]
        )
    )

    return {
        "target_ned": target_ned,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "ray_ned": ray_ned
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FINAL STATIC GEOLOCATION VALIDATION")
    print("=" * 70)

    print()
    print(f"Samples          : {N_SAMPLES}")
    print(f"Sample interval  : {SAMPLE_INTERVAL:.2f} s")
    print(f"Target plane Z   : {TARGET_Z:.3f} m")
    print("Pixel measurement: contour centroid")
    print()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_PIPELINE,
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera stream."
        )

    # --------------------------------------------------------
    # Start state workers
    # --------------------------------------------------------

    pose_thread = threading.Thread(
        target=gazebo_pose_worker,
        daemon=True
    )

    gps_thread = threading.Thread(
        target=gps_worker,
        daemon=True
    )

    pose_thread.start()
    gps_thread.start()

    # --------------------------------------------------------
    # Wait until state is ready
    # --------------------------------------------------------

    print("Waiting for state streams...")

    while True:

        with state_lock:

            ready = (
                state["uav_world"] is not None
                and
                state["camera_world"] is not None
                and
                state["R_world_pitch"] is not None
                and
                state["target_world"] is not None
                and
                state["gps_lat"] is not None
                and
                state["gps_lon"] is not None
            )

        if ready:
            break

        if stop_event.is_set():
            raise RuntimeError(
                "State worker stopped."
            )

        time.sleep(0.05)

    print("State streams ready.")
    print()
    print("Collecting samples...")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    rows = []

    valid_samples = 0

    last_sample_time = 0.0

    while valid_samples < N_SAMPLES:

        ret, frame = cap.read()

        if not ret:
            continue

        now = time.monotonic()

        if (
            now - last_sample_time
            < SAMPLE_INTERVAL
        ):
            continue

        detection = detect_target(
            frame
        )

        if detection is None:

            continue

        # ----------------------------------------------------
        # Snapshot state atomically
        # ----------------------------------------------------

        with state_lock:

            uav_world = (
                state["uav_world"].copy()
            )

            camera_world = (
                state["camera_world"].copy()
            )

            R_world_pitch = (
                state["R_world_pitch"].copy()
            )

            target_world = (
                state["target_world"].copy()
            )

            gps_lat = state["gps_lat"]
            gps_lon = state["gps_lon"]

            pose_timestamp = (
                state["pose_timestamp"]
            )

            gps_timestamp = (
                state["gps_timestamp"]
            )

        pose_age = (
            time.monotonic()
            - pose_timestamp
        )

        gps_age = (
            time.monotonic()
            - gps_timestamp
        )

        try:

            # ------------------------------------------------
            # Actual detector measurement
            # ------------------------------------------------

            u = detection["u"]
            v = detection["v"]

            # ------------------------------------------------
            # Estimator
            # ------------------------------------------------

            estimate = estimate_target(
                u,
                v,
                uav_world,
                camera_world,
                R_world_pitch,
                gps_lat,
                gps_lon
            )

            estimated_ned = (
                estimate["target_ned"]
            )

            # ------------------------------------------------
            # Ground truth target relative UAV
            #
            # Gazebo:
            #   X = East
            #   Y = North
            #   Z = Up
            #
            # NED:
            #   N = +Y
            #   E = +X
            #   D = -Z
            # ------------------------------------------------

            target_delta = (
                target_world
                - uav_world
            )

            true_ned = np.array([
                target_delta[1],
                target_delta[0],
                -target_delta[2]
            ])

            # ------------------------------------------------
            # Error
            # ------------------------------------------------

            error_n = (
                estimated_ned[0]
                - true_ned[0]
            )

            error_e = (
                estimated_ned[1]
                - true_ned[1]
            )

            error_d = (
                estimated_ned[2]
                - true_ned[2]
            )

            error_horizontal = math.hypot(
                error_n,
                error_e
            )

        except Exception:

            continue

        valid_samples += 1

        last_sample_time = time.monotonic()

        row = {
            "sample": valid_samples,

            "u": u,
            "v": v,

            "true_north": true_ned[0],
            "true_east": true_ned[1],
            "true_down": true_ned[2],

            "estimated_north":
                estimated_ned[0],

            "estimated_east":
                estimated_ned[1],

            "estimated_down":
                estimated_ned[2],

            "error_north": error_n,
            "error_east": error_e,
            "error_down": error_d,

            "horizontal_error":
                error_horizontal,

            "pose_age": pose_age,
            "gps_age": gps_age,

            "target_lat":
                estimate["target_lat"],

            "target_lon":
                estimate["target_lon"]
        }

        rows.append(row)

        print(
            f"  [{valid_samples:03d}/{N_SAMPLES}] "
            f"pixel=({u:.2f},{v:.2f}) "
            f"N={estimated_ned[0]:+.3f} "
            f"E={estimated_ned[1]:+.3f} "
            f"error={error_horizontal:.4f} m"
        )

    stop_event.set()

    cap.release()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    north_error = np.array([
        row["error_north"]
        for row in rows
    ])

    east_error = np.array([
        row["error_east"]
        for row in rows
    ])

    down_error = np.array([
        row["error_down"]
        for row in rows
    ])

    horizontal_error = np.array([
        row["horizontal_error"]
        for row in rows
    ])

    u_values = np.array([
        row["u"]
        for row in rows
    ])

    v_values = np.array([
        row["v"]
        for row in rows
    ])

    pose_ages = np.array([
        row["pose_age"]
        for row in rows
    ])

    gps_ages = np.array([
        row["gps_age"]
        for row in rows
    ])

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL STATIC VALIDATION RESULTS")
    print("=" * 70)

    print()
    print(
        f"Valid samples = "
        f"{len(rows)}"
    )

    print()
    print("Pixel measurement:")

    print(
        f"  Mean u = "
        f"{np.mean(u_values):.4f} px"
    )

    print(
        f"  Mean v = "
        f"{np.mean(v_values):.4f} px"
    )

    print(
        f"  Std u  = "
        f"{np.std(u_values):.4f} px"
    )

    print(
        f"  Std v  = "
        f"{np.std(v_values):.4f} px"
    )

    print()
    print("Position bias:")

    print(
        f"  North = "
        f"{np.mean(north_error):+.4f} m"
    )

    print(
        f"  East  = "
        f"{np.mean(east_error):+.4f} m"
    )

    print(
        f"  Down  = "
        f"{np.mean(down_error):+.4f} m"
    )

    print()
    print("RMSE:")

    print(
        f"  North = "
        f"{math.sqrt(np.mean(north_error ** 2)):.4f} m"
    )

    print(
        f"  East  = "
        f"{math.sqrt(np.mean(east_error ** 2)):.4f} m"
    )

    print(
        f"  Down  = "
        f"{math.sqrt(np.mean(down_error ** 2)):.4f} m"
    )

    print(
        f"  Horizontal = "
        f"{math.sqrt(np.mean(horizontal_error ** 2)):.4f} m"
    )

    print()
    print("Horizontal error:")

    print(
        f"  MAE = "
        f"{np.mean(horizontal_error):.4f} m"
    )

    print(
        f"  Min = "
        f"{np.min(horizontal_error):.4f} m"
    )

    print(
        f"  Max = "
        f"{np.max(horizontal_error):.4f} m"
    )

    print()
    print("State freshness:")

    print(
        f"  Pose age mean = "
        f"{np.mean(pose_ages):.4f} s"
    )

    print(
        f"  Pose age max  = "
        f"{np.max(pose_ages):.4f} s"
    )

    print(
        f"  GPS age mean  = "
        f"{np.mean(gps_ages):.4f} s"
    )

    print(
        f"  GPS age max   = "
        f"{np.max(gps_ages):.4f} s"
    )

    print()
    print("CSV:")
    print(
        f"  {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
