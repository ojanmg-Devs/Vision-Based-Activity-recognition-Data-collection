# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:59:01 2026

@author: Maj0131
"""

import cv2
import time
import numpy as np
import csv
import os
from pypylon import pylon
from ultralytics import YOLO

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "activity_data.csv")
SNAP_DIR = os.path.join(SCRIPT_DIR, "snapshots")
RECORDING_LENGTH = 15

print(f"Saving data to: {SCRIPT_DIR}")  
# Added 56: Chair, 60: Dining Table
TARGET_OBJECTS = [39, 41, 56, 60, 63, 64, 66, 67, 73] 

os.makedirs(SNAP_DIR, exist_ok=True)

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), 
    (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

LABELS = {
    '1': 'Sitting',
    '2': 'Standing',
    '3': 'Walking',
    '4': 'Sited nap',
    '5': 'Using computer',
    '6': 'Smartphone sitting',
    '7': 'Smartphone standing',
    '8': 'Talking phone sitting',
    '9': 'Talking phone standing',
    'a': 'Writing pen',
    'b': 'Reading table',
    'c': 'Reading hands',
    'd': 'Reading standing',
    'e': 'Eating meal',
    'f': 'Eating sandwich',
    'g': 'Eating snacks'
}




# --- INITIALIZE MODELS ---
try:
    print("Loading YOLO models...")
    pose_model = YOLO('yolov8n-pose.pt')
    obj_model = YOLO('yolov8n.pt')
    print("Models loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load models: {e}")
    exit(1)

# --- BASLER SETUP ---
camera = None
try:
    print("Initializing Basler camera...")
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # 1. You MUST enable the "Enable" switch first
    camera.AcquisitionFrameRateEnable.SetValue(True)

    # 2. Then set the actual value
    camera.AcquisitionFrameRate.SetValue(15.0)

    # 3. Now start the stream
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("Camera started successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize camera: {e}")
    exit(1)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
cv2.namedWindow('Neural Data Collector', cv2.WINDOW_NORMAL)

# --- ROI / CROP SETUP ---




def initialize_csv():
    if not os.path.exists(DATA_FILE):
        header = []
        for i in range(17):
            header.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"])
        # nearest_obj_id will now likely be '56' for chair when sitting
        header.extend(["nearest_obj_id", "nearest_obj_dist", "label", "snap_ref"])
        with open(DATA_FILE, mode='w', newline='') as f:
            csv.writer(f).writerow(header)

initialize_csv()

def draw_scene(img, kpts, objs):
    # Draw Skeleton
    for start, end in SKELETON_CONNECTIONS:
        p1, p2 = kpts[start], kpts[end]
        if p1[2] > 0.5 and p2[2] > 0.5:
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)
    
    # Draw Objects (Chairs, Tables, etc.)
    for obj in objs:
        ox1, oy1, ox2, oy2, conf, cls = obj
        color = (0, 255, 0) # Default Green
        if int(cls) == 56: color = (255, 0, 255) # Pink for Chairs
        if int(cls) == 60: color = (255, 165, 0) # Orange for Tables
        
        cv2.rectangle(img, (int(ox1), int(oy1)), (int(ox2), int(oy2)), color, 2)
        label_text = f"ID:{int(cls)}"
        if int(cls) == 56: label_text = "CHAIR"
        if int(cls) == 60: label_text = "TABLE"
        cv2.putText(img, label_text, (int(ox1), int(oy1)-5), 1, 1, color, 1)


def window_is_open(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:
        return False

recording_buffer = []
is_recording = False
current_label = ""




try:
    while camera.IsGrabbing():
        grabResult = None
        try:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grabResult.GrabSucceeded():
                continue

            image = converter.Convert(grabResult)
            frame = image.GetArray()
            display_frame = frame.copy()

            p_res = pose_model(frame, verbose=False, conf=0.4)[0]
            current_row = None

            if p_res.keypoints and len(p_res.keypoints.data) > 0:
                o_res = obj_model(frame, classes=TARGET_OBJECTS, verbose=False, conf=0.2)[0]
                kpts = p_res.keypoints.data[0].cpu().numpy()
                objs = o_res.boxes.data.cpu().numpy() if len(o_res.boxes) > 0 else []
                draw_scene(display_frame, kpts, objs)

                # --- NEAREST OBJECT CONTEXT ---
                nearest_id, min_dist = -1, 999.0
                if len(objs) > 0:
                    for o in objs:
                        person_center = ((kpts[11][0]+kpts[12][0])/2, (kpts[11][1]+kpts[12][1])/2)
                        obj_center = ((o[0]+o[2])/2, (o[1]+o[3])/2)
                        d = np.linalg.norm(np.array(person_center) - np.array(obj_center))
                        if d < min_dist:
                            min_dist, nearest_id = d, int(o[5])

                current_row = kpts.flatten().tolist()
                current_row.extend([nearest_id, round(min_dist, 2)])

                if is_recording:
                    row_to_save = current_row + [current_label, "None"]
                    recording_buffer.append(row_to_save)
                    cv2.putText(display_frame, f"REC: {len(recording_buffer)}/15", (10, 400), 1, 2, (0,0,255), 2)
                    if len(recording_buffer) >= RECORDING_LENGTH:
                        with open(DATA_FILE, mode='a', newline='') as f:
                            csv.writer(f).writerows(recording_buffer)
                        recording_buffer = []
                        is_recording = False
            else:
                cv2.putText(display_frame, "NO PERSON DETECTED", (10, 30), 1, 1.5, (0, 0, 255), 2)

            cv2.putText(display_frame, "[S] SNAP | [Q] QUIT", (10, 30), 1, 1.5, (0, 255, 255), 2)
            for i, (k, v) in enumerate(LABELS.items()):
                col = 0 if i < 8 else 250
                row_idx = i if i < 8 else i - 8
                cv2.putText(display_frame, f"[{k}] {v}", (10 + col, 70 + (row_idx * 25)), 1, 1.0, (255, 255, 0), 2)

            if not window_is_open('Neural Data Collector'):
                break

            cv2.imshow('Neural Data Collector', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and current_row is not None:
                try:
                    timestamp = int(time.time())
                    snap_name = f"snap_{timestamp}.jpg"
                    cv2.imwrite(os.path.join(SNAP_DIR, snap_name), display_frame)
                    snap_row = current_row + ["Snapshot", snap_name]
                    with open(DATA_FILE, mode='a', newline='') as f:
                        csv.writer(f).writerow(snap_row)
                    print(f"Snapshot saved: {snap_name}")
                except Exception as e:
                    print(f"ERROR: Failed to save snapshot: {e}")
            elif key != 255:
                try:
                    key_char = chr(key)
                    if key_char in LABELS and not is_recording:
                        is_recording = True
                        current_label = LABELS[key_char]
                        print(f"Recording: {current_label}")
                except (ValueError, KeyError):
                    pass
        except Exception as e:
            print(f"ERROR: Frame processing failed: {e}")
            continue
        finally:
            if grabResult is not None:
                grabResult.Release()
except KeyboardInterrupt:
    print("\nInterrupted by user.")
except Exception as e:
    print(f"ERROR: Camera grabbing loop failed: {e}")
finally:
    # Save any remaining buffered recording data
    if recording_buffer:
        try:
            with open(DATA_FILE, mode='a', newline='') as f:
                csv.writer(f).writerows(recording_buffer)
            print(f"Saved {len(recording_buffer)} remaining recordings to CSV.")
        except Exception as e:
            print(f"ERROR: Failed to save remaining data: {e}")
    
    # Clean up camera and OpenCV
    try:
        if camera is not None:
            camera.StopGrabbing()
            camera.Close()
            print("Camera closed successfully.")
    except Exception as e:
        print(f"ERROR: Failed to close camera: {e}")
    
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"ERROR: Failed to close windows: {e}")
    
    print("Data collection stopped. Files saved successfully.")