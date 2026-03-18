import cv2
import json
import base64
import time
import numpy as np
import face_recognition
from pathlib import Path
from datetime import date

KNOWN_FACES_DIR  = Path("known_faces")
VIDEO_PATH       = Path("video_sample_1.mov")
REPORT_HTML_OUT  = Path("report.html")
INTEGRATION_OUT  = Path("integration_output.json")

SCHOOL_NAME      = "IIT Mandi"
MATCH_THRESHOLD  = 0.55
MAX_KEYFRAMES    = 20


# ---------------- STEP 1 ----------------
def load_known_faces(folder: Path):
    known = {}

    for file in folder.iterdir():
        if file.suffix.lower() in [".jpg", ".png"]:
            img = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(img)

            if len(encodings) > 0:
                known[file.stem] = encodings
            else:
                print(f"Warning: no face in {file.name}")

    return known


# ---------------- STEP 2 ----------------
def extract_keyframes(video_path: Path, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    frames = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(2.0, (8,8))
            enhanced = clahe.apply(gray)
            frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            frames.append((idx, frame))

        idx += 1

    cap.release()
    return frames


# ---------------- STEP 3 ----------------
def detect_and_match(frame, known, threshold):
    detections = []
    rgb = frame[:, :, ::-1]

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    unknown_id = 1

    for enc, loc in zip(encodings, locations):
        best_name = None
        best_dist = 999

        for name, enc_list in known.items():
            distances = face_recognition.face_distance(enc_list, enc)
            dist = min(distances)

            if dist < best_dist:
                best_dist = dist
                best_name = name

        matched = best_dist < threshold

        if not matched:
            best_name = f"UNKNOWN_{unknown_id:03d}"
            unknown_id += 1

        confidence = max(0, 1 - best_dist)

        top, right, bottom, left = loc
        face_crop = frame[top:bottom, left:right]

        detections.append({
            "name": best_name,
            "matched": matched,
            "confidence": float(confidence),
            "bbox": (left, top, right-left, bottom-top),
            "face_crop": face_crop
        })

    return detections


# ---------------- STEP 4 ----------------
def compute_face_brightness(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 2.55)


def compute_eye_openness(face):
    return 50.0  # fallback


def compute_movement(prev, curr, bbox):
    if prev is None:
        return 0.0

    x, y, w, h = bbox

    prev_crop = cv2.cvtColor(prev[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    curr_crop = cv2.cvtColor(curr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_crop, curr_crop,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])

    return float(np.mean(mag) * 10)


# ---------------- STEP 5 ----------------
def encode_b64(img):
    img = cv2.resize(img, (240,240))
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def verdict(score):
    return "high" if score >= 75 else "moderate" if score >= 50 else "low"


def aggregate_persons(detections):
    persons = {}

    for d in detections:
        name = d["name"]
        persons.setdefault(name, []).append(d)

    output = []
    pid = 1

    for name, items in persons.items():
        brightness = np.mean([i["brightness"] for i in items])
        eye = np.mean([i["eye_openness"] for i in items])
        move = np.mean([i["movement"] for i in items])

        score = brightness*0.35 + eye*0.30 + move*0.35

        best_face = max(items, key=lambda x: x["confidence"])["face_crop"]

        output.append({
            "person_id": f"SCHOOL_P{pid:04d}",
            "name": name,
            "matched": items[0]["matched"],
            "match_confidence": float(np.mean([i["confidence"] for i in items])),
            "profile_image_b64": encode_b64(best_face),
            "frames_detected": len(items),
            "energy_score": round(score,1),
            "energy_breakdown": {
                "face_brightness": round(brightness,1),
                "eye_openness": round(eye,1),
                "movement_activity": round(move,1)
            },
            "verdict": verdict(score),
            "first_seen_frame": min(i["frame_idx"] for i in items),
            "last_seen_frame": max(i["frame_idx"] for i in items)
        })

        pid += 1

    return output


# ---------------- STEP 6 ----------------
def generate_report(persons, path):
    html = "<html><body><h1>Energy Report</h1>"

    for p in persons:
        html += f"""
        <div style='border:1px solid #ccc;padding:10px;margin:10px'>
        <img src="data:image/jpeg;base64,{p['profile_image_b64']}" width="120"><br>
        <b>{p['name']}</b><br>
        Energy: {p['energy_score']} ({p['verdict']})
        </div>
        """

    html += "</body></html>"

    with open(path, "w") as f:
        f.write(html)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    t0 = time.time()

    print("Step 1 — loading known faces ...")
    known = load_known_faces(KNOWN_FACES_DIR)

    if not known:
        print("ERROR: no faces loaded")
        exit()

    print("Step 2 — extracting keyframes ...")
    frames = extract_keyframes(VIDEO_PATH, MAX_KEYFRAMES)

    all_detections = []
    prev = None

    for idx, frame in frames:
        dets = detect_and_match(frame, known, MATCH_THRESHOLD)

        for d in dets:
            d["frame_idx"] = idx
            d["brightness"] = compute_face_brightness(d["face_crop"])
            d["eye_openness"] = compute_eye_openness(d["face_crop"])
            d["movement"] = compute_movement(prev, frame, d["bbox"])

        all_detections.extend(dets)
        prev = frame

    persons = aggregate_persons(all_detections)

    t1 = round(time.time()-t0,2)

    generate_report(persons, REPORT_HTML_OUT)

    with open(INTEGRATION_OUT,"w") as f:
        json.dump({
            "source": "p1_identity_energy",
            "school": SCHOOL_NAME,
            "date": str(date.today()),
            "video_file": str(VIDEO_PATH),
            "total_persons_matched": sum(p["matched"] for p in persons),
            "total_persons_unknown": sum(not p["matched"] for p in persons),
            "processing_time_sec": t1,
            "persons": persons
        }, f, indent=2)

    print("DONE")