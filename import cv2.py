import cv2
from time import time
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
from paddleocr import PaddleOCR
import csv
import os


class SpeedEstimator(BaseSolution):
    def __init__(self, plate_model_path, fps, skip_factor, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()
        self.spd = {}
        self.trkd_ids = []
        self.trk_pt = {}
        self.trk_pp = {}
        self.logged_ids = set()

        # Load number plate detector
        self.plate_model = YOLO(plate_model_path)

        # OCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        # Speed limit
        self.SPEED_LIMIT = 30

        # Calibration & FPS info
        self.pixels_per_meter = 10   # <-- adjust after calibration
        self.fps = fps
        self.skip_factor = skip_factor

        # Create folders and CSV
        os.makedirs("overspeed_numberplates", exist_ok=True)
        os.makedirs("violators", exist_ok=True)
        csv_file = os.path.join("violators", "violators.csv")
        if not os.path.exists(csv_file):
            with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Date", "Time", "Track ID", "Class Name", "Speed", "Number Plate"])

    def save_overspeed_numberplate(self, image_array, track_id, speed, numberplate):
        filename = f"ID_{track_id}_SPD_{speed}_PLATE_{numberplate}.jpg"
        filename = "".join(c if c.isalnum() or c in ('_', '.', '-') else '_' for c in filename)
        filepath = os.path.join("overspeed_numberplates", filename)
        cv2.imwrite(filepath, image_array)

    def perform_ocr(self, image_array):
        # Sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image_array, -1, kernel)

        # Upscale if small
        target_height = 120
        h, w = sharpened.shape[:2]
        if h < target_height:
            scale = target_height / h
            sharpened = cv2.resize(sharpened, (int(w*scale), target_height))

        # Show cropped plate for debug
        cv2.imshow("Cropped Plate", sharpened)
        cv2.waitKey(1)

        results = self.ocr.ocr(sharpened, rec=True)
        return ' '.join([result[1][0] for result in results[0]] if results[0] else "")

    def estimate_speed(self, frame):
        self.annotator = Annotator(frame, line_width=self.line_width)
        self.extract_tracks(frame)
        current_time = datetime.now()

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
            label = f"ID: {track_id} {speed_label}"
            self.annotator.box_label(box, label=label, color=colors(track_id, True))

            # Speed calculation
            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):
                if track_id not in self.trkd_ids:
                    self.trkd_ids.append(track_id)
                    # use fixed time_diff
                    time_diff = self.skip_factor / self.fps
                    pixel_diff = np.abs(self.track_line[-1][1].item() - self.trk_pp[track_id][1].item())
                    distance_m = pixel_diff / self.pixels_per_meter
                    if time_diff > 0:
                        speed_m_per_s = distance_m / time_diff
                        speed_kmh = speed_m_per_s * 3.6
                        self.spd[track_id] = round(speed_kmh)

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]
            x1, y1, x2, y2 = map(int, box)

            # Crop vehicle region
            vehicle_crop = frame[y1:y2, x1:x2]

            # Detect number plate inside vehicle
            plate_results = self.plate_model(vehicle_crop)[0]
            ocr_text = ""
            if len(plate_results.boxes) > 0:
                px1, py1, px2, py2 = plate_results.boxes.xyxy[0].cpu().numpy().astype(int)
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                ocr_text = self.perform_ocr(plate_crop)
            else:
                plate_crop = vehicle_crop

            speed = self.spd.get(track_id)
            class_name = self.names[int(cls)]

            print(f"Track ID: {track_id}, Speed: {speed}, OCR: '{ocr_text}'")

            if track_id not in self.logged_ids and ocr_text.strip() and speed is not None and speed > self.SPEED_LIMIT:
                print(f">>> Saving violator: ID {track_id}, Speed {speed} km/h, Plate '{ocr_text}'")
                self.save_overspeed_numberplate(plate_crop, track_id, speed, ocr_text)

                csv_file = os.path.join("violators", "violators.csv")
                with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        current_time.strftime("%Y-%m-%d"),
                        current_time.strftime("%H:%M:%S"),
                        track_id,
                        class_name,
                        speed,
                        ocr_text
                    ])
                self.logged_ids.add(track_id)

        self.display_output(frame)
        return frame


# Open video
cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
skip_factor = 3

print(f"Video FPS: {fps}")

# Define region points
region_points = [(0, 145), (1018, 145)]

# Initialize
speed_obj = SpeedEstimator(
    region=region_points,
    model="YOLOv11.pt",          # vehicle model
    plate_model_path="plate.pt", # number plate model
    fps=fps,
    skip_factor=skip_factor,
    line_width=2
)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % skip_factor != 0:
        continue

    result = speed_obj.estimate_speed(frame)
    cv2.imshow("RGB", result)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
