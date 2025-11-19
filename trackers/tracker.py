from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pickle
import os
import sys
import cv2

# https://claude.ai/share/8aff1f3b-5b25-466f-8696-faabe397fa8d(CLaude Explanation of this code)


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, conf=0.1, batch_size=20):

        detections = []

        for i in range(0, len(frames), batch_size):

            batch = frames[i:i + batch_size]

            detections.extend(self.model.predict(batch, conf=conf))

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Allow loading from pickle stub for efficiency if provided and desired
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            supervision_detections = sv.Detections.from_ultralytics(detection)

            # Convert 'goalkeeper' to 'player' if present
            for object_ind, class_id in enumerate(supervision_detections.class_id):
                if cls_names[class_id] == "goalkeeper":
                    supervision_detections.class_id[object_ind] = cls_names_inv["player"]

            tracked = self.tracker.update_with_detections(
                supervision_detections)

            # Prepare empty dicts for this frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in tracked:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if "player" in cls_names_inv and cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif "referee" in cls_names_inv and cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Ball detection is done separately -- use original supervision_detections
            for frame_detection in supervision_detections:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if "ball" in cls_names_inv and cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # If a stub_path was provided, save the tracking data to this file using pickle
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks
