from ultralytics import YOLO
import supervision as sv


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

    def get_object_tracks(self, frames):

        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "refrees": [],
            "ball": []
        }

        for detection in detections:

            supervision_detections = sv.Detections.from_ultralytics(detection)

            tracked = self.tracker.update_with_detections(

            supervision_detections),

           tracks["players"].append({})
           tracks["refrees"].append({})
           tracks["ball"].append({})

        return tracks
