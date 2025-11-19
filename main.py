from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker


def main():
    video_frames = read_video("Input_Videos/08fd33_4.mp4")

    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stub/track_stubs.pkl'
    )

    save_video(video_frames, "Output_Videos/output_video.avi")


if __name__ == '__main__':
    main()
