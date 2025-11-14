from utils import read_video, save_video


def main():

    video_frames = read_video("Input_Videos/08fd33_4.mp4")
    save_video(video_frames, "Output_Videos/output_video.avi")


if __name__ == '__main__':
    main()
