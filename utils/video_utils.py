import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(frames, output_path):
    height = frames[0].shape[0]
    width = frames[0].shape[1]

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 24,
                          (width, height))  # Correct order!

    for frame in frames:
        out.write(frame)

    out.release()
