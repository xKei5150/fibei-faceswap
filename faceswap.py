import numpy as np
import cv2
import insightface
import gfpgan
import torch

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def load_image(image_path):
    return cv2.imread(image_path)

def face_detection(image, app):
    return app.get(image)


def face_swap(source_image, target_face, source_face, swapper):
    return swapper.get(source_image, target_face, source_face, paste_back=True)


def face_upscale(image, upscaler):
    _, _, upscaled_image = upscaler.enhance(image, has_aligned=False, only_center_face=False)
    return upscaled_image


def process_frame(frame, app, swapper, upscaler, source_face):
    faces = face_detection(frame, app)
    processed_frames = []
    for face in faces:
        swapped_face = face_swap(frame, face, source_face, swapper)
        upscaled_frame = face_upscale(swapped_face, upscaler)
        processed_frames.append(upscaled_frame)
    return processed_frames


def process_video(video_path, output_video_path, app, swapper, upscaler, source_face):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print("Started")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            processed_frames = process_frame(frame, app, swapper, upscaler, source_face)
            for processed_frame in processed_frames:
                output_video.write(processed_frame)

            print(f"Processed {cap.get(cv2.CAP_PROP_POS_FRAMES)} out of {total_frames} frames")

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()


def main():
    source_image_path = 'assets/source.jpg'
    video_path = 'assets/target-1080p.mp4'
    output_video_path = 'assets/output_video.mp4'

    source_image = load_image(source_image_path)
    app = insightface.app.FaceAnalysis(name='buffalo_l', det_size=(640, 640))
    app.prepare(ctx_id=0)
    source_faces = face_detection(source_image, app)
    source_face = source_faces[0]
    swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx',
                                              download=False, download_zip=False)
    upscaler = gfpgan.GFPGANer(model_path='models/GFPGANv1.4.pth',
                               upscale=1, arch='clean', bg_upsampler=None)

    process_video(video_path, output_video_path, app, swapper, upscaler, source_face)


if __name__ == "__main__":
    main()