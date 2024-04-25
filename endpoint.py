from fastapi import FastAPI, File, UploadFile
import uvicorn
import cv2
import numpy as np
import os
import tempfile
from faceswap import process_frame, face_detection, process_video, load_models

app = FastAPI()

OUTPUT_DIR = "output_files"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


@app.post("/process-video")
async def process_video_endpoint(video: UploadFile = File(...), source_image: UploadFile = File(...)):
    source_image_bytes = await source_image.read()
    nparr = np.frombuffer(source_image_bytes, np.uint8)
    source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    app, swapper, upscaler = load_models()

    source_faces = face_detection(source_image, app)
    source_face = source_faces[0]

    with tempfile.NamedTemporaryFile(delete=False) as tmp_video:
        tmp_video.write(await video.read())
        tmp_video_path = tmp_video.name

    output_video_path = process_video(tmp_video_path, app, swapper, upscaler, source_face)

    output_video_filename = os.path.basename(output_video_path)
    new_output_video_path = os.path.join(OUTPUT_DIR, output_video_filename)
    os.rename(output_video_path, new_output_video_path)

    return {"output_video_path": new_output_video_path}


@app.post("/process-image")
async def process_image_endpoint(image: UploadFile = File(...), source_image: UploadFile = File(...)):
    # Read the image file
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    source_image_bytes = await source_image.read()
    nparr = np.frombuffer(source_image_bytes, np.uint8)
    source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    app, swapper, upscaler = load_models()

    source_faces = face_detection(source_image, app)
    source_face = source_faces[0]

    processed_frames = process_frame(frame, app, swapper, upscaler, source_face)

    output_files = []
    for i, processed_frame in enumerate(processed_frames):
        output_file = f"processed_image_{i + 1}.jpg"
        output_file_path = os.path.join(OUTPUT_DIR, output_file)
        cv2.imwrite(output_file_path, processed_frame)
        output_files.append(output_file_path)

    return {"output_files": output_files}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
