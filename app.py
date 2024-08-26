from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import os
import numpy as np
import random
from typing import List
import base64
from io import BytesIO

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DEFAULT_CONFIG = 'yolov3.cfg'
DEFAULT_WEIGHTS = 'yolov3.weights'
DEFAULT_CLASSES = 'yolov3.txt'
output_base_folder = 'DETECTION_OUTPUT'

if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

classes = None
with open(DEFAULT_CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(DEFAULT_WEIGHTS, DEFAULT_CONFIG)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def generate_unique_number(existing_numbers):
    while True:
        number = random.randint(100, 999)
        if number not in existing_numbers:
            existing_numbers.add(number)
            return number

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, output_folders):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    crop_img = img[int(y):int(y_plus_h), int(x):int(x_plus_w)]
    if crop_img.size == 0:
        print(f"Warning: Empty crop detected for label: {label}")
        return label, None

    crop_image_folder = os.path.join(output_folders['base_folder'], 'crop_images')
    if not os.path.exists(crop_image_folder):
        os.makedirs(crop_image_folder)

    existing_numbers = set()  
    unique_id = generate_unique_number(existing_numbers)

    crop_filename = f"{label}_{unique_id}.jpg"
    crop_image_path = os.path.join(crop_image_folder, crop_filename)
    cv2.imwrite(crop_image_path, crop_img)

    return label, crop_filename


def process_frame(image, output_folders, unique_id):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    crop_images_info = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        indices = indices.flatten()

    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        label, crop_filename = draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), output_folders)
        crop_image_info = {
            'class_name': str(classes[class_ids[i]]),
            'crop_image_path': f"crop_images/{crop_filename}",
            'uploaded_id': unique_id,
            'serial_number': i
        }
        
        crop_images_info.append(crop_image_info)

    detected_output_folder = os.path.join(output_folders['base_folder'], 'detected_output')
    if not os.path.exists(detected_output_folder):
        os.makedirs(detected_output_folder)
    
    detected_image_path = os.path.join(detected_output_folder, 'detected_image.jpg')
    cv2.imwrite(detected_image_path, image)

    return detected_image_path, crop_images_info



@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect", response_class=HTMLResponse)
async def detect(request: Request, file: UploadFile = File(...)):
    existing_numbers = set()  
    unique_id = generate_unique_number(existing_numbers)
    upload_folder = os.path.join("static", "uploads", str(unique_id))
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    image = cv2.imread(file_path)
    if image is None:
        return templates.TemplateResponse("result.html", {"request": request, "error_message": "Could not read image."})
    
    output_folders = {'base_folder': upload_folder}
    
    detected_image_path, crop_images_info = process_frame(image, output_folders, str(unique_id))
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "processed_image_path": detected_image_path,
        "crop_images_info": crop_images_info
    })

@app.post("/upload_video", response_class=HTMLResponse)
async def upload_video(request: Request, file: UploadFile = File(...)):
    existing_numbers = set()
    unique_id = generate_unique_number(existing_numbers)
    upload_folder = os.path.join("static", "uploads", str(unique_id))

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    video_path = os.path.join(upload_folder, file.filename)
    
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return templates.TemplateResponse("result_video.html", {"request": request, "error_message": "Could not open video file."})
    
    frame_count = 0
    frames_info = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_output_folder = os.path.join(upload_folder, f'frame_{frame_count}')
        if not os.path.exists(frame_output_folder):
            os.makedirs(frame_output_folder)
        
        detected_image_path, crop_images_info = process_frame(frame, {'base_folder': frame_output_folder}, str(unique_id))
        frames_info.append({
            'frame_image': detected_image_path,
            'crop_images': [
                {
                    'path': f'uploads/{unique_id}/frame_{frame_count}/{crop["crop_image_path"]}',
                    'class_name': crop["class_name"]
                }
                for crop in crop_images_info
            ]
        })


        
    cap.release()
    
    return templates.TemplateResponse("result_video.html", {
        "request": request,
        "frames_info": frames_info
    })



@app.get("/image_upload", response_class=HTMLResponse)
async def image_upload(request: Request):
    return templates.TemplateResponse("image_upload.html", {"request": request})

@app.get("/video_upload", response_class=HTMLResponse)
async def video_upload(request: Request):
    return templates.TemplateResponse("video_upload.html", {"request": request})
