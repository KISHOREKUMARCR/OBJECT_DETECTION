from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import os
import numpy as np
from fastapi import Request
import uuid  

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

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, output_folders):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    crop_img = img[int(y):int(y_plus_h), int(x):int(x_plus_w)]
    if crop_img.size == 0:
        print(f"Warning: Empty crop detected for label: {label}")
        return label 

    crop_image_folder = os.path.join(output_folders['base_folder'], 'crop_images')
    if not os.path.exists(crop_image_folder):
        os.makedirs(crop_image_folder)
    
    crop_filename = f"{label}_{uuid.uuid4().hex}.jpg"
    crop_image_path = os.path.join(crop_image_folder, crop_filename)
    cv2.imwrite(crop_image_path, crop_img)

    return label 

def process_frame(image, output_folders):
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
        label = draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), output_folders)

    detected_output_folder = os.path.join(output_folders['base_folder'], 'detected_output')
    if not os.path.exists(detected_output_folder):
        os.makedirs(detected_output_folder)
    
    detected_image_path = os.path.join(detected_output_folder, 'detected_image.jpg')
    cv2.imwrite(detected_image_path, image)

    return detected_image_path
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect", response_class=HTMLResponse)
async def detect(request: Request, file: UploadFile = File(...)):
    unique_id = uuid.uuid4().hex
    upload_folder = os.path.join(output_base_folder, unique_id)
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    image = cv2.imread(file_path)
    if image is None:
        return templates.TemplateResponse("result.html", {"request": request, "error_message": "Could not read image."})
    
    output_folders = { 'base_folder': upload_folder }
    
    detected_image_path = process_frame(image, output_folders)    
    return templates.TemplateResponse("result.html", {"request": request, "processed_image_path": detected_image_path})


@app.post("/upload_video", response_class=HTMLResponse)
async def upload_video(request: Request, file: UploadFile = File(...)):
    unique_id = uuid.uuid4().hex
    upload_folder = os.path.join(output_base_folder, unique_id)
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    video_path = os.path.join(upload_folder, file.filename)
    
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return templates.TemplateResponse("result.html", {"request": request, "error_message": "Could not open video file."})
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_output_folder = os.path.join(upload_folder, f'frame_{frame_count}')
        if not os.path.exists(frame_output_folder):
            os.makedirs(frame_output_folder)
        
        detected_image_path = process_frame(frame, { 'base_folder': frame_output_folder })
        
    cap.release()
    
    return templates.TemplateResponse("result.html", {"request": request, "processed_image_path": detected_image_path})


@app.get("/image_upload", response_class=HTMLResponse)
async def image_upload(request: Request):
    return templates.TemplateResponse("image_upload.html", {"request": request})

@app.get("/video_upload", response_class=HTMLResponse)
async def video_upload(request: Request):
    return templates.TemplateResponse("video_upload.html", {"request": request})

