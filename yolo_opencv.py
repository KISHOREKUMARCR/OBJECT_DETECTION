import cv2
import argparse
import numpy as np
import os

# Default paths
DEFAULT_CONFIG = 'yolov3.cfg'
DEFAULT_WEIGHTS = 'yolov3.weights'
DEFAULT_CLASSES = 'yolov3.txt'

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to input image')
ap.add_argument('-v', '--video', help='path to input video')
args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return label 

output_folder = 'output_crops'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

classes = None
with open(DEFAULT_CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(DEFAULT_WEIGHTS, DEFAULT_CONFIG)

def process_frame(image):
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
        indices = indices.flatten()  # Flatten the list to make it 1D if it's a 2D array

    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        label = draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        crop_img = image[int(y):int(y+h), int(x):int(x+w)]

        print(f"Crop coordinates: x={x}, y={y}, w={w}, h={h}")
        print(f"Crop image size: {crop_img.shape}")

        if crop_img.size == 0:
            print(f"Warning: Empty crop detected for label: {label}")
            continue

        filename = f"{label}_{i}_output.jpg"  # Create a filename based on label and index
        cv2.imwrite(os.path.join(output_folder, filename), crop_img)

    return image

if args.image:
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image from {args.image}")
        exit()
    processed_image = process_frame(image)
    cv2.imshow("object detection", processed_image)
    cv2.waitKey()
    cv2.imwrite("object-detection.jpg", processed_image)
    cv2.destroyAllWindows()

if args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame)
        cv2.imshow("object detection", processed_frame)

        filename = f"frame_{frame_count}_output.jpg"
        cv2.imwrite(os.path.join(output_folder, filename), processed_frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
