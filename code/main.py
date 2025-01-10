# Import the required libraries
import cv2
import numpy as np
import os

# Input and output directory paths
INPUT_PATH = r"C:\Users\krish\OneDrive\Desktop\image.jpeg"
OUTPUT_PATH = r"C:\Users\krish\OneDrive\Desktop\output.jpg"

# Model paths and configuration
MODEL_PATHS = {
    'face_pbtxt': r"C:\Users\krish\OneDrive\Desktop\model\opencv_face_detector.pbtxt",
    'face_pb': r"C:\Users\krish\OneDrive\Desktop\model\opencv_face_detector_uint8.pb",
    'age_prototxt': r"C:\Users\krish\OneDrive\Desktop\model\age_deploy.prototxt",
    'age_caffemodel': r"C:\Users\krish\OneDrive\Desktop\model\age_net.caffemodel",
    'gender_prototxt': r"C:\Users\krish\OneDrive\Desktop\model\gender_deploy.prototxt",
    'gender_caffemodel': r"C:\Users\krish\OneDrive\Desktop\model\gender_net.caffemodel"
}

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDERS = ['Male', 'Female']
PADDING_COLOR = (0, 0, 0)

# Load networks
def load_networks():
    face_net = cv2.dnn.readNet(MODEL_PATHS['face_pb'], MODEL_PATHS['face_pbtxt'])
    age_net = cv2.dnn.readNet(MODEL_PATHS['age_caffemodel'], MODEL_PATHS['age_prototxt'])
    gender_net = cv2.dnn.readNet(MODEL_PATHS['gender_caffemodel'], MODEL_PATHS['gender_prototxt'])
    return face_net, age_net, gender_net

# Detect faces
def detect_faces(net, frame, conf_threshold=0.7):
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * frame_width))
            y1 = max(0, int(detections[0, 0, i, 4] * frame_height))
            x2 = min(frame_width - 1, int(detections[0, 0, i, 5] * frame_width))
            y2 = min(frame_height - 1, int(detections[0, 0, i, 6] * frame_height))
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes

# Convert image to 1:1 ratio
def convert_to_1x1_with_face(image, face_box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = face_box
    face_center_x = (x1 + x2) // 2
    face_center_y = (y1 + y2) // 2
    max_dim = max(width, height)
    
    new_x1 = max(0, face_center_x - max_dim // 2)
    new_y1 = max(0, face_center_y - max_dim // 2)
    new_x2 = new_x1 + max_dim
    new_y2 = new_y1 + max_dim
    
    padded_image = cv2.copyMakeBorder(
        image,
        top=max(0, -new_y1),
        bottom=max(0, new_y2 - height),
        left=max(0, -new_x1),
        right=max(0, new_x2 - width),
        borderType=cv2.BORDER_CONSTANT,
        value=PADDING_COLOR
    )
    
    return padded_image[new_y1:new_y2, new_x1:new_x2]

# Predict age and gender
def predict_age_gender(face_img, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDERS[np.argmax(gender_preds[0])]
    
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_RANGES[np.argmax(age_preds[0])]
    return gender, age

# Process image
def process_image():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input image not found at {INPUT_PATH}")
    
    face_net, age_net, gender_net = load_networks()
    image = cv2.imread(INPUT_PATH)
    if image is None:
        raise ValueError("Error loading input image")
    
    face_boxes = detect_faces(face_net, image)
    if not face_boxes:
        print("No faces detected.")
        return
    
    for face_box in face_boxes:
        square_image = convert_to_1x1_with_face(image, face_box)
        face = square_image[max(0, face_box[1]):face_box[3], max(0, face_box[0]):face_box[2]]
        
        if face.size == 0:
            continue
        
        gender, age = predict_age_gender(face, age_net, gender_net)
        label = f"{gender}, {age}"
        cv2.putText(image, label, (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(image, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)
    
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"Output saved at {OUTPUT_PATH}")

if __name__ == "__main__":
    process_image()