<h1 align="center">GenAgeNet</h1>
This code detects faces in an image using a pre-trained DNN model and predicts their gender and age range using Caffe models. Detected faces are annotated with bounding boxes and labels and the output is saved as an image with the predictions.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install opencv-python opencv-contrib-python numpy
   ```

2. Enter the path of the input image whose age and gender detection you want to do
  
3. Enter the path of output image where you want to see the resultant image

4. Download the following models and paste their path in the code:

   a. [opencv_face_detector.pbtxt](https://github.com/kr1shnasomani/GenAgeNet/blob/main/model/opencv_face_detector.pbtxt)

   b. [opencv_face_detector_uint8.pb](https://github.com/kr1shnasomani/GenAgeNet/blob/main/model/opencv_face_detector_uint8.pb)

   c. [age_deploy.prototxt](https://github.com/kr1shnasomani/GenAgeNet/blob/main/model/age_deploy.prototxt)

   d. [age_net.caffemodel](https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/age_net.caffemodel)

   e. [gender_deploy.prototxt](https://github.com/kr1shnasomani/GenAgeNet/blob/main/model/gender_deploy.prototxt)

   f. [gender_net.caffemodel](https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/gender_net.caffemodel)

5. Upon running the code after doing all this the code will provide its prediction

## Model Prediction:

  Input Image:

  ![image](https://github.com/user-attachments/assets/fd71a74a-6d40-4cd1-bf5c-87235ee1cca6)

  Output Image:

  ![image](https://github.com/user-attachments/assets/e5241ec1-717f-41fc-a89d-ab733b016d89)

  Actual age of the person in the image - 36 years

## Overview:
The code implements a **Gender and Age Detection System** using pre-trained models in OpenCV's deep learning module (cv2.dnn). Below is an overview of the code:

### **Key Features**
1. **Face Detection**:
   - Uses a pre-trained DNN model to detect faces in an input image.
   - Outputs bounding boxes for detected faces with a confidence threshold.

2. **Age and Gender Prediction**: Employs two pre-trained Caffe models to predict:
     - **Gender**: Male or Female.
     - **Age Range**: One of the predefined age ranges.

3. **Image Preprocessing**:
   - Converts detected face regions into a square (1:1 aspect ratio) image.
   - Uses padding if necessary to maintain the square ratio.

4. **Result Overlay**:
   - Draws bounding boxes around detected faces.
   - Annotates the image with predicted gender and age range above each face.

5. **Output**: Saves the processed image with annotations to a specified output path.

### **Code Walkthrough**
1. **Libraries and Configuration**:
   - OpenCV is used for image processing and DNN inference.
   - Paths for the input image, output image, and pre-trained models are defined.

2. **Model Loading** (`load_networks`): Loads face detection, age prediction, and gender prediction networks from their respective files.

3. **Face Detection** (`detect_faces`): Processes the input image through the face detection model and outputs bounding boxes for all detected faces.

4. **Image Preprocessing** (`convert_to_1x1_with_face`): Converts the detected face region into a square image with optional padding to ensure uniform dimensions.

5. **Age and Gender Prediction** (`predict_age_gender`): 
     - Creates a blob from the detected face.
     - Passes the blob to gender and age prediction models.
     - Returns the predicted gender and age range.

6. **Main Processing Pipeline**:
   - Reads the input image.
   - Detects faces using the face detection model.
   - For each detected face:
     - Extracts the face region.
     - Predicts gender and age.
     - Draws the bounding box and overlays predictions on the image.
   - Saves the annotated image to the output path.


### **Pre-Trained Models**
1. **Face Detection**: `opencv_face_detector.pb` and `opencv_face_detector.pbtxt`

2. **Age Prediction**: `age_net.caffemodel` and `age_deploy.prototxt`

3. **Gender Prediction**: `gender_net.caffemodel` and `gender_deploy.prototxt`
