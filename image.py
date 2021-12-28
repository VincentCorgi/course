
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
from enum import Enum
import math
class BodyPart(Enum):
    NOSE = 0,
    LEFT_EYE = 1,
    RIGHT_EYE = 2,
    LEFT_EAR = 3,
    RIGHT_EAR = 4,
    LEFT_SHOULDER = 5,
    RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7,
    RIGHT_ELBOW = 8,
    LEFT_WRIST = 9,
    RIGHT_WRIST = 10,
    LEFT_HIP = 11,
    RIGHT_HIP = 12,
    LEFT_KNEE = 13,
    RIGHT_KNEE = 14,
    LEFT_ANKLE = 15,
    RIGHT_ANKLE = 16,


class Position:
    def __init__(self):
        self.x = 0
        self.y = 0


class KeyPoint:
    def __init__(self):
        self.bodyPart = BodyPart.NOSE
        self.position = Position()
        self.score = 0.0


class Person:
    def __init__(self):
        self.keyPoints = []
        self.score = 0.0


GRAPH_NAME = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
min_conf_threshold = float(0.4)

IM_NAME = '8.jpg'

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()


if IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)
GRAPH_NAME = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,GRAPH_NAME)

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
#print(height,width)
floating_model = (input_details[0]['dtype'] == np.float32)
def sigmoid(x):
    return 1. / (1. + math.exp(-x))
input_mean = 127.5
input_std = 127.5
yee=0
yee1=0
# Loop over every image and perform detection
for image_path in images:

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    #print(imH, imW)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    #print(input_data.shape)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if input_details[0]['dtype'] == type(np.float32(1.0)):
            input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    heat_maps = interpreter.get_tensor(output_details[0]['index'])
    offset_maps = interpreter.get_tensor(output_details[1]['index'])
    fff= interpreter.get_tensor(output_details[2]['index'])
    #print(fff.shape)
    h_pose=len(heat_maps[0])
    w_pose=len(heat_maps[0][0])
    num_key_points = len(heat_maps[0][0][0])
    # Loop over all detections and draw detection box if confidence is above minimum threshold      
    key_point_positions = [[0] * 2 for i in range(num_key_points)]
    #print('width =',key_point_positions)
    for key_point in range(num_key_points):
        max_val = heat_maps[0][0][0][key_point]
        max_row = 0
        max_col = 0
        for row in range(h_pose):
            for col in range(w_pose):
                heat_maps[0][row][col][key_point] = sigmoid(heat_maps[0][row][col][key_point])
                if heat_maps[0][row][col][key_point] > max_val:
                    max_val = heat_maps[0][row][col][key_point]
                    max_row = row
                    max_col = col
        key_point_positions[key_point] = [max_row, max_col]
        #print(key_point_positions[key_point])
    x_coords = [0] * num_key_points
    y_coords = [0] * num_key_points
    confidenceScores = [0] * num_key_points
    for i, position in enumerate(key_point_positions):
            position_y = int(key_point_positions[i][0])
            position_x = int(key_point_positions[i][1])
            
            y_coords[i] = (position[0] / float(h_pose - 1) * imH +
                           offset_maps[0][position_y][position_x][i])
            x_coords[i] = (position[1] / float(w_pose - 1) * imW +
                           offset_maps[0][position_y][position_x][i + num_key_points])
            #print('(x,y):',x_coords[i],y_coords[i])
            confidenceScores[i] = heat_maps[0][position_y][position_x][i]
            #print("confidenceScores[", i, "] = ", confidenceScores[i])
            x=int(x_coords[i])
            y=int(y_coords[i])
            #print('x=',x,'y=',y,'confidence=%.2f' %confidenceScores[i])
            if(i<5):              
                if(confidenceScores[i]>yee):
                    yee=confidenceScores[i]
                    z=int(y_coords[i])
            elif(i>10):              
                if(confidenceScores[i]>yee1):
                    yee1=confidenceScores[i]
                    zz=int(y_coords[i])
            if(confidenceScores[i]>0.1): 
                cv2.circle(image, (x,y), 3, (0, 255, 0), cv2.FILLED)
    print(z)
    print(zz)
    if(zz-z<50):
        print('fall')
    else:
        print('ok')
    cv2.imshow('Object detector', image)

    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(0) == ord('q'):
        break

# Clean up
cv2.imwrite("result.png",image)
cv2.destroyAllWindows()
