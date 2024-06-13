from pathlib import Path
import sys
import supervision as sv
import os
import streamlit as st
# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
#ROOT = ROOT.relative_to(Path.cwd())
# Sources
IMAGE = 'Image'
VIDEO = 'Video'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'
ENCROACHMENT = 'Encroachment'
JUNCTION = 'Junction Evaluation Dataset'
JUNCTIONEVAL = 'Junction Evaluation'
BENCHMARKING = "Benchmarking"
SOURCES_LIST = [IMAGE, VIDEO, RTSP, YOUTUBE, ENCROACHMENT, JUNCTION, JUNCTIONEVAL,BENCHMARKING]
os.environ['COURIER_API_KEY'] = st.secrets['COURIER_API_KEY']

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'default.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'default_detected.png'

VIDEOS_DICT = {}
EVALUATION_DICT = {}
FINAL_DICT = {}
CLASSES = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}
 
# iterate over files in
# that directory

    # ML Model config
MODEL_DIR = ROOT /'weights'
#print(MODEL_DIR)
DETECTION_MODEL = MODEL_DIR/'yolov8n.pt'
print(DETECTION_MODEL)
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'
print(DETECTION_MODEL)

# Webcam
WEBCAM_PATH = 0

def updateDirectories():
        # Videos config
    global VIDEOS_DICT,EVALUATION_DICT,FINAL_DICT
    VIDEOS_DICT = {}
    EVALUATION_DICT = {}
    FINAL_DICT = {}
    VIDEO_DIR = ROOT / 'videos'

    VIDEOS_DICT = {'video_5': VIDEO_DIR/'Footage_Feed_5.MP4' }
    #     'video_1': VIDEO_DIR /'video_1.mp4',
    #     'video_2': VIDEO_DIR /'video_2.mp4',
    #     'video_3': VIDEO_DIR /'video_3.mp4',
    #     'video_4': VIDEO_DIR/'video.mp4',
    # }
    # for i in VIDEOS_DICT.keys():
    #     print(i,VIDEOS_DICT[i])
    for filename in os.listdir(VIDEO_DIR):
        f = os.path.join(VIDEO_DIR, filename)
        # checking if it is a file
        try:
            if(filename[-4:]==".mp4" or filename[-4:] == ".AVI"):
                VIDEOS_DICT[filename] = f
            else:
                pass
        except:
            pass
    for i in VIDEOS_DICT.keys():
        print(i,VIDEOS_DICT[i])


    EVALUATION_DIR = VIDEO_DIR / 'junctionEvalDataset'
    for filename in os.listdir(EVALUATION_DIR):
        f = os.path.join(EVALUATION_DIR, filename)
        # checking if it is a file
        EVALUATION_DICT[filename] = f

    for i in EVALUATION_DICT.keys():
        print(i,EVALUATION_DICT[i])


    for i in EVALUATION_DICT.keys():
        newDict = {}
        for filename in os.listdir(EVALUATION_DICT[i]):
            f = os.path.join(i, filename)
            # checking if it is a file
            try:
                if(filename[-4:].lower()==".mp4"):
                    newDict[filename] = f
                else:
                    pass
            except:
                pass  
        FINAL_DICT[i] = newDict


    # EVALUATION_DICT = {
    #     'chickPea': EVALUATION_DIR /'chickPea.mp4',
    # }

    # print (EVALUATION_DICT['chickPea'])

    # ML Model config
    MODEL_DIR = ROOT /'weights'
    #print(MODEL_DIR)
    DETECTION_MODEL = MODEL_DIR/'yolov8n.pt'
    print(DETECTION_MODEL)
    # In case of your custome model comment out the line above and
    # Place your custom model pt file name at the line below 
    # DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

    SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'
    print(DETECTION_MODEL)

    # Webcam
    DATASET_DIR = ROOT/"Dataset"



        
