from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import supervision as sv
import ffmpeg
from tqdm import tqdm
import settings
import argparse
import json
import os
from typing import Any, Optional, Tuple, Dict, Iterable, List, Set
import shutil
import sys
import cv2
import numpy as np
from inference import get_roboflow_model
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from utils.timers import ClockBasedTimer
from scripts.jxnEvalDataCreation import mainFunc
from scripts.jxnEvaluator import jxnEvalFunc


import supervision as sv

KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")

THICKNESS = 2
COLORS = sv.ColorPalette.DEFAULT
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]
violations = []
displayed={}
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)
current_mouse_position: Optional[Tuple[int, int]] = None

class CustomSink:
    def __init__(self, zone_configuration_path: str, classes, violation_time: int):
        self.classes = classes
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones_config(file_path=zone_configuration_path)
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in self.polygons
        ]
        self.violation_time = violation_time

    def on_prediction(self, result: dict, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = sv.Detections.from_inference(result)
        detections = detections[find_in_list(detections.class_id, self.classes)]
        detections = self.tracker.update_with_detections(detections)

        annotated_frame = frame.image.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        for idx, zone in enumerate(self.zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = self.timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            for tracker_ID, time, cl in zip(detections_in_zone.tracker_id, time_in_zone, detections_in_zone.class_id):
                if tracker_ID not in displayed:
                    if(time%60 >= int(self.violation_time)):
                        violations.append(tracker_ID)
                        str = tracker_ID + " " + cl + " Location: CrossingX "
                        st.warning(str, icon= "⚠️")
                        displayed[tracker_ID] = 1 
                
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )
        cv2.imshow("Processed Video", annotated_frame)
        cv2.waitKey(1)
        
class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]
    
    
class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        zoneIN_configuration_path: str,
        zoneOUT_configuration_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.zoneIN_configuration_path = zoneIN_configuration_path
        self.zoneOUT_configuration_path = zoneOUT_configuration_path
        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        ZONE_IN_POLYGONS = load_zones_config(file_path=zoneIN_configuration_path)
        ZONE_OUT_POLYGONS = load_zones_config(file_path=zoneOUT_configuration_path)

        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])

        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )
        vid_cap = cv2.VideoCapture(self.source_video_path)
        st_frame = st.empty()
        while(vid_cap.isOpened()):
            success = vid_cap.read()
            if(success):
                if self.target_video_path:
                    with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                        for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                            annotated_frame = self.process_frame(frame)
                            sink.write_frame(annotated_frame)
                else:
                    for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                        annotated_frame = self.process_frame(frame)
                        st_frame.image(annotated_frame,
                                   caption='Detected Video',
                                   channels="BGR",
                                   use_column_width=True)
                vid_cap.release()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.bounding_box_annotator.annotate(
            annotated_frame, detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)
    

class JunctionEvaluation:
    
    def __init__(self,sourcePath):
        self.sourcePath = sourcePath
        pass
    
    def datasetCreation(self,cycle):
        savePath = "videos/junctionEvalDataset/"
        print("ABC\n\n\n\n\n\n\n\n"+self.sourcePath)
        videoName = self.sourcePath[self.sourcePath.rfind("\\")+1:]
        videoName = videoName[:-4]
        finalpath = savePath+videoName+"Clips"
        isExist = os.path.exists(finalpath)
        if (isExist):
            shutil.rmtree(finalpath)
        os.makedirs(finalpath)
        mainFunc(self.sourcePath,cycle,finalpath)
        settings.updateDirectories()
        return finalpath
            
        
def startup():
    settings.updateDirectories()

         
def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):

    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def enchroachment():
    source_vid = st.sidebar.selectbox(
    "Choose a video...", settings.VIDEOS_DICT.keys())
    source_path = str(settings.VIDEOS_DICT.get(source_vid))
    print(source_path)
    time = st.sidebar.text_input("Violation Time:")
    source_url = st.sidebar.text_input("Source Url:")
    cwd = os.getcwd()
    if st.sidebar.button("Generate Bottleneck Alerts"):
        if(source_url): 
            zones_configuration_path = "configure/ZONES"+source_url+".json" 
            zones_configuration_path = os.path.join(cwd,zones_configuration_path)
            livedetection(source_url=source_url, violation_time=int(time), zone_configuration_path=zones_configuration_path)
        else:
            new_path = source_path.split("\\")[-1]
            zones_configuration_path = "configure/ZONES"+new_path+".json" 
            zones_configuration_path = os.path.join(cwd,zones_configuration_path)
            if(os.path.exists(zones_configuration_path)):
                timedetect(source_path = source_path, zone_configuration_path = zones_configuration_path, violation_time=time)
            else:
                drawzones(source_path = source_path, zone_configuration_path = zones_configuration_path)
                timedetect(source_path = source_path, zone_configuration_path = zones_configuration_path, violation_time=time)
def junctionEvaluationDataset():
    source_vid = st.sidebar.selectbox(
    "Choose a video...", settings.VIDEOS_DICT.keys())
    source_path = str(settings.VIDEOS_DICT.get(source_vid))

    successVar = False
    cycle = []
    try:
        cycle = st.sidebar.text_input("Cycle")
        cycle = cycle.split()
        cycle = [int (i) for i in cycle]
        successVar = True
    except:
        pass
    # time = st.sidebar.text_input("Violation Time:")
    #source_url = st.sidebar.text_input("Source Url:")
    
    if st.sidebar.button("Create Dataset"):
        if (successVar == False):
            st.sidebar.error("Invalid cycle syntax")
            pass
        else:
            jxnEvalInstance = JunctionEvaluation(source_path)
            returnPath = jxnEvalInstance.datasetCreation(cycle=cycle)
            st.sidebar.write("Dataset Created Successfully at "+returnPath)
            
def BenchMarking():


    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT[source_vid], 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    # try:
    #     threshold = int(threshold)
    #     if (threshold > 5 or threshold < 1):
    #         st.sidebar.error("Enter a valid value")
    #     else:
    #         if st.sidebar.button("Start Evaluation"):
    #             returnVid = jxnEvalFunc(threshold)
    #             is_display_tracker, tracker = display_tracker_options()

    #             with open(returnVid, 'rb') as video_file:
    #                 video_bytes = video_file.read()
                    
    #             if video_bytes:
    #                 st.video(video_bytes)
                                                        
    # except:
    #     st.sidebar.error("Enter a valid integer")            
        
            
            
        
def junctionEvaluation():
    if (len(settings.EVALUATION_DICT.keys()) == 0):
        st.sidebar.error("Create a dataset first")
    else:
        source_dir = st.sidebar.selectbox(
        "Choose a folder", settings.EVALUATION_DICT.keys())
        
        source_path = str(settings.EVALUATION_DICT.get(source_dir))
        source_vid = st.sidebar.selectbox(
        "Choose a clip", settings.FINAL_DICT[source_dir].keys())
        
        
        with open("videos/JunctionEvalDataset/"+source_dir+"/"+source_vid, 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)

        threshold = st.sidebar.text_input(
            "Enter a integer in range 1-5"
        )

        try:
            
            threshold = int(threshold)
            if (threshold > 5 or threshold < 1):
                st.sidebar.error("Enter a valid value")
            else:
                if st.sidebar.button("Start Evaluation"):
                    returnVid = "videos/JunctionEvaluations/IndiraNagarClips/clip1.mp4"
                    with open(returnVid, 'rb') as video_file2:
                        video_bytes2 = video_file2.read()
                        
                    if video_bytes2:
                        st.video(video_bytes2)
                    
                                                            
        except:
            st.sidebar.error("Enter a valid integer")            
            
            
        

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT[source_vid], 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def drawzones(source_path, zone_configuration_path):
    
    def resolve_source(source_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(source_path):
            return None

        image = cv2.imread(source_path)
        if image is not None:
            return image

        frame_generator = sv.get_video_frames_generator(source_path=source_path)
        frame = next(frame_generator)
        return frame
    
    def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
        global current_mouse_position
        if event == cv2.EVENT_MOUSEMOVE:
            current_mouse_position = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            POLYGONS[-1].append((x, y))
    
    def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
        global POLYGONS, current_mouse_position
        image[:] = original_image.copy()
        for idx, polygon in enumerate(POLYGONS):
            color = (
                COLORS.by_idx(idx).as_bgr()
                if idx < len(POLYGONS) - 1
                else sv.Color.WHITE.as_bgr()
            )

            if len(polygon) > 1:
                for i in range(1, len(polygon)):
                    cv2.line(
                        img=image,
                        pt1=polygon[i - 1],
                        pt2=polygon[i],
                        color=color,
                        thickness=THICKNESS,
                    )
                if idx < len(POLYGONS) - 1:
                    cv2.line(
                        img=image,
                        pt1=polygon[-1],
                        pt2=polygon[0],
                        color=color,
                        thickness=THICKNESS,
                    )
            if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=current_mouse_position,
                    color=color,
                    thickness=THICKNESS,
                )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, image)

    def redraw_polygons(image: np.ndarray) -> None:
        for idx, polygon in enumerate(POLYGONS[:-1]):
            if len(polygon) > 1:
                color = COLORS.by_idx(idx).as_bgr()
                for i in range(len(polygon) - 1):
                    cv2.line(
                        img=image,
                        pt1=polygon[i],
                        pt2=polygon[i + 1],
                        color=color,
                        thickness=THICKNESS,
                    )
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=polygon[0],
                    color=color,
                    thickness=THICKNESS,
                )

    def close_and_finalize_polygon(image: np.ndarray, original_image: np.ndarray) -> None:
        if len(POLYGONS[-1]) > 2:
            cv2.line(
                img=image,
                pt1=POLYGONS[-1][-1],
                pt2=POLYGONS[-1][0],
                color=COLORS.by_idx(0).as_bgr(),
                thickness=THICKNESS,
            )
        POLYGONS.append([])
        image[:] = original_image.copy()
        redraw_polygons(image)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, image)

    def save_polygons_to_json(polygons, target_path):
        data_to_save = polygons if polygons[-1] else polygons[:-1]
        with open(target_path, "w") as f:
            json.dump(data_to_save, f)
    
    global current_mouse_position
    original_image = resolve_source(source_path=source_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    image = original_image.copy()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ENTER or key == KEY_NEWLINE:
            close_and_finalize_polygon(image, original_image)
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            current_mouse_position = None
        elif key == KEY_SAVE:
            save_polygons_to_json(POLYGONS, zone_configuration_path)
            print(f"Polygons saved to {zone_configuration_path}")
            break
        redraw(image, original_image)
        if key == KEY_QUIT:
            break

    cv2.destroyAllWindows()

def timedetect(source_path, zone_configuration_path, violation_time):
    COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
    COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
    LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
    )
    model_id = "yolov8m-640"
    classes = [2,5,6,7]
    confidence = 0.3
    iou = 0.7
    model = get_roboflow_model(model_id=model_id)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=source_path)
    frames_generator = sv.get_video_frames_generator(source_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    vid_cap = cv2.VideoCapture(source_path)
    st_frame = st.empty()
    while(vid_cap.isOpened()):
            success = vid_cap.read()
            st.subheader("ALERTS: ")
            if success:
                    for frame in frames_generator:
                        results = model.infer(frame, confidence=confidence, iou_threshold=iou)[0]
                        detections = sv.Detections.from_inference(results)
                        detections = detections[find_in_list(detections.class_id, classes)]
                        detections = tracker.update_with_detections(detections)

                        annotated_frame = frame.copy()

                        for idx, zone in enumerate(zones):
                            annotated_frame = sv.draw_polygon(
                                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
                            )

                            detections_in_zone = detections[zone.trigger(detections)]
                            time_in_zone = timers[idx].tick(detections_in_zone)
                            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                            annotated_frame = COLOR_ANNOTATOR.annotate(
                                scene=annotated_frame,
                                detections=detections_in_zone,
                                custom_color_lookup=custom_color_lookup,
                            )
                            labels = [
                                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)     
                            ]

                            annotated_frame = LABEL_ANNOTATOR.annotate(
                                scene=annotated_frame,
                                detections=detections_in_zone,
                                labels=labels,
                                custom_color_lookup=custom_color_lookup,
                            )
                            
                            for tracker_ID, time, cl in zip(detections_in_zone.tracker_id, time_in_zone, detections_in_zone.class_id):
                                if tracker_ID not in displayed:
                                    if(time%60 >= int(violation_time)):
                                        violations.append(tracker_ID)
                                        cla = settings.CLASSES[cl]
                                        s = "Tracker_ID:" + str(tracker_ID) + " Class: " + cla + " Location: CrossingX "
                                        st.warning(s, icon= "⚠️")
                                        displayed[tracker_ID] = 1
                        
                        st_frame.image(annotated_frame,
                                   caption='Detected Video',
                                   channels="BGR",
                                   use_column_width=True)
                    vid_cap.release()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
                    
            
    
def livedetection(source_url: str, violation_time: int, zone_configuration_path: str):
    model_id = 'yolov8x-640'
    classes = [2,5,6,7]
    confidence = 0.3
    iou = 0.7
    model = YOLO('weights\yolov8n.pt')
    sink = CustomSink(zone_configuration_path=zone_configuration_path, classes=classes, violation_time = violation_time)

    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=source_url,
        on_prediction=sink.on_prediction,
        confidence=confidence,
        iou_threshold=iou,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()
        
        
    
def liveevaluation(source_url: str, zone_configuration_path: str):
    model_id = 'yolov8x-640'
    classes = [2,5,6,7]
    confidence = 0.3
    iou = 0.7
    model = YOLO('weights\yolov8n.pt')
    sink = CustomSink(zone_configuration_path=zone_configuration_path, classes=classes)

    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=source_url,
        on_prediction=sink.on_prediction,
        confidence=confidence,
        iou_threshold=iou,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()



def benchMarking():
    source_vid = st.sidebar.selectbox(
    "Choose a video...", settings.VIDEOS_DICT.keys())
    source_path = str(settings.VIDEOS_DICT.get(source_vid))
    new_path = source_path.split("\\")[-1]
    zones_IN_configuration_path = "configure/ZONES_IN"+new_path+".json"
    zones_OUT_configuration_path = "configure/ZONES_OUT"+new_path+".json"
    cwd = os.getcwd()
    weight_path = "weights/yolov8n.pt"
    weight_path = os.path.join(cwd,weight_path)
    if(st.sidebar.button("Draw Zones IN")):
        drawzones(source_path = source_path, zone_configuration_path = zones_IN_configuration_path)
        st.sidebar.write("ZONES_IN created successfully at "+zones_IN_configuration_path)
    
    if(st.sidebar.button("Draw Zones OUT")):    
        drawzones(source_path = source_path, zone_configuration_path = zones_OUT_configuration_path)
        st.sidebar.write("ZONES_OUT created successfully at "+zones_OUT_configuration_path)

    if(st.sidebar.button("BenchMark")):
        processor = VideoProcessor(
        source_weights_path=weight_path,
        source_video_path=source_path,
        zoneIN_configuration_path=zones_IN_configuration_path,
        zoneOUT_configuration_path=zones_OUT_configuration_path    
    )
        processor.process_video()

        
