from ultralytics import YOLO

import streamlit as st
import cv2

import supervision as sv


import settings

from typing import Any, Optional, Tuple, Dict, Iterable, List, Set

import cv2
import numpy as np
from inference import get_roboflow_model
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer
from inference import InferencePipeline

from structures.CustomSink import CustomSink

KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
FRAME_NUM = 0
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
                                    if(time%60 >= float(violation_time)):
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
