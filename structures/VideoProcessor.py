from ultralytics import YOLO

import streamlit as st
import cv2

import supervision as sv

from tqdm import tqdm

from typing import Any, Optional, Tuple, Dict, Iterable, List, Set

import cv2
import numpy as np
from utils.general import load_zones_config

from collections import Counter

from structures.DetectionManager import DetectionsManager

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
curr_count_dict={}
prev_count_dict= {}
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)
current_mouse_position: Optional[Tuple[int, int]] = None


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
        time: int = 2
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.zoneIN_configuration_path = zoneIN_configuration_path
        self.zoneOUT_configuration_path = zoneOUT_configuration_path
        self.time = time
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
                st.subheader("ALERTS: ")
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
        global curr_count_dict
        global prev_count_dict
        global FRAME_NUM
        FRAME_NUM +=1
        called = 0
        def updateCount():
            curr_count_dict[(zone_out_id,zone_in_id)] = count

        
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
                    updateCount()
                    if(FRAME_NUM%(24*60*self.time)==0 and not called):
                        
                        temp_1 = Counter(curr_count_dict)
                        temp_1.subtract(prev_count_dict)
                        temp_1 = dict(temp_1)
                        st.warning(temp_1)
                        prev_count_dict = curr_count_dict.copy()
                        called = 1
                        
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
