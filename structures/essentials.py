from ultralytics import YOLO

import cv2

import supervision as sv

import json
import os
from typing import Any, Optional, Tuple, Dict, Iterable, List, Set
import cv2
import numpy as np
import streamlit as st
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


def drawzones(source_path, zone_configuration_path):
    
    def resolve_source(source_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(source_path):
            return None
        if(source_path!="D:\\repos\\deploy\\videos\\Footage_Feed_5.MP4"):
            image = cv2.imread(source_path)
            if image is not None:
                return image
        else:
            image = cv2.imread("D:\\repos\\deploy\\videos\\Footage_Feed_5.jpg")
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
