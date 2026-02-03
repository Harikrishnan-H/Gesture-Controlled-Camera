"""
    Gesture Controlled Camera Project
    Harikrishnan Haridas
    
    Gesture-based image capture using Raspberry Pi and IMX500.

    Initializes the IMX500 neural network, loads label data,
    configures Picamera2 for both preview and optional high resolution still capture,
    starts the live preview with detection overlays, continuously processes
    detections in real time, and captures images based on gesture validation logic.

    Behaviour:
    - Runs object detection on live preview frames.
    - Validates face palm distance against configured thresholds.
    - Triggers timed capture with cooldown and delay enforcement.
    - Supports two capture modes:
        * HIGH  - captures 1920x1080 clean image without overlays
        * LOW   - captures preview image including overlays

    Reads:
        --model        Path to IMX500 model (.rpk)
        --labels       Path to labels text file
        --resolution   "high" or "low" image capture mode

   Results:
        Displays live preview.
        Saves captured images to SAVE_DIR.
        
    Run the script: 
        
    cd ~/Project_Team6/yolo-uv

    uv run python ../script/Gesture_Controlled_Camera_Team6.py \
  --model ../model/network.rpk \
  --labels ../model/labels.txt \
  --resolution high


"""


# ===================== LIBRARIES =====================
import argparse #Library Module to parse command-line arguments
import os
import time 
import math
from pathlib import Path
from functools import lru_cache #For cache results of get_labels()
import cv2 #For drawing and image operations
import numpy as np 
from picamera2 import MappedArray, Picamera2 #Picamera2 core classes
from picamera2.devices import IMX500 #IMX500 device integration
from picamera2.devices.imx500 import (
    NetworkIntrinsics,
    postprocess_nanodet_detection
) #Information about model/network
from picamera2.devices.imx500.postprocess import scale_boxes #For resizing detection boxes


# ===================== CONFIG =====================

MIN_DIST = 1.0 #Min Normalized Distance between face and palm. If below, then "too close"
MAX_DIST = 2.0  #Max Normalized Distance between face and palm. If above, then "too far"
CAPTURE_COOLDOWN = 3 #Min Time(in seconds) between two capture triggers
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "captured_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ===================== GLOBALS =====================
last_detections = [] #Detections from last frame
last_results = None #Latest Detection result
last_capture_time = 0 #Timestamp of last capture
last_distance = None #Last calculated distance between face and palm
last_status = "N/A" #Last status related to distance("VALID","TOO CLOSE" or "TOO FAR")
capture_requested = False #Flag indicating the capture has been requested
show_capturing_overlay = False #Flag to "CAPTURING" on preview window
pending_filename = None #File path where the next captured image will be saved
capture_trigger_time = 0 #Timestamp when capture was requested (start of delay)
CAPTURE_DELAY = 3.0  #Delay in seconds between trigger and actual capture

# ===================== DETECTION CLASS =====================
class Detection:
    """
    Simple container for a single detection result.
    Stores:
      - category: class/index of the detected object
      - conf: confidence score of the detection
      - box: bounding box in camera (preview) coordinates
    """
    def __init__(self, coords, category, conf, metadata):
        self.category = category #Detected class index
        self.conf = conf #Confidence Score for detection
        self.box = imx500.convert_inference_coords(coords, metadata, picam2) #Detection coordinate conversion from inference space to camera preview space

# ===================== PARSE DETECTIONS =====================
def parse_detections(metadata):
    """
    Takes camera metadata, reads the IMX500 neural network outputs,
    post-processes them into bounding boxes, scores, and classes,
    and returns a list of Detection objects filtered by threshold.
    """
    global last_detections
    threshold = args.threshold #Confidence threshold from command-line argument
    iou = args.iou #IOU threshold for postprocessing
    max_detections = args.max_detections #Max number of detections to keep

    np_outputs = imx500.get_outputs(metadata, add_batch=True) #Retrieve Neural Network Outputs from IMX500 for a frame
    if np_outputs is None: 
        return last_detections #If outputs are missing, reuse last detections to avoid glitches

    input_w, input_h = imx500.get_input_size() #Get input dimensions expected by the network
    if intrinsics.postprocess == "nanodet": #Post-process based on the type of model
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=threshold,
            iou_thres=iou,
            max_out_dets=max_detections
        )[0] #NanoDet postprocess helper
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False) #Scale bounding boxes back to input resolution
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0] #Generic 3-output model(boxes, scores, classes)
        boxes = boxes / input_h #Normalize boxes by input height
        boxes = boxes[:, [1, 0, 3, 2]] #Reorder box coordinates from (y1, x1, y2, x2) to (x1, y1, x2, y2)
        boxes = np.array_split(boxes, 4, axis=1) #Split into four separate arrays for x, y, w, h
        boxes = zip(*boxes)

    #Create Detection objects for each valid detection above threshold
    last_detections = [
        Detection(box, cls, score, metadata)
        for box, score, cls in zip(boxes, scores, classes)
        if score >= threshold
    ]
    return last_detections

# ===================== LABELS =====================
@lru_cache
def get_labels():
    """
    Returns the list of class labels for the model.
    Cached using lru_cache so the file is not re-read every time.
    """
    return intrinsics.labels

# ===================== DISTANCE UTILS =====================
def get_center(box):
    """
    Given a bounding box in (x, y, w, h) format, return the center point (cx, cy).
    """
    x, y, w, h = box
    return x + w // 2, y + h // 2

def normalized_distance(face_box, palm_box):
    """
    Computes the normalized distance between the centers of the face and palm boxes.
    Normalization uses the face width, making the distance scale-invariant.
    """
    fx, fy = get_center(face_box) #Face center 
    px, py = get_center(palm_box) #Palm center 
    pixel_dist = math.sqrt((fx - px) ** 2 + (fy - py) ** 2) #Euclidean distance between face center and palm center(in pixels)
    face_width = face_box[2] #Width of face bounding box
    return pixel_dist / face_width #Normalize distance by face width

# ===================== DISTANCE VALIDATION =====================
def valid_gesture_distance(detections):
    """
    Checks if both face and palm are detected and whether their
    normalized distance is within the accepted range (MIN_DIST, MAX_DIST).
    Updates last_distance and last_status and returns True/False.
    """
    global last_distance, last_status
    labels = get_labels() #List of class labels
    face_det = None #Detection for face
    palm_det = None #Detection for palm gesture

    for det in detections: #Face and palm detections from the list
        label = labels[int(det.category)]
        if label == "Face":
            face_det = det
        elif label == "Palm_Gesture":
            palm_det = det

    if face_det is None or palm_det is None: #If none, then gesture is not valid
        last_distance = None
        last_status = "Face or Palm Missing"
        return False

    dist = normalized_distance(face_det.box, palm_det.box) #Normalized distance between face and palm
    last_distance = dist #For display

    #Distance range and status
    if dist < MIN_DIST:
        last_status = "TOO CLOSE"
        return False
    elif dist > MAX_DIST:
        last_status = "TOO FAR"
        return False
    else:
        last_status = "VALID"
        return True

# ===================== DRAW + CAPTURE =====================
def draw_detections(request, stream="main"):
    """
    pre_callback function for Picamera2.
    Runs before each frame is displayed:
      - draws bounding boxes and labels
      - draws distance and status text
      - optionally draws a 'CAPTURING...' overlay
      - checks capture condition and sets capture_requested flag
    """
    global capture_requested, pending_filename
    global show_capturing_overlay, capture_trigger_time, last_capture_time

    #Latest Detection results
    detections = last_results
    if detections is None:
        return

    labels = get_labels() #Label names for classes
    with MappedArray(request, stream) as m: #Map current frame buffer, in order to draw on OpenCV
        #Draw detections
        for det in detections:
            x, y, w, h = det.box
            label = f"{labels[int(det.category)]} ({det.conf:.2f})"
            #Recatngle around detected object
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #Label text above bounding box
            cv2.putText(m.array, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        #Distance + status
        if last_distance is not None:
            #Normalized distance at top-left
            cv2.putText(m.array, f"Distance: {last_distance:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)
            #Display the status
            cv2.putText(m.array, f"Status: {last_status}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if last_status == "VALID" else (0, 0, 255), 2)

        #CAPTURING overlay (visible only during delay)
        if show_capturing_overlay:
            overlay = m.array.copy() #Copy of the frame to draw a translucent display
            #Rectangle at the top of the frame
            cv2.rectangle(overlay, (0, 0),
                          (m.array.shape[1], 90),
                          (0, 0, 0), cv2.FILLED)
            #Blend overlay into original frame for translucency
            cv2.addWeighted(overlay, 0.4, m.array, 0.6, 0, m.array)
            #Draw "CAPTURING" text on overlay
            cv2.putText(m.array, "CAPTURING...",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3)

        #Trigger capture (NO capture here)
        current_time = time.time()
        # Check if:
        #  - capture is not already requested
        #  - gesture distance is valid
        #  - cooldown time has passed since last capture
        if (not capture_requested and valid_gesture_distance(detections) and current_time - last_capture_time > CAPTURE_COOLDOWN):
            pending_filename = SAVE_DIR / f"capture_{int(current_time)}.jpg" #Filename for next image using timestamp
            capture_requested = True #Signal capture request
            show_capturing_overlay = True #Capturing overlay during delay
            capture_trigger_time = current_time #Time of capture trigger
            last_capture_time = current_time #Update time of capture trigger 

# ===================== ARGUMENTS =====================
def get_args():
    """
    Defines and parses the command-line arguments for this script.
    Returns an object 'args' with attributes:
      - model: path to the IMX500 model file (.rpk)
      - labels: path to the labels text file
      - resolution: 'high' or 'low' (controls capture mode)
      - threshold: confidence threshold for detections
      - iou: IOU threshold for postprocessing
      - max_detections: maximum number of detections to keep
    """
    parser = argparse.ArgumentParser() #Argument parser 
    parser.add_argument("--model", type=str, required=True) #Path to model file
    parser.add_argument("--labels", type=str, required=True) #Path to labels file
    parser.add_argument("--resolution", type=str, default="low") #"high" for high-resolution and "low" for low-resolution image capture
    parser.add_argument("--threshold", type=float, default=0.55) #Detection confidence threshold
    parser.add_argument("--iou", type=float, default=0.65) #IOU threshold for postprocessing
    parser.add_argument("--max-detections", type=int, default=10) #Max detections per frame
    return parser.parse_args() #parse and return arguments

# ===================== MAIN =====================  

if __name__ == "__main__":
    args = get_args() #Command-line arguments parsing
    imx500 = IMX500(args.model) #Create IMX500 device using the provided model path
    resolution = args.resolution #Resolution Mode from arguments
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics() #Get network intrinsics (model meta-information) or default if missing
    intrinsics.task = "object detection" #Setting task type explicitly

    #Load labels from the labels file and assign to intrinsics
    with open(args.labels, "r") as f:
        intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults() #Filling missing intrinsics fields with defaults

    picam2 = Picamera2(imx500.camera_num) #Picamera2 instance using IMX500 camera index

    #Create a separate still configuration for high resolution capture
    still_config = picam2.create_still_configuration(main={"size": (1920, 1080)})  # high-res, no overlays

    #Preview config and Start
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}  #Uses intrinsics.inference_rate for the preview frame rate
    )

    imx500.show_network_fw_progress_bar() #To show progress while loading network firmware into IMX500
    picam2.start(config, show_preview=True) #Start the camera with the preview configuration and enable the preview

    picam2.pre_callback = draw_detections #Register draw_detections as the pre-callback, so it runs before each frame is shown

    while True: #Main loop
        last_results = parse_detections(picam2.capture_metadata()) #Capture metadata from the camera and parse detections from IMX500

        if capture_requested: #Capture has been requested (valid gesture and passed cooldown)
            elapsed = time.time() - capture_trigger_time  #Compute how long ago the capture was triggered

            # After the delay, proceed to capture
            if elapsed >= CAPTURE_DELAY:
                show_capturing_overlay = False   # overlay disappears
                print("[INFO] Capturing still image...")

                #Choose capture path based on RESOLUTION
                if resolution.upper() == "HIGH":
                    #Temporarily disable pre_callback so NO boxes / overlays on the still
                    old_callback = picam2.pre_callback
                    picam2.pre_callback = None
                    #High-res, clean still: switch to still_config, capture, then restore
                    picam2.switch_mode_and_capture_file(still_config, str(pending_filename))
                    #Restore preview callback for live stream
                    picam2.pre_callback = old_callback
                else:
                    #Capture from current preview configuration
                    picam2.capture_file(str(pending_filename))

                print(f"[INFO] Image saved: {pending_filename}") #Log that the image has been saved
                capture_requested = False #Reset the capture request flag
    
