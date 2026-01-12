import cv2
import math
from ultralytics import YOLO
from time import perf_counter
from kcf_tracker import TrackerKCF_create, TrackerReliabilityKCF_create

# --- CONFIGURATION ---
SCALE_FACTOR = 0.5
DETECTION_INTERVAL = 2
CONFIDENCE_THRESHOLD = 0.4
DISTANCE_THRESHOLD = 30  # Pixels. If center distance > this, it's a new object.
# TRACKER_CLASS = cv2.TrackerCSRT_create
# TRACKER_CLASS = cv2.TrackerKCF_create
TRACKER_CLASS = TrackerReliabilityKCF_create

# Create yolo model for detection people
model = YOLO("models/yolov8x.pt") 

# Create video reader
video = cv2.VideoCapture("data/video_4.mp4")

trackers = []   # List for storing all trackers
frame_count = 0 # Number of frames processed
global_id_counter = 1   # Global ID that counts number of unique people tracked since the beggining
ids_in_roi = []  # Ids of objects that entered ROI

# Function that returns center point of bounding box
def get_center(bbox):
    """Calculates the center point (x, y) of a bounding box."""
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    return cx, cy

# Function that checks if point is inside the ROI
def point_in_roi(roi, point):
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    x, y = point
    if roi_x1 < x < roi_x2 and roi_y1 < y < roi_y2:
        return True
    else:
        return False


ret, first_frame = video.read() # Read first frame

first_frame = cv2.resize(first_frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)   # Resize image for faster processing

# ROI selection for people counting
roi_x, roi_y, roi_w, roi_h = cv2.selectROI("Tracking", first_frame, showCrosshair=True)
roi_x1, roi_y1 = roi_x, roi_y
roi_x2 = roi_x + roi_w
roi_y2 = roi_y + roi_h
roi = [roi_x1, roi_y1, roi_x2, roi_y2]

# For calculating fps
start_time = perf_counter()

while True:
    ret, original_frame = video.read()
    if not ret: break
    
    # Resize image for faster processing
    frame = cv2.resize(original_frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        
    if frame_count % DETECTION_INTERVAL == 0:
        # Detect people using YOLO
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, classes=[0], imgsz=640)
        
        # Parse YOLO detections into a clean list of [x, y, w, h]
        detections = []
        for result in results:
            for box in result.boxes:    # Get the left top corner of bbox, width and height 
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = int(x2 - x1)
                h = int(y2 - y1)
                detections.append((int(x1), int(y1), w, h))

        # 2. MATCHING (Euclidean Distance)
        # Mark all current trackers as "not updated" for this frame
        for obj in trackers:
            obj['updated'] = False

        for det_box in detections:
            det_center = get_center(det_box)
            
            matched_obj = None
            min_dist = float('inf')

            # Find the closest existing tracker
            for obj in trackers:
                # If we don't have a last_bbox (newly added), skip or assume center
                if 'last_bbox' not in obj: continue
                
                tr_center = get_center(obj['last_bbox'])
                
                # Calculate Euclidean Distance
                dist = math.hypot(det_center[0] - tr_center[0], det_center[1] - tr_center[1])
                
                if dist < min_dist:
                    min_dist = dist
                    matched_obj = obj

            # DECISION: Is it the same object?
            if matched_obj and min_dist < DISTANCE_THRESHOLD:
                # YES -> Update existing tracker (Re-init to fix drift)
                # We do NOT increment the ID, we keep matched_obj['id']
                if not matched_obj['updated']: # Prevent multiple detections claiming one tracker
                    matched_obj['tracker'] = TRACKER_CLASS()
                    matched_obj['tracker'].init(frame, det_box)
                    matched_obj['last_bbox'] = det_box
                    matched_obj['updated'] = True
            else:
                # NO -> It's a new object
                new_tracker = TRACKER_CLASS()
                new_tracker.init(frame, det_box)
                trackers.append({
                    'tracker': new_tracker,
                    'id': global_id_counter,
                    'color': (0, 255, 0),
                    'last_bbox': det_box,
                    'updated': True
                })
                global_id_counter += 1
                
        # Optional: Prune trackers that were NOT updated by YOLO?
        # If you want to kill trackers that YOLO didn't see, uncomment below:
        # trackers = [obj for obj in trackers if obj['updated']]
            
    # else:
    # TRACKING LOOP (Intermediate frames)
    for obj in trackers:
        success, bbox = obj['tracker'].update(frame)
        
        if success:
            # Store bbox for the next distance calculation
            obj['last_bbox'] = bbox
            
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, obj['color'], 2, 1)
            
            text = f"ID: {obj['id']}"
            cv2.putText(frame, text, (p1[0], p1[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj['color'], 1)
        else:
            # If KCF loses the object, remove it
            trackers.remove(obj)

    for obj in trackers:
        bbox = obj['last_bbox']
        bbox_center = get_center(bbox)
        if point_in_roi(roi, bbox_center):
            if obj['id'] not in ids_in_roi:
                ids_in_roi.append(obj['id'])

    cv2.putText(frame, f"People counter: {len(ids_in_roi)}", (5, 15), 0, 0.5, (0, 0, 255), 2)

    # Draw ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

    # Measure fps
    curr_time = perf_counter()
    fps = frame_count / (curr_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (5, 35), 0, 0.5, (0, 0, 255), 2)
    prev_time = curr_time

    frame_count += 1
    cv2.imshow("Tracking", cv2.resize(frame, None, fx=1/SCALE_FACTOR, fy=1/SCALE_FACTOR))
    
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'): break

video.release()
cv2.destroyAllWindows()