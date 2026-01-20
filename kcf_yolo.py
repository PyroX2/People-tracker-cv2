import cv2
import math
from ultralytics import YOLO
from time import perf_counter
from kcf_tracker import TrackerKCF_create
from segment import Segmentator


# --- CONFIGURATION ---
SCALE_FACTOR = 1
DETECTION_INTERVAL = 5
CONFIDENCE_THRESHOLD = 0.35
MISSES_ALLOWED = 2
IOU_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 50  # Pixels. If center distance > this, it's a new object.
# TRACKER_CLASS = cv2.TrackerCSRT_create
# TRACKER_CLASS = cv2.TrackerKCF_create
TRACKER_CLASS = TrackerKCF_create

last_click = None

def mouse_callback(event, x, y, flags, param):
    global last_click
    
    if event == cv2.EVENT_LBUTTONDOWN:
        last_click = (x, y)
        print(f"User clicked at: x={x}, y={y}")

# Function that returns center point of bounding box
def get_center(bbox):
    """Calculates the center point (x, y) of a bounding box."""
    x1, y1, x2, y2 = bbox
    cx = int((x2 + x1) / 2)
    cy = int((y2 + y1) / 2)
    return cx, cy

# Function that checks if point is inside the ROI
def point_in_roi(roi, point):
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    x, y = point
    if roi_x1 < x < roi_x2 and roi_y1 < y < roi_y2:
        return True
    else:
        return False
    

# Function that returns IoU of two bounding boxes in format [x1, y1, x2, y2]
def get_iou(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def xyxy2xywh(bbox):
    return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

def xywh2xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

def main():
    global last_click

    # Create yolo model for detection people
    model = YOLO("models/yolov8l.pt") 

    window_name = "Tracking"

    # Create video reader
    video = cv2.VideoCapture("data/aerial-view-of-crowds-2.mp4")


    trackers = []   # List for storing all trackers
    frame_count = 0 # Number of frames processed
    global_id_counter = 1   # Global ID that counts number of unique people tracked since the beggining
    ids_in_roi = []  # Ids of objects that entered ROI
    
    selected_id = None  # ID of selected object to display stats
    selected_color = [0, 0, 0]

    ret, first_frame = video.read() # Read first frame

    first_frame = cv2.resize(first_frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)   # Resize image for faster processing

    # ROI selection for people counting
    roi_x, roi_y, roi_w, roi_h = cv2.selectROI(window_name, first_frame, showCrosshair=True)
    roi_x1, roi_y1 = roi_x, roi_y
    roi_x2 = roi_x + roi_w
    roi_y2 = roi_y + roi_h
    roi = [roi_x1, roi_y1, roi_x2, roi_y2]

    segmentator = Segmentator()

    # For calculating fps
    start_time = perf_counter()

    cv2.namedWindow(window_name)
        
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, original_frame = video.read()
        if not ret: break
        
        # Resize image for faster processing
        frame = cv2.resize(original_frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        original_frame = frame.copy()

        if frame_count % DETECTION_INTERVAL == 0:
            # Detect people using YOLO
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, classes=[0], imgsz=640, iou=0.2)
            
            # Parse YOLO detections into a clean list of [x, y, w, h]
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append((int(x1), int(y1), int(x2), int(y2)))

            # Mark all current trackers as "not updated" for this frame
            for obj in trackers:
                obj['updated'] = False

            not_matched_dets = []

            # Match detection bboxes to existing tracking bboxes
            for det_box in detections:
                matched_obj = None
                best_iou = 0

                # Find the closest existing tracker
                for obj in trackers:
                    if obj['updated']: continue # Prevent assigning multiple det bboxes to one track bbox
                    track_bbox = obj['last_bbox']
                    iou = get_iou(det_box, track_bbox)
                    if iou > best_iou:
                        matched_obj = obj
                        best_iou = iou

                if matched_obj and best_iou > IOU_THRESHOLD:
                    # YES -> Update existing tracker (Re-init to fix drift)
                    # We do NOT increment the ID, we keep matched_obj['id']
                    matched_obj['tracker'] = TRACKER_CLASS()
                    matched_obj['tracker'].init(frame, xyxy2xywh(det_box))
                    matched_obj['last_bbox'] = det_box
                    matched_obj['updated'] = True
                else:
                    not_matched_dets.append(det_box)

            not_matched_trackers = [obj for obj in trackers if not obj['updated']]

            # Match using euclidian distance
            for det_box in not_matched_dets:
                det_center = get_center(det_box)

                det_img = frame[det_box[1]:det_box[3], det_box[0]:det_box[2]]

                matched_obj = None
                min_dist = float('inf')

                for obj in not_matched_trackers:
                    track_center = get_center(obj['last_bbox'])
                    dist = math.hypot(track_center[0]-det_center[0], track_center[1]-det_center[1])
                    
                    if dist < min_dist:
                        matched_obj=obj
                        min_dist = dist

                    tracker_img = frame[obj['last_bbox'][1]:obj['last_bbox'][3], obj['last_bbox'][0]:obj['last_bbox'][2]]
                
                if matched_obj and min_dist < DISTANCE_THRESHOLD:
                
                    matched_obj['tracker'] = TRACKER_CLASS()
                    matched_obj['tracker'].init(frame, xyxy2xywh(det_box))
                    matched_obj['last_bbox'] = det_box
                    matched_obj['updated'] = True
                else:
                    # Create new object
                    new_tracker = TRACKER_CLASS()
                    new_tracker.init(frame, xyxy2xywh(det_box))
                    trackers.append({
                        'tracker': new_tracker,
                        'id': global_id_counter,
                        'color': (0, 255, 0),
                        'last_bbox': det_box,
                        'updated': True,
                        'misses_counter': 0
                    })
                    global_id_counter += 1

            cleaned_trackers = []
            for obj in trackers:
                if obj['updated']:
                    obj['misses_counter'] = 0
                    cleaned_trackers.append(obj)
                else:
                    obj['misses_counter'] += 1
                    if obj['misses_counter'] > MISSES_ALLOWED:
                        continue
                    cleaned_trackers.append(obj)
            trackers = cleaned_trackers


            # trackers = [obj for obj in trackers if obj['updated']]

        for obj in trackers:
            success, bbox = obj['tracker'].update(frame)

            # If object was not detected dont draw it
            if obj['misses_counter'] > 0:
                continue

            bbox = xywh2xyxy(bbox)

            is_selected = False

            if last_click is not None:
                is_selected = point_in_roi(bbox, last_click)
                
            if is_selected:
                selected_id = obj["id"]
                selected_color = segmentator.get_dominant_color(original_frame, bbox)
                last_click = None
            
            if success:
                # Store bbox for the next distance calculation
                obj['last_bbox'] = bbox

                if obj["id"] == selected_id:
                    bbox_color = selected_color
                else:
                    bbox_color = obj['color']
                
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(frame, p1, p2, bbox_color, 2, 1)
                
                text = f"ID: {obj['id']}"
                cv2.putText(frame, text, (p1[0], p1[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
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
        cv2.imshow(window_name, cv2.resize(frame, None, fx=1/SCALE_FACTOR, fy=1/SCALE_FACTOR))
        
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'): break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()