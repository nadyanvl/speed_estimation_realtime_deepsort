from ultralytics import YOLO
import cv2
from helper import create_video_writer
from ultralytics.utils.plotting import Annotator
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import random

class_colors = {
    'Bicycle': (0, 0, 255),
    'Bus': (0, 255, 255),    
    'Car': (0, 127, 255),
    'Motorcycle': (0, 255, 127),
    'Train': (255, 127, 0),  
    'Truck': (255, 0, 255), 
    # Add more class-color mappings as needed
}

CONFIDENCE_THRESHOLD = 0.4
WHITE = (255, 255, 255)

# initialize the video capture object
video_cap = cv2.VideoCapture("assets/testing1.mp4")
# initialize the video writer object
writer = create_video_writer(video_cap, "output_vehicle_test.mp4")

# load the pre-trained YOLOv8n model
model = YOLO("trained_model/vehicle_model2.pt")
#model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=10)

# Dictionary to store the previous positions of tracked objects
prev_positions = {}

while True:
    start = time.time()

    ret, frame = video_cap.read()

    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]
    #annotator = Annotator(frame)
    # initialize the list of bounding boxes and confidences
    results = []

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        class_name = model.names[class_id]
        # add the bounding box (x, y, w, h), confidence, and class id to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_name])

    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)

    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue
        
        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()
        det_class = track.det_class
        
        # Assign a unique color to each class based on the mapping
        if det_class in class_colors:
            color = class_colors[det_class]
        else:
            # Assign a random color if the class is not in the mapping
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Initialize speed_pixels_per_sec
        speed_pixels_per_sec = 0

        # Calculate the speed of the object
        if track_id in prev_positions:
            prev_x, prev_y, prev_time = prev_positions[track_id]
            current_time = time.time()
            time_diff = (current_time - prev_time)

            # Calculate distance traveled (assuming constant speed)
            distance = ((xmin - prev_x) ** 2 + (ymin - prev_y) ** 2) ** 0.5

            print(f'time_diff: {time_diff} - distance: {distance}')
            # Calculate speed in pixels per second
            speed_pixels_per_sec = distance / time_diff

            # convert speed_pixels_per_sec to other units (km/h) based on frame rate and object scale

            # Update the previous position
            prev_positions[track_id] = (xmin, ymin, current_time)
        else:
            prev_positions[track_id] = (xmin, ymin, time.time())

        # Draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), color, -1)
        cv2.putText(frame, f"{track_id} {det_class} | Speed: {speed_pixels_per_sec:.2f} px/s", (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        #annotator.box_label(ltrb, f"{track_id}: {det_class} | Speed: {speed_pixels_per_sec:.2f} px/s", color=color)
    # end time to compute the fps
    end = time.time()
    # show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start) * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start):.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    #frame = annotator.result()
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()
