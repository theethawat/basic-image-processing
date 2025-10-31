import cv2
import os
import importlib
import initial_detectron as initial
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np

importlib.reload(initial)
sys.path.append("../algorithms")

input_video_path = "../video-dataset/20230812_180148.mp4"
input_video_split = input_video_path.split("/")[-1].split(".")
output_video_path = f"output_{input_video_split[0]}.mp4"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

initial.initial_detectron()
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

import sort as srt
import cat_predictor as cp

importlib.reload(cp)
importlib.reload(srt)


# Helper Functions
def get_box_center(box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


# Load the Video
cap = cv2.VideoCapture(input_video_path)

# Get Video Properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Define codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# get cat predictor
cat_predictor = cp.get_predictor()

count = 0

os.makedirs("frames", exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cat_metadata = MetadataCatalog.get("cat_train")
my_sort = srt.Sort()
trajectories = {}  # Add before the while loop

tracked_classes = {}

# Processing on Each Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    outputs = cat_predictor(frame)
    instances = outputs["instances"]
    predicted_class_ids_tensor = instances.pred_classes
    predicted_class_ids = predicted_class_ids_tensor.cpu().tolist()
    cat_train_class = cat_metadata.thing_classes
    predicted_class_names = [
        cat_train_class[class_id] for class_id in predicted_class_ids
    ]

    # print('Field',instances._fields)
    # print('scores',instances.scores)
    score_tensor = instances.scores
    score_tensor_size = score_tensor.size(dim=0)
    score_tensor = torch.reshape(score_tensor,[score_tensor_size,1])
    combined_tensor = torch.cat((instances.pred_boxes.tensor, score_tensor), dim=1)
    combined_tensor_np = combined_tensor.cpu().numpy()

    # Update the tracking in SORT 
    updated_sort = my_sort.update(combined_tensor_np) 
    for track in updated_sort:
        x1, y1, x2, y2, track_id = track
        tracked_center = get_box_center((x1, y1, x2, y2))
        # Find the closed predicted class for this track

        min_distance = float('inf')
        assigned_class = "unknown"
        
        prediction_box_tensor = instances.pred_boxes.tensor.cpu().numpy()
        for i, box in enumerate(prediction_box_tensor):
            prediction_box_center = get_box_center(box)
            distance =  np.linalg.norm(np.array(tracked_center) - np.array(prediction_box_center))
            # If this prediction box is close, update the assigned class
            if distance < min_distance:
                min_distance = distance
                assigned_class = predicted_class_names[i]
        
        # Update the tracked_classes dictionary
        tracked_classes[track_id] = assigned_class
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((count, x1, y1, x2, y2))
        
        # Draw bounding box and track ID on frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"ID:{track_id} {assigned_class}", (int(x1), int(y1)-10),
                    cv2.QT_FONT_NORMAL, 0.8, (0,255,0), 2)
    # if count < 10:
    #     print(f"Predicted Class Name {predicted_class_names}")
    #     print(f'Updated Result: {updated_sort}')
    v = Visualizer(
        frame[:, :, ::-1],
        metadata=cat_metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW,
    )
    predicted_image = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite(f"frames/captured_frame_{count}.jpg", frame)
    out.write(frame)
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(predicted_image.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.savefig(f"frames/captured_frame_{count}.jpg")
    # You need to run the detection here.
    plt.close()
    count += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


with open("trajectory.txt", "w") as f:
    for track_id in sorted(trajectories):
        f.write(f"Track ID {int(track_id)}\n")
        for frame_idx, x1, y1, x2, y2 in trajectories[track_id]:
            f.write(f"Frame Index{frame_idx} X1:{x1:.2f} Y2:{y1:.2f} X2:{x2:.2f} Y2:{y2:.2f}\n")
        f.write("\n")
