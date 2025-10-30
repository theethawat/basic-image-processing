import cv2
import os
import importlib
import initial_detectron as initial
import matplotlib.pyplot as plt
import sys
import torch

importlib.reload(initial)
sys.path.append("../algorithms")

input_video_path = "../video-dataset/1741177313185.mp4"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

initial.initial_detectron()
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

import sort as srt
import cat_predictor as cp

importlib.reload(cp)
importlib.reload(srt)


# Load the Video
cap = cv2.VideoCapture(input_video_path)

# Get Video Properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


# get cat predictor
cat_predictor = cp.get_predictor()

count = 0

os.makedirs("frames", exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cat_metadata = MetadataCatalog.get("cat_train")
my_sort = srt.Sort()

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
    if count < 10:
        print(f"Predicted Class Name {predicted_class_names}")
        print(f'Updated Result: {updated_sort}')
    v = Visualizer(
        frame[:, :, ::-1],
        metadata=cat_metadata,
        scale=0.5,
        instance_mode=ColorMode.IMAGE_BW,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.savefig(f"frames/captured_frame_{count}.jpg")
    # You need to run the detection here.
    count += 1
