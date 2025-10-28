import os
import numpy as np
import json
from detectron2.structures import BoxMode
import pandas as pd
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog


def get_cat_dict(directory, jsonfile_name):
    classes = ["Khawlahm", "Khawniewping", "Khawtang"]
    dataset_dicts = []
    json_file = os.path.join(directory, jsonfile_name)
    with open(json_file) as f:
        json_ann_file = json.load(f)

    image_list = json_ann_file["images"]
    annotation_list = pd.DataFrame(json_ann_file["annotations"])

    for image in image_list:
        record = {}
        filename = os.path.join(directory, image["file_name"])

        record["file_name"] = filename
        record["height"] = image["height"]
        record["width"] = image["width"]
        image_id = image["id"]

        # find the annotation of this image
        annos = annotation_list[annotation_list["image_id"] == image_id]
        objs = []
        for idx, anno in annos.iterrows():
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": anno["segmentation"],
                "category_id": anno["category_id"],
                "iscrowd": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

print("-------------")

for d in ["train", "test"]:
    DatasetCatalog.register(
        "cat_" + d, lambda d=d: get_cat_dict("cat-dataset/", "result.json")
    )
    MetadataCatalog.get("cat_" + d).set(
        thing_classes=["Khawlahm", "Khawniewping", "Khawtang"]
    )

cat_metadata = MetadataCatalog.get("cat_train")

print("Config Initialization will be continue")

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("cat_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("It will initial the trainer")

try:
    trainer = DefaultTrainer(cfg)
except Exception as e:
    # By this way we can know about the type of error occurring
    print("The error is: ", e)

print("Trainer is initial success")
print(trainer)
