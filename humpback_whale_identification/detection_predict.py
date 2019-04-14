import os
import sys
import numpy as np
import pandas as pd
import time
import json
import cv2

from src import util
from models.retinanet.keras_retinanet import models
from models.retinanet.keras_retinanet.utils.image import *

with open('SETTINGS.json') as f:
    json_data = json.load(f)
retinanet_path = json_data["DETECTION_MODEL"]
retinanet_model = models.load_model(os.path.join('models/retinanet', 'snapshots', 'resnet50_coco_best_v2.1.0.h5'), backbone_name='resnet50')

test_dir = json_data["TEST_DIR"]
submission_dir = json_data["SUBMISSION_DIR"]

sz = 512    # todo image size to be checked

# threshold for non-max-suppression for each model
nms_threshold = 0

# threshold for including boxes from retinanet
score_threshold = 0.05

# threshold for including isolated boxes from either model
solo_min = 0.15

# test_ids = []
test_outputs = []

start = time.time()
for i, fname in enumerate(os.listdir(test_dir)[1:2]):
    print("Predicting boxes for image # {}\r".format(i+1))
    fpath = os.path.join(test_dir, fname)
    fid = fname[:-4]

    # boxes_pred, scores = util.get_detection_from_file(fpath, retinanet_model, sz)
    img = read_image_bgr(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    # img = resize_image(img)
    boxes_pred, scores, labels = retinanet_model.predict_on_batch(np.expand_dims(img, axis=0))  # todo write generator

    boxes_pred = boxes_pred[0]
    scores = scores[0]
    indices = np.where(scores > score_threshold)[0]
    scores = scores[indices]
    boxes_pred = boxes_pred[indices]
    boxes_pred = util.nms(boxes_pred, scores, nms_threshold)[0]  # choose the first one

    x1 = boxes_pred[0]
    y1 = boxes_pred[1]
    x2 = boxes_pred[2]
    y2 = boxes_pred[3]

    # test_ids.append(fid)
    test_outputs.append((fid, x1, y1, x2, y2))
end = time.time()
print("Elapsed time = {0:.3f} seconds".format(end - start))

if __name__ == '__main__':
    pd.DataFrame(test_outputs).to_csv('locations.csv', header=False, index=False)
    print("Finished")
