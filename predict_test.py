import subprocess
from pathlib import Path
import pandas as pd
from utils import sliding_window, make_sliding_windows
import numpy as np
import pickle
from utils import create_bounding_boxes_df
from skimage.feature import hog
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from utils import read_img, create_map, connect_components, create_submission_file

DATASET = "test"
MODEL_PATH = "models/randomforest.model"
model = pickle.load(open(MODEL_PATH, "rb"))
sizes = [(64,64),(128,128),(256,256)]
step_sizes = [(16, 16), (32, 32), (64, 64)]
y_start_stop= [(0.3, 0.56), (0.2, 0.7), (0.35, .7)]
threshold = 0.3

boxes_list = []

test_set_path = Path(DATASET)
if not Path(test_set_path).exists():
    # Download the dataset
    subprocess.run(['bash', 'car_dectector/scripts/download_test_dataset.sh'])

test_files = [x.as_posix() for x in test_set_path.glob('**/*')]
test_files = sorted(test_files)

for image_path in tqdm(test_files):
    image = read_img(image_path)
    boxes = []
#    for size in sizes:
    for window, coords in make_sliding_windows(image, sizes, step_sizes, y_start_stop):
        # resizing the image
        new_size = (64, 64)
        resized_window = cv2.resize(window, new_size)
        
        # Extracting the features
        fd, hog_image = hog(resized_window, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), visualize=True,channel_axis=-1)

        # Features reshaping
        fd = fd.reshape(-1, 1)
        fd = np.transpose(fd,(1,0))

        # Getting a probability
        proba = model.predict_proba(fd)
        
        if proba[0][1] > 0.6:
            boxes += [coords[0],coords[1] , coords[2], coords[3], proba[0][1]]
    
    heatmap = create_map(boxes, image,threshold)
    plt.imsave("binary_heatmap.png",heatmap)    
    merged_bboxes = connect_components(heatmap)
    boxes_list.append(merged_bboxes)
    # Draw bounding boxes on image
    
    for bbox in merged_bboxes:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+ bbox[2], bbox[1]+ bbox[3]), (0, 255, 0), 2)
    new_file = Path("results") / Path(image_path).name
    cv2.imwrite(new_file.as_posix(), image)



create_submission_file(boxes_list,test_files)  
print("Done")