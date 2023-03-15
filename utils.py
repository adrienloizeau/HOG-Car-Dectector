import numpy as np
import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest
from instruction_utils import run_length_encoding, bounding_boxes_to_mask

def sliding_window_OLD(image, window_size=64, stride=20, padding=0):
    """
    Slide a window across an image with a given stride and padding.

    Args:
        image: The input image.
        window_size: The size of the window.
        stride: The stride to slide the window.
        padding: The amount of padding to apply to the image.
    
    Returns:
        A generator that yields the sliding windows along with their starting and ending coordinates.
    """
    # Add padding to the image
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

    # Loop over the sliding window
    for y in range(max(padding, 200), min(padded_image.shape[0] - window_size + 1, 500), stride):
        for x in range(0, padded_image.shape[1] - window_size + 1, stride):
            yield padded_image[y:y+window_size, x:x+window_size], (x, y, window_size, window_size)

def sliding_window(image, window_size, step_size, y_start=0.2, y_stop=0.7):
    for y in range(int(image.shape[0]*y_start), int(image.shape[0] * y_stop) - window_size[0], step_size[1]):
        for x in range(0, image.shape[1] - window_size[1], step_size[0]):
            yield (image[y: y + window_size[1], x: x + window_size[0]], (x, y,window_size[0],window_size[1]))

def make_sliding_windows(image, window_sizes, step_sizes, starts_stops):
    for i in range(len(window_sizes)):
        yield from sliding_window(image, window_sizes[i], step_sizes[i], y_start=starts_stops[i][0], y_stop=starts_stops[i][1])

import cv2
import numpy as np
from itertools import zip_longest

# Load image and boxes
image = cv2.imread("test/001.jpg")

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def overlap(box1, box2, threshold=0.5):
    """
    Determine if two boxes overlap based on their intersection over union (IoU).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return False
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection / float(area1 + area2 - intersection)
    return iou >= threshold

# Create heatmap
def create_map(boxes, img, threshold):
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for x,y,w, h, proba in grouper(5, boxes):
        heatmap[y:y+h,x:x+w] += proba*100

    max = heatmap.max()
    x,y  = heatmap.shape
    for i in range(x):
        for j in range(y):
            heatmap[i,j]= heatmap[i,j]/max
    
    plt.imsave("heatmap.png", heatmap)
    # Threshold heatmap
    heatmap[heatmap < threshold] = 0.0
    heatmap[heatmap >= threshold] = 1.0
    return heatmap

def connect_components(heatmap):
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(heatmap.astype(np.uint8))

    # Compute bounding boxes
    bboxes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append([x, y, width, height])

    # Merge overlapping bounding boxes
    merged_bboxes = []
    while len(bboxes) > 0:
        bbox = bboxes.pop(0)
        merged_bbox = bbox
        i = 0
        while i < len(bboxes):
            if overlap(merged_bbox, bboxes[i]):
                merged_bbox = merge(merged_bbox, bboxes[i])
                bboxes.pop(i)
            else:
                i += 1
        merged_bboxes.append(merged_bbox)
    return merged_bboxes

def create_submission_file(boxes_list, files_names):
    rows = []
    for i, file_name in enumerate(files_names):
        rle = run_length_encoding(bounding_boxes_to_mask(boxes_list[i], H, W))
        rows.append([file_name, rle])

    df_prediction = pd.DataFrame(columns=['Id', 'Predicted'], data=rows).set_index('Id')
    df_prediction.to_csv('sample_submission2.csv')
    print("file created")

def read_img(path: str):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_bounding_boxes_df(images, frame_ids=["train/A_001.jpg"]):
    """
    Create a dataframe with the bounding boxes in the right format to use the show annotation function.
    Args:
        boxes (numpy.ndarray or list): An array or list of bounding box coordinates.
        frame_id (str): The ID of the frame to which the bounding boxes belong.
    Returns:
        A Pandas dataframe with the bounding box coordinates in the correct format.
    """
    # Check if boxes is not an array and convert to a string format
    # Convert array of boxes to list of strings
    bounding_boxes_str = ""
    for img in images:
        for boxes in img:
            bounding_boxes_str += ' '.join(str(coord) for coord in boxes)
    # Create dataframe with list of strings
    df_boxes = pd.DataFrame(data={"frame_id": frame_ids, "bounding_boxes": bounding_boxes_str})
    
    return df_boxes

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='train/A_001.jpg', help='Path of the image')
    parser.add_argument('--window_size', type=int, default= 64, help='Size of the window')
    parser.add_argument('--stride', type=int, default=10, help='Size of the stride')
    parser.add_argument('--padding', type=int, default=0, help='Size of the padding')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    image = image[:,:,::-1]
    window_size = args.window_size
    stride = args.stride
    padding = args.padding

    for window in sliding_window(image, window_size, stride, padding):
        pass
    print(window.shape)
    print(window)
    fig, ax = plt.subplots()
    ax.imshow(window)
    plt.savefig("window.png")
