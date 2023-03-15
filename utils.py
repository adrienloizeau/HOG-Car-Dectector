import numpy as np
import pandas as pd
import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest
from instruction_utils import run_length_encoding, bounding_boxes_to_mask

df_ground_truth = pd.read_csv('./train.csv')
H, W = 720, 1280
N = len(df_ground_truth)

def sliding_window(image, window_size, step_size, y_start=0.2, y_stop=0.7):
    """
    Generate sliding windows across the image.

    Args:
        image: The input image.
        window_size: The size of the window.
        step_size: The distance to slide the window.
        y_start: The starting vertical coordinate.
        y_stop: The stopping vertical coordinate.

    Returns:
        A generator that yields the sliding windows along with their starting and ending coordinates.
    """
    for y in range(int(image.shape[0]*y_start), int(image.shape[0] * y_stop) - window_size[0], step_size[1]):
        for x in range(0, image.shape[1] - window_size[1], step_size[0]):
            yield (image[y: y + window_size[1], x: x + window_size[0]], (x, y,window_size[0],window_size[1]))

def make_sliding_windows(image, window_sizes, step_sizes, starts_stops):
    """ 
    Generate sliding windows of multiple sizes.

    Args:
        image: The input image.
        window_sizes: A list of window sizes.
        step_sizes: A list of step sizes.
        starts_stops: A list of starting and stopping vertical coordinates.

    Returns:
        A generator that yields the sliding windows along with their starting and ending coordinates.
    """
    for i in range(len(window_sizes)):
        yield from sliding_window(image, window_sizes[i], step_sizes[i], y_start=starts_stops[i][0], y_stop=starts_stops[i][1])

def grouper(n, iterable, fillvalue=None):
    """ 
    Collect data into fixed-length chunks or blocks.

    Args:
        n: The number of items to group.
        iterable: The iterable to group.
        fillvalue: The value to use for padding when the iterable is exhausted.

    Returns:
        An iterable of tuples containing the grouped items.
    """
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def overlap(box1, box2, threshold=0.5):
    """
    Determine if two boxes overlap based on their intersection over union (IoU).

    Args:
        box1: The first box.
        box2: The second box.
        threshold: The threshold to use for the IoU
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

def create_map(boxes, img, comparaison_threshold, poor_threshold):
    """
    Creates a heatmap of the detected objects in an image.

    Args:
        boxes (numpy.ndarray): An array of detected object bounding boxes.
        img (numpy.ndarray): The input image.
        comparaison_threshold (float): Threshold value to compare the heatmap values.
        poor_threshold : Threshold value for poor detection: used when not a lot boxes are found
                        aren't enough boxes
    Returns:
        A heatmap of the detected objects.
    """

    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    
    # Add the probabilities to the heatmap in the corresponding locations
    for x,y,w, h, proba in grouper(5, boxes):
        heatmap[y:y+h,x:x+w] += proba*100

    num_boxes= len(boxes)/5
    x,y  = heatmap.shape

    # If there isn't a lot of boxes, only take the ones with value bigger than poor_threshold
    if num_boxes< 6:
        plt.imsave("heatmap.png", heatmap)
        heatmap[heatmap < poor_threshold] = 0.0
        heatmap[heatmap >= poor_threshold] = 1.0
        return heatmap
    else: 
        # Normalize the heatmap values and compare them to the comparison threshold
        max = heatmap.max()
        for i in range(x):
            for j in range(y):
                heatmap[i,j]= heatmap[i,j]/max
        
        plt.imsave("monitoring/heatmap.png", heatmap)
        heatmap[heatmap < comparaison_threshold] = 0.0
        heatmap[heatmap >= comparaison_threshold] = 1.0
        return heatmap

def connect_components(heatmap):
    """
    Find connected components in the heatmap and merge overlapping bounding boxes.
    Args:
        heatmap (numpy.ndarray): A heatmap of the detected objects.
    Returns:
    A list of merged bounding boxes.
    """
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
    """
    Create a CSV submission file for the detected bounding boxes.
    
    Args:
        boxes_list (list): A list of detected bounding boxes.
        files_names (list): A list of file names for each bounding box.
    """
    rows = []
    for i, file_name in enumerate(files_names):
        rle = run_length_encoding(bounding_boxes_to_mask(boxes_list[i], 720, 1280))
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
