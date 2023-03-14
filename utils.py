import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from itertools import zip_longest

# def sliding_window(image, window_size=64, stride=20, padding=0):
#     """
#     Slide a window across an image with a given stride and padding.

#     Args:
#         image: The input image.
#         window_size: The size of the window.
#         stride: The stride to slide the window.
#         padding: The amount of padding to apply to the image.
    
#     Returns:
#         A generator that yields the sliding windows along with their starting and ending coordinates.
#     """
#     # Add padding to the image
#     padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

#     # Loop over the sliding window
#     for y in range(0, padded_image.shape[0] - window_size + 1, stride):
#         for x in range(0, padded_image.shape[1] - window_size + 1, stride):
            # yield padded_image[y:y+window_size, x:x+window_size], (x, y,window_size, window_size)    

def sliding_window(image, window_size=64, stride=20, padding=0):
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


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

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
