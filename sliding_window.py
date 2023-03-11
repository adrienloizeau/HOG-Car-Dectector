import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

def sliding_window(image, window_size, stride, padding):
    """
    Slide a window across an image with a given stride and padding.

    Args:
        image: The input image.
        window_size: The size of the window.
        stride: The stride to slide the window.
        padding: The amount of padding to apply to the image.

    Returns:
        A generator that yields the sliding windows.
    """
    # Add padding to the image
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

    # Loop over the sliding window
    for y in range(0, padded_image.shape[0] - window_size + 1, stride):
        for x in range(0, padded_image.shape[1] - window_size + 1, stride):
            yield padded_image[y:y+window_size, x:x+window_size]


parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='train/A_001.jpg', help='Path of the image')
parser.add_argument('--window_size', type=int, default= 64, help='Size of the window')
parser.add_argument('--stride', type=int, default=10, help='Size of the stride')
parser.add_argument('--padding', type=int, default=0, help='Size of the padding')
args = parser.parse_args()


if __name__ == '__main__':
    image = cv2.imread(args.image)
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
