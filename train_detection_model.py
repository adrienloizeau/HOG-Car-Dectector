import argparse
import subprocess
import numpy as np 
import pandas as pd
from pathlib import Path
from skimage.feature import hog
from torchvision.io import read_image
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.transform import rotate, warp, ProjectiveTransform
from skimage.color import rgb2gray
import random
import pickle


parser = argparse.ArgumentParser(description='Arguments for the script.')
parser.add_argument('--dataset_path', type=str, default="/content/classification-dataset/data", help='Path for the dataset')
parser.add_argument('--use_existing_dataset', type=bool, default=False, help='Boolean argument to indicate whether to use an existing dataset or create a new one.')
parser.add_argument('--csv_dataset_path', type=str, default='', help='Path to the existing dataset. This argument is used only if use_existing_dataset is True.')

args = parser.parse_args()

def extract_hog_features(img_path, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    # Load the image and convert to grayscale
    img = cv2.imread(img_path)
    
    # Apply random affine transformations to the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img1 = img[:,:,0]
    img2 = img[:,:,1]
    img3 = img[:,:,2]
    
    # Extract HOG features from the transformed images
    fd1, hog_image1 = hog(img1, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=True, channel_axis=False)
    fd2, hog_image2 = hog(img2, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=True, channel_axis=False)
    fd3, hog_image3 = hog(img3, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=True, channel_axis=False)
    
    # Normalize the HOG features
    scaler = StandardScaler()
    hog_image1 = scaler.fit_transform(fd1)
    hog_image1 = np.ravel(hog_image1)

    hog_image2 = scaler.fit_transform(fd2)
    hog_image2 = np.ravel(hog_image2)
    
    hog_image3 = scaler.fit_transform(fd3)
    hog_image3 =np.ravel(hog_image3)

    contenated_hogs= np.concatenate((hog_image1,hog_image2,hog_image3),axis= 0)
    return contenated_hogs

def apply_transformations(img):
    # Apply random affine transformations to the image
    angles = np.arange(-10, 10, 1)
    scales = np.arange(0.9, 1.1, 0.01)
    transforms = []
    for angle in angles:
        for scale in scales:
            tform = ProjectiveTransform(scale=scale, rotation=np.deg2rad(angle))
            transforms.append(tform)
    random.shuffle(transforms)
    img_transformed = [warp(img, transforms[i]) for i in range(10)]
    return img_transformed

def create_dataset():
  # Download the dataset if not done
  dataset_path = Path(DATASET)
  if not dataset_path.exists():
    # Download the dataset and make the dataframe associated
    subprocess.run(['bash', 'car_dectector/scripts/download_classif_dataset.sh'])

  # List of the non-vehicles files and vehicles files
  non_vehicles_path = Path(DATASET + "/non-vehicles")
  p = non_vehicles_path.glob("**/*")
  n_v_files = sorted([x.as_posix() for x in p if x.is_file()])
  vehicles_path = Path(DATASET + "/vehicles")
  p = vehicles_path.glob("**/*")
  v_files = sorted([x.as_posix() for x in p if x.is_file()])

  # Creating the files list
  files = n_v_files + v_files

  # Creating the labels list 
  n_v_label = np.zeros(len(n_v_files))
  v_label = np.ones(len(v_files))
  labels = np.concatenate((n_v_label, v_label),axis = 0)

  df_dataset_raw = pd.DataFrame(data= {"path":files,"labels":labels})

  # Create a new column in your DataFrame to store the HOG features
  df_dataset_raw['hog_features'] = None

  # Loop over each row in the DataFrame and extract HOG features for the corresponding image path
  print("Extracting the HOG features of the dataset")
  for i, row in tqdm(df_dataset_raw.iterrows(), total=len(df_dataset_raw)):
    print(i)
    img_path = row['path']
    contenated_hogs = extract_hog_features(img_path)

    df_dataset_raw.at[i, 'hog_features'] = contenated_hogs

  df_dataset = shuffle(df_dataset_raw, random_state= 42).reset_index(drop=True)
  return df_dataset

if __name__ == "__main__":
  DATASET = args.dataset_path
  if args.use_existing_dataset :
      df_dataset = pd.read_csv(args.csv_dataset_path)
  else: 
      df_dataset = create_dataset()

  print(df_dataset.head())
  # For each of the HOG features
  print("#"*50)
  X_train, X_test, y_train, y_test = train_test_split(df_dataset['hog_features'],df_dataset['labels'], test_size = 0.3, random_state=42)
  X_train,  X_test = np.vstack(X_train),np.vstack(X_test)

  # Fitting the model on the training data
  clf = RandomForestClassifier()
  print("Training classifier" + "...")
  clf.fit(X_train, y_train) 
  print("Done")

  # Predicting the value of the test dataset
  y_pred = clf.predict(X_test)

  # Getting the metrics
  acc= accuracy_score(y_test, y_pred)

  # Showing the results
  print(f"Classifier's accuracy :{acc}")

  answer = input("Do you want to save the model ? (y/n)").lower()
  if bool(answer and answer != 'n'):
      name = input("name the model").lower()
      pickle.dump(clf, open(name+".model", "wb"))

  print(name , "saved")
  print("**FINISHED**")
