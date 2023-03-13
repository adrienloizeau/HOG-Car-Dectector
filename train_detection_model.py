import argparse
import subprocess
import numpy as np 
import pandas as pd
from Pathlib import Path
from skimage.feature import hog
from torchvision.io import read_image
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle

class config:
    IMG_PATH = "/content/the-car-connection-picture-dataset"
    DATASET = "/dataset/vehicle-detection-image-set"


parser = argparse.ArgumentParser(description='Arguments for the script.')
parser.add_argument('--use_existing_dataset', type=bool, default=False, help='Boolean argument to indicate whether to use an existing dataset or create a new one.')
parser.add_argument('--dataset_path', type=str, default='', help='Path to the existing dataset. This argument is used only if use_existing_dataset is True.')

args = parser.parse_args()

def extract_hog_features(img_path, orientations = 9, pixels_per_cell = (8,8), cells_per_block = (2,2)):
    img = np.transpose(read_image(img_path), (1, 2, 0))
    # Extract HOG features with dummy parameters
    fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, visualize=True, multichannel=True)
    return fd

def create_dataset():
    ### Download the dataset and make the dataframe associated
    subprocess.run(['bash', 'scripts/download_classif_dataset.sh'])

    # List of the non-vehicles files and vehicles files
    non_vehicles_path = Path(config.DATASET + "/non-vehicles")
    p = non_vehicles_path.glob("**/*")
    n_v_files = sorted([x.as_posix() for x in p if x.is_file()])
    vehicles_path = Path(config.FINAL_DATASET + "/vehicles")
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
    for i, row in tqdm(df_dataset_raw.iterrows()):
        img_path = row['path']
        hog_features = extract_hog_features(img_path)

        df_dataset_raw.at[i, 'hog_features'] = hog_features
    
    df_dataset = shuffle(df_dataset_raw, random_state= 42).reset_index(drop=True)
    
    return df_dataset


if args.use_existing_dataset :
    df_dataset = pd.read_csv(args.dataset_path)
else: 
    df_dataset = create_dataset()

X_train, X_test, y_train, y_test = train_test_split(df_dataset['hog_features'],df_dataset['labels'], test_size = 0.3, random_state=42)
X_train,  X_test = np.vstack(X_train),np.vstack(X_test)

# Fitting the model on the training data
clf = RandomForestClassifier()
print("Training classifier...")
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
