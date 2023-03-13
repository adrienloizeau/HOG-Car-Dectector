#! zsh 
# Install Kaggle
pip install -q kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Downloading datasets
kaggle datasets download -d brsdincer/vehicle-detection-image-set

# Unzipping datasets
pip install unzip
mkdir dataset
mkdir dataset/vehicle-detection-image-set
unzip vehicle-detection-image-set.zip -d dataset
rm -rf vehicle-detection-image-set.zip
