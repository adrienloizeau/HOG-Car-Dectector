#! zsh 
# Install Kaggle
pip install -q kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Downloading datasets
kaggle competitions download -c vic-kaggle-challenge-2023

# Unzipping datasets
pip install unzip
unzip vic-kaggle-challenge-2023
rm -rf vic-kaggle-challenge-2023

# Copy test files in a test file 
mkdir test
cp Assignment2/test/test test
