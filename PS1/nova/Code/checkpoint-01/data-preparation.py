import os
import shutil


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"Directory already exists at {path}")
    return path

directory_path = create_dir("/content/dataset")

source_dir = "/content/drive/MyDrive/DATASET/Images"
destination_dir = "/content/dataset/Images"

os.makedirs(destination_dir, exist_ok=True)

for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    if os.path.isdir(folder_path):
        #print(f"Processing folder: {folder_name}")
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            #print(f"File found: {file_name}")
            if os.path.isfile(file_path) and file_name.endswith(('_leftImg8bit.png')):
                try:
                    shutil.copy(file_path, destination_dir)
                    #print(f"Successfully copied {file_name} from {folder_path} to {destination_dir}")
                except Exception as e:
                    print(f"Error copying {file_name}: {e}")

# Source directories
image_source_dir = "/content/drive/MyDrive/DATASET/Images"
label_source_dir = "/content/drive/MyDrive/DATASET/Labels"

# Destination directories
destination_dir = "/content/dataset"

# Create destination directories if they don't exist
os.makedirs(os.path.join(destination_dir, "Images"), exist_ok=True)
os.makedirs(os.path.join(destination_dir, "Labels"), exist_ok=True)

# Get the list of folders sorted alphabetically in the image source directory
image_folder_names = sorted(os.listdir(image_source_dir))

# Iterate through each folder in the image source directory
for folder_name in image_folder_names:
    image_folder_path = os.path.join(image_source_dir, folder_name)
    label_folder_path = os.path.join(label_source_dir, folder_name)
    
    # Check if the item in the image source directory is a folder
    if os.path.isdir(image_folder_path):
        # Iterate through each file in the image folder
        for image_file_name in os.listdir(image_folder_path):
            image_file_path = os.path.join(image_folder_path, image_file_name)
            # Check if the item is a file and ends with '_leftImg8bit.png'
            if os.path.isfile(image_file_path) and image_file_name.endswith(('_leftImg8bit.png')):
                # Check if the corresponding label file exists
                corresponding_label_file_name = image_file_name.replace('_leftImg8bit.png', '_gtFine_color.png')
                corresponding_label_file_path = os.path.join(label_folder_path, corresponding_label_file_name)
                if os.path.isfile(corresponding_label_file_path):
                    try:
                        # Copy the image file to the destination image directory
                        shutil.copy(image_file_path, os.path.join(destination_dir, "Images"))
                        # Copy the label file to the destination label directory
                        shutil.copy(corresponding_label_file_path, os.path.join(destination_dir, "Labels"))
                        print(f"Successfully copied {image_file_name} and its corresponding label from {image_folder_path} and {label_folder_path} to {destination_dir}")
                    except Exception as e:
                        print(f"Error copying {image_file_name} and its corresponding label: {e}")


!pip install split-folders
import splitfolders

# Path to the directory containing 'Images' and 'Labels' folders
source_dir = '/content/dataset'

# Path to the directory where you want to save the split data
output_dir = '/content/split_data'

# Split the data into train (80%) and test (20%) sets
splitfolders.fixed(source_dir, output=output_dir, seed=42, fixed=(80, 20), group_prefix=None)