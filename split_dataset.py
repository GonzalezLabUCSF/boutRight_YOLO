import os
import random
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
random.seed(42)

# Paths
annotations_path = os.path.join(os.getcwd(), 'annotations')
images_path = os.path.join(os.getcwd(), 'images_for_training')
dataset_path = os.path.join(os.getcwd(), 'dataset')

images_dir = os.path.join(dataset_path, 'images')
labels_dir = os.path.join(dataset_path, 'labels')
train_images_dir = os.path.join(images_dir, 'train')
val_images_dir = os.path.join(images_dir, 'val')
train_labels_dir = os.path.join(labels_dir, 'train')
val_labels_dir = os.path.join(labels_dir, 'val')

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Function to convert XML to YOLO format
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, output_dir):
    in_file = open(xml_file)
    out_file = open(os.path.join(output_dir, os.path.basename(xml_file).replace(".xml", ".txt")), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in ["bout", "call"]:
            cls_id = 0 if cls == "bout" else 1
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# Get list of labeled images
labels = [f for f in os.listdir(annotations_path) if f.endswith('.xml')]
images = [f.replace('.xml', '.png') for f in labels]

# Split data into train and val sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Function to copy and convert files
def copy_and_convert_files(image_list, label_list, image_src_dir, label_src_dir, image_dst_dir, label_dst_dir):
    for image, label in zip(image_list, label_list):
        # Copy image
        shutil.copy(os.path.join(image_src_dir, image), os.path.join(image_dst_dir, image))
        # Convert and copy label
        convert_annotation(os.path.join(label_src_dir, label), label_dst_dir)

# Copy and convert train files
copy_and_convert_files(train_images, train_labels, images_path, annotations_path, train_images_dir, train_labels_dir)

# Copy and convert val files
copy_and_convert_files(val_images, val_labels, images_path, annotations_path, val_images_dir, val_labels_dir)

print("Files copied and converted successfully!")
