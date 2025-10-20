import os
import random
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.io import wavfile
import numpy as np
import math
import os
from glob import glob
import numpy as np
from PIL import Image
import time
from IPython.display import clear_output

# Directory paths
wav_files = glob(os.path.join(os.getcwd(), 'songs_for_training', '*.wav'))
images_path = os.path.join(os.getcwd(), 'images_for_training')

# Create images_for_training directory if it doesn't exist
if not os.path.exists(images_path):
    os.makedirs(images_path)

total_time = 0
num_files_processed = 0

# Function to process each WAV file
def process_wav_file(wav_file,images_path):
    global total_time, num_files_processed
    start_time = time.time()
    
    # Generate the expected output .png file name
    img_name = os.path.basename(wav_file).replace('.wav', '.png')
    img_path = os.path.join(images_path, img_name)

    # Skip processing if the image already exists
    if os.path.exists(img_path):
        clear_output(wait=True)
        print(f"Image already exists for {wav_file}, skipping...")
        return

    try:
        # Generate the spectrogram from the wav file using the original method
        spectrogram, fs, data_length = filtered_spectrogram(wav_file)
        
        # Convert the spectrogram to an image and save it
        spectrogram_image = (spectrogram * 255).astype(np.uint8)
        pil_image = Image.fromarray(spectrogram_image)
        pil_image.save(img_path)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        num_files_processed += 1
        avg_time_per_file = total_time / num_files_processed
        
        # Clear the output and print the timing information
        clear_output(wait=True)
        print(f"Saved image {img_path} in {elapsed_time:.2f} seconds.")
        print(f"Average time per file: {avg_time_per_file:.2f} seconds.")
        
        # Close the image file to free up resources
        pil_image.close()
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")

def filtered_spectrogram(wav_file):
    #length of fft
    lend = 34
    #overlap of fft
    overlap =33
    #time length for exponential window of fft
    ts = 3
    #low cut frequency in Hz
    lc = 500
    #high cut frequency in Hz
    hc = 20000
    # color of image settings
    # contribution of each channel to color
    RGBch = [0.8,1.5,1.5]

    #import the audio data
    fs, data = wavfile.read(wav_file)
    #round length of data and overlap
    lend = round((lend/1E3)*fs)
    overlap =round((overlap/1E3)*fs)
    #next power of two definition
    def nextpow2(x):
        return 1 if x == 0 else 2**math.ceil(math.log2(x))
    #calculate next power of two
    nfft = nextpow2(lend)

    # Butterworth filter
    def butter_bp(data, lc, hc, fs, order=3):
       nyq = 0.5*fs
       low = lc/nyq
       high = hc/nyq
       b,a = signal.butter(order,[low, high], btype='band')
       data_filtered = signal.lfilter(b, a, data)
       return data_filtered

    data = butter_bp(data, lc, hc, fs, order=5)
    #normalize signal
    data = data/max(abs(data))
    # make windows for spectrogram
    t = np.linspace(-lend/2+1,lend/2,num=lend)
    sigma = (ts/1E3)*fs
    w = np.exp(-(t/sigma)**2)
    #dw = -2*w*(t/(sigma**2))
    dw = np.exp(-(t/(2*sigma))**2)
    #Calculate spectrograms
    [f,t,sx] = signal.spectrogram(data,fs=fs, window=w, noverlap=overlap, nfft=nfft )
    [_,_,sxx] = signal.spectrogram(data,fs=fs, window=dw, noverlap=overlap, nfft=nfft)
    #average of both spectrograms
    image_array = np.log2(abs(sx)+abs(sx))/2
    #obtain tresholds for background
    minmax = [np.percentile(image_array ,80),np.percentile(image_array,99)]
    #subtract backgroud
    image_array = np.minimum(image_array,minmax[1])
    image_array = np.maximum(image_array,minmax[0])
    #normalize
    image_array = (image_array-np.min(image_array))/(np.max(image_array)-np.min(image_array))
    #flip spectrogram
    image_array = np.flip(image_array,0)
    #convert to color
    sz = (image_array.shape[0]-1,image_array.shape[1]-1,3)
    image_color = np.zeros(sz)
    tmp = image_array
    image_color[:,:,0] = RGBch[0]*tmp[0:-1,0:-1]
    tmp = np.diff(image_array,1,axis=0)
    image_color[:,:,1] = RGBch[1]*tmp[:,0:-1]
    tmp = np.diff(image_array,1,axis=1)
    image_color[:,:,2] = RGBch[2]*tmp[0:-1,:]
    return image_color, fs, len(data)

def split_dataset(annotations_path, images_path, dataset_path, test_size=0.2, random_seed=42):
    """
    Prepare the dataset by splitting images and annotations into training and validation sets,
    converting annotations to YOLO format, and copying files to the appropriate directories.

    Parameters:
    annotations_path (str): Path to the annotations directory.
    images_path (str): Path to the images directory.
    dataset_path (str): Path to the dataset directory.
    test_size (float): Proportion of the dataset to include in the validation split.
    random_seed (int): Seed for random number generator for reproducibility.
    """
    # Set the seed for reproducibility
    random.seed(random_seed)

    # Paths
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
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=test_size, random_state=random_seed)

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
