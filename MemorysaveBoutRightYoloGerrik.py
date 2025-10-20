import os
import psutil
import os
from glob import glob
import shutil
import torch
from PIL import Image
import numpy as np
from scipy.io import wavfile
from scipy import signal
from tqdm import tqdm
import io
from IPython.display import clear_output
import concurrent.futures
import uuid
import pandas as pd
import math
import zipfile
import warnings
import threading
import hashlib


# Get number of logical CPUs
logical_cpus = psutil.cpu_count(logical=True)

# Calculate 80% of CPUs
target_threads = max(1, int(logical_cpus * 0.8))
print(target_threads)

# Set the number of threads for libraries that use OpenMP
os.environ["OMP_NUM_THREADS"] = str(target_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(target_threads)
os.environ["MKL_NUM_THREADS"] = str(target_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(target_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(target_threads)

print(f"Limiting to {target_threads} threads out of {logical_cpus} logical CPUs")










# Initialize a lock object
lock = threading.Lock()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.getcwd(),r'yolov5\runs\train\exp13\weights\best.pt'))
#folder to temporarily save images for YOLO detection
#this is needed to run parallel computing without problems
temp_path = r'C:\Temp'

# Function to filter and generate spectrogram
def filtered_spectrogram(filepath):
    # Length of FFT
    lend = 34
    # Overlap of FFT
    overlap = 33
    # Time length for exponential window of FFT
    ts = 3
    # Low cut frequency in Hz
    lc = 500
    # High cut frequency in Hz
    hc = 20000
    # Color of image settings
    # Contribution of each channel to color
    RGBch = [0.8, 1.5, 1.5]

    # Import the audio data
    fs, data = wavfile.read(filepath)
    # Round length of data and overlap
    lend = round((lend / 1E3) * fs)
    overlap = round((overlap / 1E3) * fs)
    # Next power of two definition
    def nextpow2(x):
        return 1 if x == 0 else 2**math.ceil(math.log2(x))
    # Calculate next power of two
    nfft = nextpow2(lend)

    # Butterworth filter
    def butter_bp(data, lc, hc, fs, order=3):
        nyq = 0.5 * fs
        low = lc / nyq
        high = hc / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        data_filtered = signal.lfilter(b, a, data)
        return data_filtered

    data = butter_bp(data, lc, hc, fs, order=5)
    # Normalize signal
    data = data / max(abs(data))
    # Make windows for spectrogram
    t = np.linspace(-lend / 2 + 1, lend / 2, num=lend)
    sigma = (ts / 1E3) * fs
    w = np.exp(-(t / sigma)**2)
    dw = np.exp(-(t / (2 * sigma))**2)
    # Calculate spectrograms
    [f, t, sx] = signal.spectrogram(data, fs=fs, window=w, noverlap=overlap, nfft=nfft)
    [_, _, sxx] = signal.spectrogram(data, fs=fs, window=dw, noverlap=overlap, nfft=nfft)
    # Average of both spectrograms
    image_array = np.log2(abs(sx) + abs(sxx)) / 2
    # Obtain thresholds for background
    minmax = [np.percentile(image_array, 80), np.percentile(image_array, 99)]
    # Subtract background
    image_array = np.minimum(image_array, minmax[1])
    image_array = np.maximum(image_array, minmax[0])
    # Normalize
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    # Flip spectrogram
    image_array = np.flip(image_array, 0)
    # Convert to color
    sz = (image_array.shape[0] - 1, image_array.shape[1] - 1, 3)
    image_color = np.zeros(sz)
    tmp = image_array
    image_color[:, :, 0] = RGBch[0] * tmp[0:-1, 0:-1]
    tmp = np.diff(image_array, 1, axis=0)
    image_color[:, :, 1] = RGBch[1] * tmp[:, 0:-1]
    tmp = np.diff(image_array, 1, axis=1)
    image_color[:, :, 2] = RGBch[2] * tmp[0:-1, :]
    
    return image_color, fs, len(data)

# Function to generate spectrogram and check for bouts using YOLOv5
def check_for_bouts(wav_file, temp_path):
    temp_img_path = None
    try:
        # Ensure the temp_path directory exists
        os.makedirs(temp_path, exist_ok=True)
        
        # Generate the spectrogram and convert it to an image
        spectrogram, fs, data_length = filtered_spectrogram(wav_file)
        spectrogram_image = (spectrogram * 255).astype(np.uint8)
        pil_image = Image.fromarray(spectrogram_image)
        
        # Generate a unique filename for the temporary spectrogram image in the temp_path directory
        temp_img_filename = f'temp_spectrogram_{uuid.uuid4().hex}.png'
        temp_img_path = os.path.join(temp_path, temp_img_filename)
        pil_image.save(temp_img_path)
        
        # Verify the saved image
        with Image.open(temp_img_path) as img:
            img.verify()
        
        # Use YOLOv5 to detect bouts in the spectrogram image
        results = model(temp_img_path)
        
        # Filter detections to only include bouts (class 0)
        bouts = [bbox for bbox in results.xyxy[0] if bbox[5] == 0]
        
        # Check if any bouts are detected
        return len(bouts) > 0, results.xyxy[0], spectrogram.shape[1], data_length, fs
    except ValueError as e:
        if "File format" in str(e):
            print(f"Error in check_for_bouts for {wav_file}: {e}")
        else:
            raise e  # Re-raise other types of ValueErrors
        return False, [], 0, 0, 0
    except Exception as e:
        print(f"Error in check_for_bouts for {wav_file}: {e}")
        return False, [], 0, 0, 0
    finally:
        # Remove the temporary image file
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)


# Function to process WAV file
def process_wav_file(wav_file, songs_dir, noise_calls_dir, bouts_csv_path, calls_csv_path, scanned_files, scanned_csv_path, processed_files):
    try:
        has_bouts, bboxes, spectrogram_length, data_length, fs = check_for_bouts(wav_file, temp_path)

        results = []
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            entry = {
                'wav_folder_path': os.path.dirname(wav_file),
                'wav_filename': os.path.basename(wav_file),
                'spectrogram_start_time': x1.item(),
                'spectrogram_end_time': x2.item(),
                'wav_file_start_time': (x1.item() / spectrogram_length) * (data_length / fs),
                'wav_file_end_time': (x2.item() / spectrogram_length) * (data_length / fs),
                'confidence': conf.item(),
                'label_class': cls.item(),
                'bbox_x1': x1.item(),
                'bbox_y1': y1.item(),
                'bbox_x2': x2.item(),
                'bbox_y2': y2.item()
            }
            results.append(entry)

        # Append results to CSV files
        if results:
            bouts_results = [entry for entry in results if entry['label_class'] == 0]
            calls_results = [entry for entry in results if entry['label_class'] == 1]

            with lock:
                if bouts_results:
                    append_to_csv(bouts_results, bouts_csv_path)
                if calls_results:
                    append_to_csv(calls_results, calls_csv_path)

        # Move the file based on detection results
        if has_bouts:
            shutil.move(wav_file, os.path.join(songs_dir, os.path.basename(wav_file)))
            with lock:
                processed_files.add(os.path.basename(wav_file))  # Thread-safe addition
            print(f"Moved {wav_file} to Songs")
        else:
            shutil.move(wav_file, os.path.join(noise_calls_dir, os.path.basename(wav_file)))
            print(f"Moved {wav_file} to Noise_Calls")

        # Update scanned files and save to CSV immediately
        with lock:
            scanned_files.append(os.path.basename(wav_file))
            pd.DataFrame({'filename': scanned_files}).to_csv(scanned_csv_path, index=False)

        return {
            'filepath': wav_file,
            'filename': os.path.basename(wav_file),
            'bboxes': bboxes,
            'spectrogram_length': spectrogram_length,
            'data_length': data_length,
            'fs': fs
        }
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        return None

# Function to re-process WAV file that was missed in the original scan
def reprocess_wav_file(wav_file, noise_calls_dir, bouts_csv_path, calls_csv_path):
    try:
        has_bouts, bboxes, spectrogram_length, data_length, fs = check_for_bouts(wav_file, temp_path)

        results = []
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls = bbox
            entry = {
                'wav_folder_path': os.path.dirname(wav_file),
                'wav_filename': os.path.basename(wav_file),
                'spectrogram_start_time': x1.item(),
                'spectrogram_end_time': x2.item(),
                'wav_file_start_time': (x1.item() / spectrogram_length) * (data_length / fs),
                'wav_file_end_time': (x2.item() / spectrogram_length) * (data_length / fs),
                'confidence': conf.item(),
                'label_class': cls.item(),
                'bbox_x1': x1.item(),
                'bbox_y1': y1.item(),
                'bbox_x2': x2.item(),
                'bbox_y2': y2.item()
            }
            results.append(entry)

        # Append results to CSV files
        if results:
            bouts_results = [entry for entry in results if entry['label_class'] == 0]
            calls_results = [entry for entry in results if entry['label_class'] == 1]

            with lock:
                if bouts_results:
                    append_to_csv(bouts_results, bouts_csv_path)
                if calls_results:
                    append_to_csv(calls_results, calls_csv_path)

        if has_bouts:
            print(f"Reprocessed and found bouts in {wav_file}")
        else:
            print(f"Reprocessed {wav_file}, no bouts found")
            
            # Add this block to move files without bouts to the Noise_Calls folder
            shutil.move(wav_file, os.path.join(noise_calls_dir, os.path.basename(wav_file)))
            print(f"Moved {wav_file} to Noise_Calls after re-scan")

    except Exception as e:
        print(f"Error reprocessing {wav_file}: {e}")


# Function to append results to CSV
def append_to_csv(results, csv_path):
    # Define the expected columns
    columns = [
        'wav_folder_path', 'wav_filename', 'spectrogram_start_time', 'spectrogram_end_time',
        'wav_file_start_time', 'wav_file_end_time', 'confidence', 'label_class',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
    ]
    
    # Convert results to DataFrame
    new_df = pd.DataFrame(results, columns=columns)
    
    # Append to the CSV file
    if os.path.exists(csv_path):
        new_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        new_df.to_csv(csv_path, mode='w', header=True, index=False)

# Function to hash individual files to avoid duplicates
def hash_file(filepath):
    """Returns the MD5 hash of the file content."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to gather filename of wav files in folders and zip files
def get_all_wav_files(hatch_path, bird_path):
    wav_files = {}  # Use a dictionary to avoid duplicates by file content hash
    songs_dir = os.path.join(hatch_path, 'Songs')
    noise_calls_dir = os.path.join(hatch_path, 'Noise_Calls')

    # Extract ZIP files recursively
    zip_files = glob(os.path.join(hatch_path, '**', '*.zip'), recursive=True)
    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # Check if it's a .wav file
                    if file_info.filename.endswith('.wav'):
                        try:
                            # Test individual file for corruption
                            zip_ref.extract(file_info, hatch_path)
                        except Exception as e:
                            print(f"Skipping corrupted file {file_info.filename} in {zip_file}: {e}")
                            continue  # Skip corrupted file and continue with the next one
        except zipfile.BadZipFile:
            print(f"Skipping corrupted ZIP file: {zip_file}")
            continue  # Skip the corrupted ZIP and move on

        # Move the ZIP file to the birdname folder (at the same level as the mic folder)
        shutil.move(zip_file, os.path.join(bird_path, os.path.basename(zip_file)))

    # Function to add files based on their content hash
    def add_wav_files(dir_path):
        for wav_file in glob(os.path.join(dir_path, '**', '*.wav'), recursive=True):
            file_hash = hash_file(wav_file)
            if file_hash not in wav_files:
                wav_files[file_hash] = wav_file
    
    # Add WAV files from the main directory, Songs, and Noise_Calls
    add_wav_files(hatch_path)
    if os.path.exists(songs_dir):
        add_wav_files(songs_dir)
    if os.path.exists(noise_calls_dir):
        add_wav_files(noise_calls_dir)
    
    return list(wav_files.values())  # Return only the unique file paths

# Function to check if a file has already been scanned
def is_already_scanned(wav_file, scanned_files):
    return os.path.basename(wav_file) in scanned_files

# Function to check if a song file is in the bouts CSV
def is_in_bouts_csv(wav_file, bouts_csv_path):
    if os.path.exists(bouts_csv_path):
        try:
            bouts_df = pd.read_csv(bouts_csv_path)
            return os.path.basename(wav_file) in bouts_df['wav_filename'].values
        except pd.errors.EmptyDataError:
            return False
    return False
