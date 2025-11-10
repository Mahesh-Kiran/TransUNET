import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
import cv2

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Data augmentation
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # Resize if needed
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # Handle different image dimensions
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # Add channel dimension for grayscale
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)  # HWC to CHW
        
        # Ensure image is float32 and normalized
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        
        image = torch.from_numpy(image)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample

class OilSpill_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, sensor_type='both'):
        self.transform = transform
        self.split = split
        self.sensor_type = sensor_type
        self.base_dir = base_dir
        
        # Read the file list
        self.sample_list = []
        
        # Handle different split names
        if split == "test_vol":
            list_file = os.path.join(list_dir, 'test_vol.txt')
        else:
            list_file = os.path.join(list_dir, split + '.txt')
        
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                self.sample_list = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(self.sample_list)} samples from {list_file}")
        else:
            print(f"List file {list_file} not found. Creating sample list from directory structure...")
            self._create_sample_list()
    
    def _create_sample_list(self):
        """Create sample list from directory structure"""
        sensors = ['palsar', 'sentinel'] if self.sensor_type == 'both' else [self.sensor_type]
        
        # Determine the correct directory based on split
        if self.split == 'train':
            split_dir = 'train'
        else:  # test or test_vol
            split_dir = 'test'
        
        for sensor in sensors:
            image_dir = os.path.join(self.base_dir, split_dir, sensor, 'image')
            if os.path.exists(image_dir):
                for img_name in os.listdir(image_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        sample_name = f"{sensor}/{img_name.split('.')[0]}"
                        self.sample_list.append(sample_name)
        
        print(f"Created sample list with {len(self.sample_list)} samples")
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]
        
        # Parse sensor and filename
        if '/' in sample_name:
            sensor, file_name = sample_name.split('/', 1)
        else:
            # Fallback: assume first available sensor
            sensor = 'palsar' if os.path.exists(os.path.join(self.base_dir, 'train', 'palsar')) else 'sentinel'
            file_name = sample_name
        
        # Determine split directory - CORRECTED LOGIC
        if self.split == 'train':
            split_dir = ''  # Training data is directly in base_dir
        else:  # test or test_vol
            split_dir = ''  # Test data is also directly in base_dir
        
        # Find image file with correct extension
        image_path = None
        label_path = None
        
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            if self.split == 'train':
                img_candidate = os.path.join(self.base_dir, 'train', sensor, 'image', file_name + ext)
                lbl_candidate = os.path.join(self.base_dir, 'train', sensor, 'label', file_name + ext)
            else:  # test or test_vol
                img_candidate = os.path.join(self.base_dir, 'test', sensor, 'image', file_name + ext)
                lbl_candidate = os.path.join(self.base_dir, 'test', sensor, 'label', file_name + ext)
            
            if os.path.exists(img_candidate) and os.path.exists(lbl_candidate):
                image_path = img_candidate
                label_path = lbl_candidate
                break
        
        if image_path is None or label_path is None:
            raise FileNotFoundError(f"Could not find image or label for {sample_name}")
        
        # Load image
        if image_path.lower().endswith(('.tif', '.tiff')):
            try:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    image = np.array(Image.open(image_path))
            except:
                image = np.array(Image.open(image_path))
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Load label
        if label_path.lower().endswith(('.tif', '.tiff')):
            try:
                label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                if label is None:
                    label = np.array(Image.open(label_path))
            except:
                label = np.array(Image.open(label_path))
        else:
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # Handle multi-channel images by converting to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if len(label.shape) == 3:
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        
        # Ensure proper data types
        image = image.astype(np.float32)
        label = label.astype(np.uint8)
        label = (label > 127).astype(np.uint8)
        
        # Normalize image to [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        #commenting this for testing if it works
        # Ensure label values are in correct range
        # unique_labels = np.unique(label)
        # if label.max() > 2:
        #     # Map label values to 0, 1, 2 range
        #     label = label // (255 // max(2, len(unique_labels) - 1))
        
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        sample['case_name'] = sample_name
        return sample
