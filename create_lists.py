import os
import random
from pathlib import Path

def create_oil_spill_lists(base_dir, list_dir, train_ratio=0.8, val_from_train=True):
    """
    Create train.txt and test_vol.txt for oil spill dataset
    """
    
    # Create output directory
    os.makedirs(list_dir, exist_ok=True)
    
    # Collect training samples
    train_samples = []
    sensors = ['palsar', 'sentinel']
    
    print("Collecting training samples...")
    for sensor in sensors:
        image_dir = os.path.join(base_dir, 'train', sensor, 'image')
        label_dir = os.path.join(base_dir, 'train', sensor, 'label')
        
        if os.path.exists(image_dir) and os.path.exists(label_dir):
            image_files = set([f.split('.')[0] for f in os.listdir(image_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            label_files = set([f.split('.')[0] for f in os.listdir(label_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            
            # Only include samples that have both image and label
            common_files = image_files.intersection(label_files)
            
            for file_name in common_files:
                sample_name = f"{sensor}/{file_name}"
                train_samples.append(sample_name)
            
            print(f"Found {len(common_files)} {sensor} training samples")
    
    # Collect test samples
    test_samples = []
    print("Collecting test samples...")
    for sensor in sensors:
        image_dir = os.path.join(base_dir, 'test', sensor, 'image')
        label_dir = os.path.join(base_dir, 'test', sensor, 'label')
        
        if os.path.exists(image_dir) and os.path.exists(label_dir):
            image_files = set([f.split('.')[0] for f in os.listdir(image_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            label_files = set([f.split('.')[0] for f in os.listdir(label_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            
            # Only include samples that have both image and label
            common_files = image_files.intersection(label_files)
            
            for file_name in common_files:
                sample_name = f"{sensor}/{file_name}"
                test_samples.append(sample_name)
            
            print(f"Found {len(common_files)} {sensor} test samples")
    
    # Shuffle samples
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    
    # Create train/validation split if requested
    if val_from_train and train_ratio < 1.0:
        train_count = int(len(train_samples) * train_ratio)
        actual_train = train_samples[:train_count]
        val_samples = train_samples[train_count:]
        
        print(f"Split training data: {len(actual_train)} train, {len(val_samples)} validation")
        
        # Write validation list
        with open(os.path.join(list_dir, 'val.txt'), 'w') as f:
            for sample in val_samples:
                f.write(sample + '\n')
    else:
        actual_train = train_samples
    
    # Write train.txt
    with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
        for sample in actual_train:
            f.write(sample + '\n')
    
    # Write test_vol.txt
    with open(os.path.join(list_dir, 'test_vol.txt'), 'w') as f:
        for sample in test_samples:
            f.write(sample + '\n')
    
    print(f"\nCreated list files:")
    print(f"  train.txt: {len(actual_train)} samples")
    print(f"  test_vol.txt: {len(test_samples)} samples")
    if val_from_train and train_ratio < 1.0:
        print(f"  val.txt: {len(val_samples)} samples")
    
    return len(actual_train), len(test_samples)

if __name__ == "__main__":
    # UPDATE THIS PATH TO YOUR ACTUAL DATASET LOCATION
    base_dir = "../dataset" 
    list_dir = "./lists/lists_OilSpill"
    
    # Create the list files
    train_count, test_count = create_oil_spill_lists(base_dir, list_dir, train_ratio=0.8)
    
    print(f"\nList creation completed successfully!")
    print(f"Use these commands to train and test:")
    print(f"Training: python train.py --dataset OilSpill --vit_name R50-ViT-B_16")
    print(f"Testing: python test.py --dataset OilSpill --vit_name R50-ViT-B_16")
