import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_yolo_data(csv_path, img_dir, out_dir, img_width=640, img_height=480, test_size=0.2):
    df = pd.read_csv(csv_path)
    
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)
        
    images = df['image'].unique()
    train_imgs, val_imgs = train_test_split(images, test_size=test_size, random_state=42)
    
    def process_split(split_imgs, split_name):
        for img_name in split_imgs:
            img_data = df[df['image'] == img_name]
            
            # Copy image
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(out_dir, 'images', split_name, img_name)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"Warning: Image {src_img} not found.")
                continue
                
            # Write label
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(out_dir, 'labels', split_name, label_name)
            
            with open(label_path, 'w') as f:
                for _, row in img_data.iterrows():
                    cx = row['center_x']
                    cy = row['center_y']
                    
                    # Ensure bbox is a reasonable size to enclose the tube, rotated or not
                    w = max(row['bbox_w'], row['bbox_h']) * 1.2
                    h = w
                    
                    # Normalized bbox
                    norm_cx = cx / img_width
                    norm_cy = cy / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height
                    
                    # Keypoint 1: Center
                    kp1_x = cx / img_width
                    kp1_y = cy / img_height
                    kp1_v = 2 # visible
                    
                    angle_rad = np.deg2rad(row['angle_deg'])
                    
                    # Keypoint 2: Hinge (opposite to tip)
                    kp2_x_px = cx - 20 * np.cos(angle_rad)
                    kp2_y_px = cy + 20 * np.sin(angle_rad)
                    kp2_x = kp2_x_px / img_width
                    kp2_y = kp2_y_px / img_height
                    kp2_v = 2
                    
                    # Keypoint 3: Tip/Tab (approx 20 pixels away from center)
                    kp3_x_px = cx + 20 * np.cos(angle_rad)
                    kp3_y_px = cy - 20 * np.sin(angle_rad)
                    kp3_x = kp3_x_px / img_width
                    kp3_y = kp3_y_px / img_height
                    kp3_v = 2
                    
                    # class_id norm_cx norm_cy norm_w norm_h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v kp3_x kp3_y kp3_v
                    line = f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {kp1_x:.6f} {kp1_y:.6f} {kp1_v} {kp2_x:.6f} {kp2_y:.6f} {kp2_v} {kp3_x:.6f} {kp3_y:.6f} {kp3_v}\n"
                    f.write(line)

    process_split(train_imgs, 'train')
    process_split(val_imgs, 'val')
    
    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath(out_dir)}
train: images/train
val: images/val

names:
  0: tube

kpt_shape: [3, 3] # number of keypoints, number of dims (x, y, visible)
"""
    with open(os.path.join(out_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)
        
    print(f"Dataset prepared successfully at {out_dir}")

if __name__ == "__main__":
    os.makedirs('src', exist_ok=True)
    prepare_yolo_data('annotations.csv', 'images', 'yolo_data')
