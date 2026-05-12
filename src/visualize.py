import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

def visualize(model_path, csv_path, val_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(model_path)
    df = pd.read_csv(csv_path)
    
    val_images = os.listdir(val_dir)
    
    # Process just the first 5 images for visualization
    for img_name in val_images[:5]:
        img_path = os.path.join(val_dir, img_name)
        img = cv2.imread(img_path)
        
        # Draw Ground Truth
        gt_data = df[df['image'] == img_name]
        for _, row in gt_data.iterrows():
            cx = int(row['center_x'])
            cy = int(row['center_y'])
            angle_rad = np.deg2rad(row['angle_deg'])
            
            # Tip (Green for GT)
            tx = int(cx + 20 * np.cos(angle_rad))
            ty = int(cy - 20 * np.sin(angle_rad))
            
            # Hinge (Blue for GT)
            hx = int(cx - 20 * np.cos(angle_rad))
            hy = int(cy + 20 * np.sin(angle_rad))
            
            cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1) # Center
            cv2.line(img, (cx, cy), (tx, ty), (0, 255, 0), 2) # Line to tip
            cv2.line(img, (cx, cy), (hx, hy), (255, 0, 0), 2) # Line to hinge
            
        # Draw Predictions
        results = model(img_path, verbose=False)
        result = results[0]
        
        if result.keypoints is not None and result.keypoints.data is not None:
            kpts = result.keypoints.data.cpu().numpy()
            for obj_kpts in kpts:
                # Keypoints (Red for Pred Tip, Cyan for Pred Hinge)
                cx, cy = int(obj_kpts[0][0]), int(obj_kpts[0][1])
                hx, hy = int(obj_kpts[1][0]), int(obj_kpts[1][1])
                tx, ty = int(obj_kpts[2][0]), int(obj_kpts[2][1])
                
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1) # Center
                cv2.line(img, (cx, cy), (tx, ty), (0, 0, 255), 2) # Line to tip
                cv2.line(img, (cx, cy), (hx, hy), (255, 255, 0), 2) # Line to hinge
                
        out_path = os.path.join(out_dir, f"vis_{img_name}")
        cv2.imwrite(out_path, img)
        print(f"Saved visualization to {out_path}")

if __name__ == '__main__':
    model_path = 'runs/pose2/tube_detection/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
    else:
        visualize(model_path, 'annotations.csv', 'yolo_data/images/val', 'visualizations')
