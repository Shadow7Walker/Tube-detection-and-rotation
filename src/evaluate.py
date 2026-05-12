import os
import pandas as pd
import numpy as np
from ultralytics import YOLO

def calculate_angle(center, tip):
    """Calculate angle in degrees [0, 360) given center and tip coordinates."""
    dx = tip[0] - center[0]
    dy = tip[1] - center[1]
    
    # Mathematical angle (y decreases upwards in images, so -dy)
    angle_rad = np.arctan2(-dy, dx)
    angle_deg = np.rad2deg(angle_rad)
    
    # Normalize to [0, 360)
    angle_deg = angle_deg % 360
    return angle_deg

def angular_error(a1, a2):
    """Calculate the shortest angular distance between two angles in degrees."""
    diff = abs(a1 - a2) % 360
    return min(diff, 360 - diff)

def evaluate(model_path, csv_path, val_dir, dist_threshold=30.0):
    model = YOLO(model_path)
    df = pd.read_csv(csv_path)
    
    val_images = os.listdir(val_dir)
    
    total_gt = 0
    total_pred = 0
    true_positives = 0
    angle_errors = []
    
    for img_name in val_images:
        img_path = os.path.join(val_dir, img_name)
        
        # Ground truth for this image
        gt_data = df[df['image'] == img_name]
        total_gt += len(gt_data)
        
        # Run inference
        results = model(img_path, verbose=False)
        result = results[0]
        
        predictions = []
        if result.keypoints is not None and result.keypoints.data is not None:
            kpts = result.keypoints.data.cpu().numpy() # [N, 2, 3]
            for obj_kpts in kpts:
                # obj_kpts is [3, 3] -> (x, y, conf) for center, hinge, tip
                center = obj_kpts[0][:2]
                hinge = obj_kpts[1][:2]
                tip = obj_kpts[2][:2]
                pred_angle = calculate_angle(center, tip)
                predictions.append((center, pred_angle))
                
        total_pred += len(predictions)
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred_center, pred_angle in predictions:
            best_dist = float('inf')
            best_gt_idx = -1
            best_gt_angle = None
            
            for idx, row in gt_data.iterrows():
                if idx in matched_gt:
                    continue
                    
                gt_center = np.array([row['center_x'], row['center_y']])
                dist = np.linalg.norm(pred_center - gt_center)
                
                if dist < best_dist and dist < dist_threshold:
                    best_dist = dist
                    best_gt_idx = idx
                    best_gt_angle = row['angle_deg']
                    
            if best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                true_positives += 1
                error = angular_error(pred_angle, best_gt_angle)
                angle_errors.append(error)
                
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    mean_angle_error = np.mean(angle_errors) if angle_errors else 0
    
    print("=== Evaluation Results ===")
    print(f"Total Images Evaluated: {len(val_images)}")
    print(f"Total Ground Truth Tubes: {total_gt}")
    print(f"Total Predicted Tubes: {total_pred}")
    print(f"True Positives: {true_positives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Mean Angular Error: {mean_angle_error:.2f} degrees")
    print("==========================")
    
    return precision, recall, f1, mean_angle_error

if __name__ == '__main__':
    model_path = 'runs/pose/runs/pose2/tube_detection/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
    else:
        evaluate(model_path, 'annotations.csv', 'yolo_data/images/val')
