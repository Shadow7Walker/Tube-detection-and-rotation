import os
from ultralytics import YOLO

def main():
    # Load a model
    # yolov8n-pose.pt is the nano pose model
    model = YOLO('yolov8n-pose.pt')  

    # Train the model
    # We use a lower number of epochs as it's a small dataset, 
    # but 100 epochs is a good default to ensure convergence.
    # GPU is automatically detected by ultralytics
    results = model.train(
        data='yolo_data/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=8, # small batch size for 70 images
        project='runs/pose',
        name='tube_detection'
    )
    
    print("Training complete. Results saved to runs/pose/tube_detection")

if __name__ == '__main__':
    main()
