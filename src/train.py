import os
from ultralytics import YOLO

def main():
    # Load a model
    # yolov8m-pose.pt is the medium pose model
    model = YOLO('yolov8m-pose.pt')  

    # Train the model
    # We use a lower number of epochs as it's a small dataset, 
    # but 100 epochs is a good default to ensure convergence.
    # GPU is automatically detected by ultralytics
    results = model.train(
        data=os.path.abspath('yolo_data/dataset.yaml'),
        epochs=100,
        imgsz=640,
        batch=8, # small batch size for 70 images
        project=os.path.abspath('runs'),
        name='tube_detection'
    )
    
    print("Training complete. Results saved to runs/tube_detection")

if __name__ == '__main__':
    main()
