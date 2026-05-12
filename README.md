# Tube Detection Take-Home Assignment

## Problem

You are given a dataset of **70 overhead RGB images** (640x480 pixels) of microcentrifuge tubes on various surfaces. Each image contains between 3 and 6 tubes.

**Ground truth annotations** are provided for all images: the center position and rotation angle of each tube lid.

Your job:

1. Build a system that detects tube positions and orientations in the images.
2. Evaluate your system's performance against the ground truth.
3. Report and analyze your results.

---

## Data

Everything you need is in the `data/` folder.

### Images

`data/annotated_images/` contains **70 PNG images** (640x480, RGB).

Each image is an overhead photo of a surface with microcentrifuge tubes. The number of tubes per image varies (3-6). Backgrounds include desks, white surfaces, black surfaces, and mixed-color surfaces.

### Annotations

`data/annotations.csv` contains ground truth for all 70 images (371 total tubes).

| Column      | Type   | Description                                                |
|-------------|--------|------------------------------------------------------------|
| `image`     | string | Image filename (e.g. `2659ffa5-color.png`)                  |
| `center_x`  | float  | Tube lid center, x-coordinate in pixels                    |
| `center_y`  | float  | Tube lid center, y-coordinate in pixels                    |
| `bbox_x`    | float  | Bounding box top-left x                                    |
| `bbox_y`    | float  | Bounding box top-left y                                    |
| `bbox_w`    | float  | Bounding box width                                         |
| `bbox_h`    | float  | Bounding box height                                        |
| `bbox_rotation` | float | Bounding box rotation in degrees (clockwise)           |
| `angle_deg` | float  | Tube lid rotation angle in degrees, range [0, 360), defined by joint-to-tab direction |

### Coordinate System

- Origin is the **top-left** corner of the image.
- X increases rightward, Y increases downward.
- Angle 0 degrees points along the **positive X-axis** (rightward).
- Angles increase **counter-clockwise**.
- Rotation angle of the tube is defined by the direction of the joint to the tab.

---

## Submission

Submit your work through the **Google Form** (link provided separately).

You will need to provide:

- Public GitHub link
- Description of your approach
- Your reported metrics (precision, recall, F1, angle error)
- Your written analysis and next-steps proposal
- How you used AI

---

## Rules

- Any programming language. Any libraries.
- AI coding tools are allowed.
- Your written analysis should be your own thinking.
- Work independently.

---

## How to Run the System

### 1. Setup Environment
This project uses `uv` for fast dependency management.
#### Note: if you already have uv installed and downloaded the entire git repository, you can skip to step 2, the environment will be set up automatically during the first run.
To set up the environment and install the required libraries, run:
```bash
uv init
uv add ultralytics opencv-python pandas scikit-learn matplotlib
```
*(Alternatively, you can use `pip install ultralytics opencv-python pandas scikit-learn matplotlib` in your standard Python environment).*

### 2. Prepare the Data
The data preparation script parses `annotations.csv`, generates the 3-keypoint topology (Center, Hinge, Tab), and splits the images into train/validation sets for YOLO.
```bash
uv run python src/data_prep.py
```

### 3. Train the Model
This script fine-tunes the `yolov8m-pose.pt` model on the generated dataset.
```bash
uv run python src/train.py
```

### 4. Evaluate Performance
This script calculates Precision, Recall, F1-Score, and Mean Angular Error on the validation set.
```bash
uv run python src/evaluate.py
```

### 5. Generate Visualizations
This script generates sample images with Ground Truth vectors (Green) and Predicted vectors (Red for Tab, Blue for Hinge).
```bash
uv run python src/visualize.py
```
*The resulting images will be saved in the `visualizations/` directory.*
