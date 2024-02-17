# Parking Spot Detection and Occupancy Monitoring

## Description

This project implements a real-time system for detecting parking spaces and identifying vehicle occupancy in a video stream using computer vision and machine learning. It leverages efficient techniques like mask overlap and connected components analysis to accurately localize parking spots, and employs an XGBoost classifier for robust car detection. Mathematical and time delay optimizations ensure suitability for real-time applications.

## Key Features

- Efficient Parking Spot Detection: Employs mask overlap and connected components for precise localization. Adaptable to various parking lot layouts and lighting conditions.
- Robust Car Detection: XGBoost model trained on diverse car datasets for high accuracy. Handles challenging scenarios like overlapping vehicles or partial occlusions.
- Real-Time Performance: Mathematical and time delay optimizations for efficient inference. Scalable to handle larger video streams and multiple cameras.
- Customizable Output: Visualizes results on the video stream, highlighting occupied and free spaces. Provides real-time parking occupancy count.

## Dependencies

- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn
- XGBoost
- (Optional) Additional libraries for visualization (e.g., Matplotlib)

## Installation

1. Clone the repository:

```bash
git clone [https://github.com/your-username/parking-spot-detection.git](https://github.com/your-username/parking-spot-detection.git)
