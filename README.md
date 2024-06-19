# Football Analysis Project

## Introduction

This project aims to detect and track players, referees, and footballs in a video using YOLO, a state-of-the-art object detection model. Additionally, it assigns players to teams based on their jersey colors using K-means clustering, measures team ball control, estimates camera movement using optical flow, and calculates player speed and distance covered.

## Features

- Object detection and tracking using YOLO
- Team assignment based on jersey colors using K-means clustering
- Calculation of team ball control percentage
- Estimation of camera movement between frames using optical flow
- Perspective transformation to represent scene depth and measure player movement in meters
- Calculation of player speed and distance covered

## Modules Used

- **YOLO**: AI object detection model
- **K-means**: Pixel segmentation and clustering for jersey color detection
- **Optical Flow**: Measure camera movement between frames
- **Perspective Transformation**: Represent scene depth and perspective
- **Speed and Distance Calculation**: Calculate player speed and distance covered

## Trained Models

- Trained YOLO v5 model

## Sample Video

A sample input video is provided for testing purposes: `input_videos/08fd33_4.mp4`

## Requirements

To run this project, you need to have the following requirements installed:

- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Contributing

Contributions are welcome! If you find any issues or have ideas for improvements, please open an issue or submit a pull request.
