## YOLOv3 Object Detection for COCO Dataset on Videos

This is a Python implementation of object detection using the YOLOv3 algorithm on videos with the COCO dataset. The COCO dataset contains images with more than 80 different object categories such as person, car, bicycle, etc. The implementation detects objects in each frame of a video and draws bounding boxes around them with their respective labels and confidence scores.
Dependencies

    - OpenCV (cv2)
    - NumPy

## Setup

    Clone the repository: git clone https://github.com/username/repo.git
    Install the dependencies: pip install opencv-python numpy
    Download the YOLOv3 weights file: yolov3.weights
    Download the YOLOv3 configuration file: yolov3.cfg
    Download the COCO dataset labels file: coco.names

## Usage

    Navigate to the cloned repository: cd repo
    Run the script with the video file as the argument: python yolov3_video_detection.py path/to/video/file.mp4

## Results

The script will save each frame with the bounding boxes and labels drawn onto them in a new folder called frames. The video will also be displayed in a window with the detected objects and their respective confidence scores.

YOLOv3 Video Detection
## Acknowledgments

This implementation is based on the YOLOv3 implementation by Joseph Redmon. The COCO dataset is provided by Microsoft COCO.
## License

This project is licensed under the MIT License - see the LICENSE file for details.
