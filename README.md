# Face Recognition Application

This project implements a real-time face recognition system using Python and OpenCV. It uses the `face_recognition` library to encode and recognize faces from a predefined list of images. The application displays the recognized person's name and role in an overlay on a video feed.

---

## Features

- Detect and recognize faces in real-time from a webcam feed.
- Associate recognized faces with names and predefined roles.
- Display information dynamically on the video frame.
- Handles scenarios where no faces are detected in images or live feed.

---

## Technologies Used

- **Python**: Core programming language.
- **OpenCV**: For video capture and real-time image processing.
- **face_recognition**: For face detection and encoding.
- **NumPy**: For numerical operations.

---


---

## How It Works

1. **Preprocessing**:
   - Reads all image files from the `persons/` directory.
   - Encodes faces from the images and stores them for comparison.

2. **Face Detection**:
   - Captures video from the webcam.
   - Detects faces in each frame and encodes them.

3. **Face Matching**:
   - Compares detected faces with stored encodings.
   - Identifies the person if a match is found within a defined confidence threshold.

4. **Information Display**:
   - Overlays the person's name and role dynamically on the video frame.

---

## Prerequisites

- Python 3.x
- Required Python Libraries:
  - OpenCV: `pip install opencv-python`
  - face_recognition: `pip install face-recognition`
  - NumPy: `pip install numpy`

---

## Setup Instructions

1. Clone this repository or download the source code.
2. Place images of known persons in the `persons/` directory. Name the files as `Name.jpg` (e.g., `John.jpg`).
3. Install the required Python libraries.
4. Run the script:
   ```bash
   python3 test.py


