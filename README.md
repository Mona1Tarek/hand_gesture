# Hand Gesture Controlled Robot (with Machine Learning)

This project enables real-time robot control using hand gestures detected via a webcam. It leverages computer vision (MediaPipe, OpenCV) and machine learning (scikit-learn) to recognize gestures and trigger robot actions.

## Features
- **Gesture Data Collection:** Capture hand gesture data using your webcam and label them for training.
- **Model Training:** Train a machine learning model (SVM) to classify hand gestures.
- **Real-Time Prediction:** Use the trained model to recognize gestures in real time and control a robot (or simulate control).

## File Overview
- `gesture_capture.py`: Collects hand gesture data and saves it as `gesture_data.pkl`.
- `traning.py`: Trains an SVM classifier on the collected data and saves the model as `gesture_model.pkl`.
- `predicting.py`: Runs real-time gesture recognition and triggers robot control functions based on predictions.
- `gesture_data.pkl`: Pickle file containing collected gesture data (features and labels).
- `gesture_model.pkl`: Pickle file containing the trained SVM model.

## Dependencies
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe
- scikit-learn
- numpy

Install dependencies with:
```bash
pip install opencv-python mediapipe scikit-learn numpy
```

## Usage
### 1. Collect Gesture Data
Run the data collection script and follow on-screen instructions to label gestures:
```bash
python gesture_capture.py
```
Press:
- `f` for 'forward'
- `s` for 'stop'
- `l` for 'left'
- `r` for 'right'
- `q` to quit and save data

### 2. Train the Model
Train the gesture recognition model:
```bash
python traning.py
```
This will output the model's accuracy and save it as `gesture_model.pkl`.

### 3. Run Real-Time Prediction
Start real-time gesture recognition and robot control:
```bash
python predicting.py
```
Show gestures to your webcam. The script will print and visualize the recognized gesture and trigger the corresponding robot control function (edit these functions as needed for your robot).

## Customization
- Edit the robot control functions in `predicting.py` to interface with your actual robot hardware.
- Add more gestures by updating the label map and retraining the model.

## Notes
- Ensure your webcam is connected and accessible.
- The scripts use only x and y coordinates of hand landmarks for simplicity.
- The project is a template; adapt it for your specific robot or application.

## License
MIT License 