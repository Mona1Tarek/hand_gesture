import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# --- YOUR ROBOT CONTROL FUNCTIONS (edit as needed) ---
def move_forward():
    print("🚗 Moving Forward")

def move_left():
    print("↪️ Turning Left")

def move_right():
    print("↩️ Turning Right")

def stop():
    print("🛑 Stopping")

def control_robot(gesture):
    if gesture == "forward":
        move_forward()
    elif gesture == "left":
        move_left()
    elif gesture == "right":
        move_right()
    elif gesture == "stop":
        stop()

# --- Real-time capture and prediction ---
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = []
            for pt in handLms.landmark:
                lm.extend([pt.x, pt.y])  # 21 points * 2
            
            # Predict gesture
            prediction = model.predict([lm])[0]
            control_robot(prediction)  # trigger control

            # Visualize
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.imshow("Prediction", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()