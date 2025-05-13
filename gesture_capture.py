import cv2
import mediapipe as mp
import pickle

# Initialize Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize storage
X = []
y = []
gesture_label = "none"

# Label key map
key_to_label = {
    ord('f'): "forward",
    ord('s'): "stop",
    ord('l'): "left",
    ord('r'): "right"
}

print("Press:")
print("  [f] for 'forward'")
print("  [s] for 'stop'")
print("  [l] for 'left'")
print("  [r] for 'right'")
print("  [q] to quit and save data")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = []
            for pt in handLms.landmark:
                lm.extend([pt.x, pt.y])  # Only x and y (ignore z)

            if gesture_label != "none":
                X.append(lm)
                y.append(gesture_label)

            # Draw landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Display current label on screen
    cv2.putText(img, f"Label: {gesture_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Collection", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in key_to_label:
        gesture_label = key_to_label[key]
        print(f"Switched label to '{gesture_label}'")

cap.release()
cv2.destroyAllWindows()

# Save the collected data
with open("gesture_data.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("✅ Data collection complete and saved as 'gesture_data.pkl'")
