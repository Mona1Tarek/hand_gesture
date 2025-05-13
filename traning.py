import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
with open("gesture_data.pkl", "rb") as f:
    X, y = pickle.load(f)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save the trained model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model trained and saved as gesture_model.pkl")
