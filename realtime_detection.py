import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = "saved_model/object_classifier.h5"
model = load_model(MODEL_PATH)

# Define class names based on training
# Make sure the order matches train_data.class_indices from training
class_names = ['background', 'bottle', 'phone']  # Update if different

# Image settings
IMG_SIZE = (128, 128)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img, verbose=0)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display prediction
    label = f"{class_names[class_id]}: {confidence:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video
    cv2.imshow("Real-Time Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
