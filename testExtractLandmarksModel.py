import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter

class HandGestureRecognizer:
    def __init__(self, model_path, max_num_hands=2, detection_confidence=0.8, tracking_confidence=0.8):
        # Load the trained model
        self.model = self._load_model(model_path)

        # Initialize MediaPipe Hands and Drawing utilities
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        # Gesture labels corresponding to the model's output classes
        self.gesture_labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        # Create a deque (sliding window) to store the recent predictions
        self.prediction_window = deque(maxlen=5)  # Adjust the window size as needed

    def _load_model(self, model_path):
        """Loads the gesture classification model."""
        try:
            model = load_model(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

    def preprocess_landmarks(self, landmarks):
        """Preprocesses the landmarks into a flat array for model input."""
        flat_landmarks = []
        for lm in landmarks:
            flat_landmarks.extend([lm.x, lm.y, lm.z])  # Use x, y, z for 3D data
        return np.array(flat_landmarks).flatten()

    def predict_gesture(self, landmark_array):
        """Predicts the gesture based on the landmark array."""
        try:
            prediction = self.model.predict(np.expand_dims(landmark_array, axis=0))
            predicted_class = np.argmax(prediction)
            predicted_gesture = self.gesture_labels[predicted_class]
            confidence = prediction[0][predicted_class] * 100
            return predicted_gesture, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0

    def run(self):
        """Runs the real-time hand gesture recognition."""
        # Start video capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting real-time hand gesture recognition. Press 'q' to quit.")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image. Exiting...")
                    break

                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image and find hands
                results = self.hands.process(image_rgb)

                # Check if any hand is detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks and connections on the frame
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Preprocess the landmarks for prediction
                        landmark_array = self.preprocess_landmarks(hand_landmarks.landmark)

                        # Predict the gesture
                        predicted_gesture, confidence = self.predict_gesture(landmark_array)

                        if predicted_gesture:
                            # Add the prediction to the sliding window
                            self.prediction_window.append(predicted_gesture)

                            # Determine the most frequent prediction in the window
                            most_common_gesture, count = Counter(self.prediction_window).most_common(1)[0]

                            # Display the most frequent predicted gesture on the frame if confidence is high enough
                            if count >= len(self.prediction_window) // 2 and confidence > 70:  # Threshold to display
                                cv2.putText(frame, f'{most_common_gesture} ({confidence:.2f}%)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

                # Display the frame
                cv2.imshow('Hand Gesture Recognition', frame)

                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("Real-time testing ended.")

