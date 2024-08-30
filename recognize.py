from testExtractLandmarksModel import HandGestureRecognizer  # Updated import for the new recognizer class


def main():

    # Initialize hand gesture recognizer and run real-time recognition
    recognizer = HandGestureRecognizer(model_path='model/hand_gesture_classification_model.h5')
    recognizer.run()
    
    print("Processing and recognition completed.")

if __name__ == "__main__":
    main()
