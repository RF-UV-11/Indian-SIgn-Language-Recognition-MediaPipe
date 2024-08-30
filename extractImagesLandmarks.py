import cv2
import os
import numpy as np
import mediapipe as mp


class HandLandmarkExtractor:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence)
        self.landmark_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                                (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                                (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                                (255, 128, 0), (0, 255, 128), (128, 255, 0), (255, 0, 128),
                                (0, 128, 255), (128, 0, 255), (255, 255, 128), (128, 255, 255)]
        self.connection_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                                  (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                                  (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                                  (255, 128, 0), (0, 255, 128), (128, 255, 0), (255, 0, 128),
                                  (0, 128, 255), (128, 0, 255), (255, 255, 128), (128, 255, 255)]

    def custom_draw_styled_landmarks(self, image, hand_landmarks):
        """
        Draws the hand landmarks and connections with custom colors on the given image.
        """
        for idx, landmark in enumerate(hand_landmarks.landmark):
            cx, cy = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (cx, cy), 5, self.landmark_colors[idx % len(self.landmark_colors)], -1)

        for connection_idx, connection in enumerate(self.mp_hands.HAND_CONNECTIONS):
            start_idx, end_idx = connection
            start_landmark = hand_landmarks.landmark[start_idx]
            end_landmark = hand_landmarks.landmark[end_idx]

            x_start, y_start = int(start_landmark.x * image.shape[1]), int(start_landmark.y * image.shape[0])
            x_end, y_end = int(end_landmark.x * image.shape[1]), int(end_landmark.y * image.shape[0])

            cv2.line(image, (x_start, y_start), (x_end, y_end), self.connection_colors[connection_idx % len(self.connection_colors)], 2)

    def extract_and_save_landmarks(self, video_path, image_output_path, keypoints_output_path, max_frames=100):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, frame_count // max_frames)
        frame_index = 0
        extracted_frames = 0

        while extracted_frames < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                black_frame = np.zeros_like(frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.custom_draw_styled_landmarks(black_frame, hand_landmarks)
                        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        npy_output_path = os.path.join(keypoints_output_path, f'frame_{extracted_frames:02d}.npy')
                        np.save(npy_output_path, keypoints)

                    image_output_image_path = os.path.join(image_output_path, f'frame_{extracted_frames:02d}.png')
                    cv2.imwrite(image_output_image_path, black_frame)
                    extracted_frames += 1

            frame_index += 1

        cap.release()

    def close(self):
        self.hands.close()


def process_videos(input_dir, output_dir, output_keypoints_dir, extractor):
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        letter_input_dir = os.path.join(input_dir, letter)
        letter_output_image_dir = os.path.join(output_dir, letter)
        letter_output_keypoints_dir = os.path.join(output_keypoints_dir, letter)

        if os.path.exists(letter_input_dir):
            for video_file in os.listdir(letter_input_dir):
                video_path = os.path.join(letter_input_dir, video_file)
                if os.path.isfile(video_path) and video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    extractor.extract_and_save_landmarks(video_path, letter_output_image_dir, letter_output_keypoints_dir)

