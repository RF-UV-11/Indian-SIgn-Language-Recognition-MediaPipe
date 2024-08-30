from extractImagesLandmarks import HandLandmarkExtractor, process_videos
import utils


def main():
    input_dir = 'sign_language_videos'
    output_dir = 'extracted_landmarks_images'
    output_keypoints_dir = 'extracted_landmarks_keypoints'

    # Subdirectories for each letter
    sub_dirs = [letter for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']

    # Create necessary directories for images and keypoints
    utils.create_directories(output_dir, sub_dirs)
    utils.create_directories(output_keypoints_dir, sub_dirs)

    # Initialize landmark extractor
    extractor = HandLandmarkExtractor()

    # Process videos to extract landmarks and save images/keypoints
    process_videos(input_dir, output_dir, output_keypoints_dir, extractor)
    
    # Close the extractor to release resources
    extractor.close()

if __name__ == "__main__":
    main()
