import os


def create_directories(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)
