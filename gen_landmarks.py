import sys
import numpy as np
import matplotlib.pyplot as plt
from create_scene import save_polygons

def generate_landmarks(num):
    samples = []
    while len(samples) < num:
        new_sample = np.random.rand(2) * 2
        if all(np.all(new_sample != existing_sample) for existing_sample in samples):
            samples.append(new_sample)

    return np.array(samples)

# Usage: python3 gen_landmarks.py numberLandmarks FilenameToSaveTo
if __name__ == '__main__':
    if len(sys.argv) < 3: print('Usage: python3 gen_landmarks.py <numberLandmarks> <FilenameToSaveTo>')
    landmarks = generate_landmarks(int(sys.argv[1]))
    save_polygons(landmarks, sys.argv[2])


