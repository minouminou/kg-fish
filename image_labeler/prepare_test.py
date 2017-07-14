"""
Prepare test data
"""
import os
import cv2

in_dir = '~/kg-fish/data/raw/test_stg1/'
out_dir = '~/kg-fish/data/test_stg1'

img_filenames = [f for f in os.listdir(in_dir) if f.lower().endswith(".jpg")]
img_filenames.sort()
for img_filename_index, img_filename in enumerate(img_filenames):
    print(img_filename_index, img_filename)
    img_path = os.path.join(in_dir, img_filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite(os.path.join(out_dir, img_filename), img)


