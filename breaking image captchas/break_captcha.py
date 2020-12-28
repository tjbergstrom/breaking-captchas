# break_captcha.py
#
# $ python3 -W ignore break_captcha.py
#
# Upload a list of images from a test dataset directory
# See a demonstration of how the process works
# And see the system breaking the input captchas
#
# The input filenames are named in the order that the thumbnails are processed,
# And with the expected thumbs that should be selected,
# Such as 001000000.png where the thumb in the top right corner whould be selected.
# And that's what I'm using to evaluate.


import os
import argparse
from imutils import paths
from extract_captcha import Extract as ec
from select_thumbs import Select


if __name__ == "__main__":
    dataset = "test_uploads"
    img_paths = sorted(list(paths.list_images(dataset)))
    S = Select()
    correct = 0
    total = 0
    for img_path in img_paths:
        captcha_txt = ec.extract_txt(img_path)
        if not captcha_txt:
            continue
        thumbnails = ec.extract_thumbs(img_path)
        if len(thumbnails[0]) < 6:
            continue
        expected_selections = ec.extract_expected(img_path)
        selections = S.make_selections(captcha_txt, thumbnails)
        cor, tot = S.draw(img_path, thumbnails, selections, expected_selections)
        correct += cor
        total += tot
    print(f"Overal accuracy: {(correct/total)*100:.2f}%")



##
