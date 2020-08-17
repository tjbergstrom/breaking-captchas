# break_captchas.py
# August 2020
# Use a trained model with unlabeled input captchas
# And see if the model's predictions are correct

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from imutils import paths

dataset = "test_captchas"
HXW = 24
img_paths = sorted(list(paths.list_images(dataset)))
dbug_img_show = False
model = load_model("model.model")
lb = pickle.loads(open("lb.pickle", "rb").read())

print()

# Load each original captcha
for img_path in img_paths:
    filename = img_path.split(os.path.sep)[-1]
    filename = os.path.splitext(filename)[0]
    filename_cpy = filename
    # Extract each char from the captcha
    img = cv2.imread(img_path)
    img_cpy = img
    if dbug_img_show:
        cv2.imshow(filename_cpy, img)
        cv2.waitKey(0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if dbug_img_show:
        cv2.imshow("gray scale", img)
        cv2.waitKey(0)
    img = cv2.threshold(img, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if dbug_img_show:
        cv2.imshow("threshold", img)
        cv2.waitKey(0)
    contours = cv2.findContours(img.copy(),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    found_chars = []
    boxs = []
    # Make sure the chars are sorted from left to right
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        boxs.append( (x, y, w, h) )
    boxs = sorted(boxs, key=lambda x: x[0])
    # Show each found char to the model and make a prediction
    for box in boxs:
        (x, y, w, h) = box
        cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)
        box = img[y-2:y+h+2, x-2:x+w+2]
        found_chars.append(box)
        if dbug_img_show:
            cv2.imshow(digit, box)
            cv2.waitKey(0)
    if len(found_chars) != 4:
        continue
    decoded_captcha = ""
    for img in found_chars:
        if img.shape[0] < 10 or img.shape[1] < 10:
            continue
        img = cv2.resize(img, (HXW, HXW))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        prob = model.predict(img)[0]
        idx = np.argmax(prob)
        label = lb.classes_[idx]
        # Add the prediction of each found char to the final captcha decoding
        decoded_captcha += label
    # See if the final captcha decoding is the same as the file name
    print("Filename:", img_path)
    if decoded_captcha == filename_cpy:
        decoded_captcha += " - correct"
    else:
        decoded_captcha += " - wrong"
    print("    Decoded captcha is:", decoded_captcha, "\n")
    cv2.imshow(decoded_captcha, img_cpy)
    cv2.waitKey(0)






#
