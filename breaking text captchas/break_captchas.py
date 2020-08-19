# USAGE
# python classify.py --model model.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="model.model")
ap.add_argument("-l", "--labelbin", default="lb.pickle")
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

img_path = args["image"]

# load the image
img = cv2.imread(img_path)
output = img.copy()
HXW = 24

filename = img_path.split(os.path.sep)[-1]
filename_copy = filename
filename = os.path.splitext(filename)[0]
#digits = list(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.threshold(img, 0, 255,
cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(img.copy(),
cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0]
digit_boxs = []
boxs = []
data = []
cv2.imshow("img", img)
cv2.waitKey(0)
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    boxs.append( (x, y, w, h) )
boxs = sorted(boxs, key=lambda x: x[0])
for box in boxs:
	(x, y, w, h) = box
	cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)
	box = img[y-2:y+h+2, x-2:x+w+2]
	cv2.imshow("img", img)
	cv2.waitKey(0)
	img = cv2.resize(img, (HXW, HXW))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	data.append(img)
data = np.array(data, dtype="float") / 255.0




#image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the label
# binarizer
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

digits = ""
# classify the input image
for image in data:
	proba = model.predict(image)
	idx = np.argmax(proba)
	label = lb.classes_[idx]
	digits += label

print("digits", digits)
if digits == filename:
	correct = "correct"
else:
	correct = "wrong"

# we'll mark our prediction as "correct" of the input image filename
# contains the predicted label text (obviously this makes the
# assumption that you have named your testing image files this way)
#filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
#correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# build the label and draw the label on the image
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.putText(output, digits, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
print("[INFO] {}".format(digits))
cv2.imshow("Output", output)
cv2.waitKey(0)
