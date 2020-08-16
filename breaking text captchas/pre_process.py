





#####################################
#
# P$UED0 C0DE
#
#   load image paths
#   for path in image paths:
#       load image
#       get filename
#       convert image to gray scale
#       thresh hold the image
#       find contours
#       for contour in contours:
#           make image, a box around the contour
#           save this image to a list
#           save the corresponding label from the filename to a list
#   now you have the two lists that will be used for training
#
#
#####################################

from imutils import paths
import cv2
import os

dataset = "captchas"
HXW = 24
data = []
labels = []
img_paths = sorted(list(paths.list_images(dataset)))
itr = 0
# Load each original captcha
for img_path in img_paths:
    filename = img_path.split(os.path.sep)[-1]
    filename = os.path.splitext(filename)[0]
    digits = list(filename)
    # Extract each char from the captcha
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(img.copy(),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    digit_boxs = []
    #cv2.imshow("captcha", img)
    #cv2.waitKey(0)
    # Save the char and its corresponding label to lists
    tmp_data = []
    tmp_labels = []
    for (contour, digit) in zip(contours, digits):
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)
        box = img[y-2:y+h+2, x-2:x+w+2]
        tmp_data.append(box)
        tmp_labels.append(digit)
        #cv2.imshow(digit, box)
        #cv2.waitKey(0)
    if len(tmp_data) != 4:
        continue
    for (img, label) in zip(tmp_data, tmp_labels):
        data.append(img)
        labels.append(label)

#print(len(data))
#print(len(labels))
#for (img, label) in zip(data, labels):
    #cv2.imshow(label, img)
    #cv2.waitKey(0)
