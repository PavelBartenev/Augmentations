import os
import random
import cv2 as cv

from background_add import add_background
from picture_augs import add_printing_defects
from picture_augs import transform

path = 'D:/CV/task4/'

scans = os.listdir(path + "documents")
surfaces = os.listdir(path + "surfaces")

for i in range(10):
    for j, scan in enumerate(scans):
        surface = random.choice(surfaces)

        image = cv.imread(path + "documents/"+ scan)
        back = cv.imread(path + "surfaces/" + surface)

        image = add_printing_defects(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
        image, back, corners, corNum = transform(image, back)
        image, corners = add_background(image, back, corners, corNum)

        cv.imwrite(path + "augmented/" + str(i) + str(j) + ".png", image)
