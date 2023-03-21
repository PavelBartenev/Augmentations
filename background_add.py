import numpy as np
import random
import cv2 as cv


def place_background(img1, img2):
    alpha = img1[:, :, 3]
    alpha = (np.repeat(alpha, 3).reshape(img1.shape[0], img1.shape[1], 3) / 255)
    value = np.array((img1[:, :, :3] * alpha) + img2 * (np.ones((img1.shape[0], img1.shape[1], 3)) - alpha),
                     dtype=np.uint8)
    result = (np.clip(value, 0, 255))
    return result


def add_background(image, surface, corners, cor_count):
    surface = cv.GaussianBlur(surface, (7, 7), 0)

    max_x = np.max([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
    min_x = np.min([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
    max_y = np.max([corners[0][1], corners[1][1], corners[2][1], corners[3][1]])
    min_y = np.min([corners[0][1], corners[1][1], corners[2][1], corners[3][1]])

    delta = 40

    pos_x = random.randint(np.max([int(surface.shape[1] / 2 - (max_x - min_x) / 2 - delta), 0]),
                           np.min([int(surface.shape[1] / 2 - (max_x - min_x) / 2 + delta),
                           int(surface.shape[1] - (max_x - min_x))]))

    pos_y = random.randint(np.max([int(surface.shape[0] / 2 - (max_y - min_y) / 2 - delta), 0]),
                           np.min([int(surface.shape[0] / 2 - (max_y - min_y) / 2 + delta),
                           int(surface.shape[0] - (max_y - min_y))]))

    if cor_count == 1:
        pos_y = 0
    elif cor_count == 2:
        pos_x = 0
    elif cor_count == 3:
        pos_x = 0
        pos_y = 0

    rows, cols, ch = image.shape

    if pos_x + cols > surface.shape[1]:
        pos_x = surface.shape[1] - cols

    if pos_y + rows > surface.shape[0]:
        pos_y = surface.shape[0] - rows

    for i in range(4):
        corners[i][0] += pos_x
        corners[i][1] += pos_y
        corners[i] = corners[i].tolist()

    common_back = surface[pos_y: rows + pos_y, pos_x: cols + pos_x]
    result = place_background(image, common_back)
    surface[pos_y: pos_y + rows, pos_x: pos_x + cols] = result

    return surface, corners
