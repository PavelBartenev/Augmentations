import cv2 as cv
import numpy as np
import random
import math


def add_line_colour(image):
    size = int(image.shape[0] / 4)

    pos = random.randint(1, image.shape[1])

    colours = (np.random.randint(0, 255, size, dtype=np.uint8).reshape(size, 1) *
               np.ones((1, size, 3), dtype=np.uint8)).reshape(size, 3)

    t = np.min([8, image.shape[1] - pos])

    steps = np.linspace(0, t, t + 1, dtype=np.uint8)
    masks = np.random.randint(0, 2, steps.shape[0])
    steps = list(set(steps * masks))
    
    for step in steps:
        image[:size, pos + step] = colours
        image[size:size * 2, pos + step] = colours
        image[size * 2:size * 3, pos + step] = colours
        image[size * 3:size * 4, pos + step] = colours

    return image


def add_line(image, color=(0, 0, 0)):
    position = random.randint(0, image.shape[1])
    weight = random.randint(1, 3)
    image_draw = cv.line(image, (position, 0), (position, image.shape[0]), color, weight)
    return image_draw


def add_noise(image):
    mean = random.randint(0, 5)
    sigma = random.uniform(0.5, 0.9) ** 0.5

    row, col, ch = image.shape

    gauss = np.uint8(np.random.normal(mean, sigma, (row, col, ch)) * 255)
    gauss = gauss.reshape(row, col, ch)

    noisy = cv.add(image, gauss)

    return noisy


def add_gradient(image):
    mask = cv.inRange(image, (255, 255, 255),
                      (255, 255, 255))

    mask_print = cv.bitwise_not(mask)
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    mask_print = cv.cvtColor(mask_print, cv.COLOR_GRAY2BGR)
    mask_print = cv.bitwise_and(mask_print, image)

    alpha = np.linspace(0.3, 1, mask_print.shape[1]).reshape(mask_print.shape[1], 1)
    value = np.array(mask_print * alpha + (np.ones((mask_print.shape[0], mask_print.shape[1], 3)) * 255)
                     * (np.ones((mask_print.shape[1], 1)) - alpha), dtype=np.uint8)

    mask_print = (np.clip(value, 0, 255))
    mask = cv.bitwise_and(mask, image)
    result = cv.add(mask, mask_print)

    return result


def add_printing_defects(image):
    image = add_noise(image)
    image = add_line(image)
    image = add_line_colour(image)
    image = add_gradient(image)

    return image


def change_corners(dst, matrix):
    corner_1 = matrix @ np.array([0, 0, 1])
    corner_2 = matrix @ np.array([0, dst.shape[0] - 1, 1])
    corner_3 = matrix @ np.array([dst.shape[1] - 1, dst.shape[0] - 1, 1])
    corner_4 = matrix @ np.array([dst.shape[1] - 1, 0, 1])

    data = [(corner_1 / corner_1[2])[0:2], (corner_2 / corner_2[2])[0:2],
            (corner_3 / corner_3[2])[0:2], (corner_4 / corner_4[2])[0:2]]

    return data


def change_perspective(image):
    h, w = image.shape[0] - 1, image.shape[1] - 1
    
    x1_to, y1_to, x2_to, y2_to = random.randint(0, 60), random.randint(0, 60), \
                                 random.randint(w - 100, w), random.randint(0, 65)

    x3_to, y3_to, x4_to, y4_to = random.randint(0, 30), random.randint(h - 100, h), \
                                 random.randint(x2_to - 30, x2_to + 30), random.randint(h - 100, h)
    
    position1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    position2 = np.float32([[x1_to, y1_to], [x2_to, y2_to], [x3_to, y3_to], [x4_to, y4_to]])
    
    M = cv.getPerspectiveTransform(position1, position2)
    corners = change_corners(image, M)
    
    max_x = np.max([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
    min_x = np.min([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
    max_y = np.max([corners[0][1], corners[1][1], corners[2][1], corners[3][1]])
    min_y = np.min([corners[0][1], corners[1][1], corners[2][1], corners[3][1]])

    cor_count = 0

    if min_y < 0:
        if min_x < 0:
            cor_count = 3
        else:
            cor_count = 1
    elif min_x < 0:
        cor_count = 2
        
    dst = cv.warpPerspective(image, M, (int(np.max([max_x - min_x, max_x])), int(np.max([max_y - min_y, max_y]))),
                             flags=cv.INTER_LINEAR)
    
    return dst, corners, cor_count


def transform(image, back):
    if (image.shape[0] - image.shape[1]) * (back.shape[0] - back.shape[1]) < 0:
        back = rotate(back, 90)

    image, corners, cor_count = change_perspective(image)
    coeff = random.uniform(0.8, 1.)

    k = min(back.shape[0] * coeff / image.shape[0], back.shape[1] * coeff / image.shape[1])

    image = cv.resize(image, (int(k * image.shape[1]), int(k * image.shape[0])))
    corners = [k * corners[0], k * corners[1], k * corners[2], k * corners[3], ]

    return image, back, corners, cor_count


def rotate(image, angle, flag=cv.INTER_LINEAR):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    rotation_mat = cv.getRotationMatrix2D(center, angle, 1)
    radians = math.radians(angle)

    sin = math.sin(radians)
    cos = math.cos(radians)

    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - center[1])

    rotated_mat = cv.warpAffine(image, rotation_mat, (bound_w, bound_h), flags=flag)
    
    return rotated_mat
