import cv2
import os
import numpy as np
import network


def main():
    print('OpenCV version {} '.format(cv2.__version__))

    current_dir = os.path.dirname(__file__)

    author = '021'
    training_folder = os.path.join(current_dir, 'data/training/', author)
    test_folder = os.path.join(current_dir, 'data/test/', author)

    training_data = []
    for filename in os.listdir(training_folder):
        img = cv2.imread(os.path.join(training_folder, filename), 0)
        if img is not None:
            data = prepare(img)
            result = [[0], [1]] if "genuine" in filename else [[1], [0]]
            result = np.array(result)
            result = np.reshape(result, (2, 1))
            training_data.append((data, result))

    test_data = []
    for filename in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder, filename), 0)
        if img is not None:
            data = prepare(img)
            result = 1 if "genuine" in filename else 0
            test_data.append((data, result))

    net = network.Net([901, 500, 500, 2])
    net.sgd(training_data, 30, 10, 3.0, test_data)


def prepare(input):
    # preprocessing the image input
    clean = cv2.fastNlMeansDenoising(input)
    tresh = cv2.adaptiveThreshold(clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = crop(tresh)

    # 40x10 image as a flatten array
    flatten_img = cv2.resize(img, (40, 10), interpolation=cv2.INTER_AREA).flatten()

    # resize to 400x100
    resized = cv2.resize(img, (400, 100), interpolation=cv2.INTER_AREA)
    columns = np.sum(resized, axis=0)  # sum of all columns
    lines = np.sum(resized, axis=1)  # sum of all lines

    h, w = img.shape
    aspect = w / h

    data = np.array([*flatten_img, *columns, *lines, aspect], dtype=float)
    return np.reshape(data, (901, 1))


def crop(img):
    inverted = 255 - img
    points = cv2.findNonZero(inverted)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y+h, x: x+w]


if __name__ == '__main__':
    main()