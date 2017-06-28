import cv2
import os
import numpy as np
import network

def main():
    print('OpenCV version {} '.format(cv2.__version__))

    current_dir = os.path.dirname(__file__)

    training_folder = os.path.join(current_dir, 'data/training/021')
    test_folder = os.path.join(current_dir, 'data/test/021')

    training_data = []
    for filename in os.listdir(training_folder):
        training_class = [[1],[0]] if "_ge_" in filename else [[0],[1]]
        img = cv2.imread(os.path.join(training_folder, filename), 0)
        if img is not None:
            data = prepare(img)
            training_data.append((data, np.array(training_class, dtype=int)))

    test_data = []
    for filename in os.listdir(test_folder):
        test_class = 1 if "_ge_" in filename else 0
        img = cv2.imread(os.path.join(test_folder, filename), 0)
        if img is not None:
            data = prepare(img)
            test_data.append((data, test_class))

    net = network.Net([901, 500, 2])
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

    temp = np.append(np.concatenate((flatten_img,columns,lines)),aspect)

    x = [[j] for j in temp]

    return x


def crop(img):
    inverted = 255 - img
    points = cv2.findNonZero(inverted)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y+h, x: x+w]


if __name__ == '__main__':
    main()