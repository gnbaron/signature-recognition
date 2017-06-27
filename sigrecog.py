import cv2
import os
import numpy as np

def main():
    print('OpenCV version: '+ cv2.__version__)

    current_dir = os.path.dirname(__file__)

    input_folder = os.path.join(current_dir, 'data/training/021')
    input_file = '13_021.PNG'
    img = cv2.imread(os.path.join(input_folder, input_file), 0)

    img = preprocess(img)

    print(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess(img):
    # constrast
    #img = img * 1.3 - 255
    denoised = cv2.fastNlMeansDenoising(img)
    tresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cropped = crop(tresh)
    return cropped


def crop(img):
    inverted = 255 - img
    points = cv2.findNonZero(inverted)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y+h, x: x+w]


def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == '__main__':
    main()