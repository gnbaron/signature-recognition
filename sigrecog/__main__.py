import cv2
import os
import numpy as np

def main():
    print('OpenCV version: '+ cv2.__version__)

    dir = os.path.dirname(__file__)

    inputFolder = os.path.join(dir, '../data')
    inputFile = '001001_000.png'
    img = cv2.imread(os.path.join(inputFolder, inputFile), 0)
    
    # constrast
    res = np.float32(img, dtype = np.uint8) * (1.2 / 255)

    # denoising
    img = cv2.fastNlMeansDenoising(img)

    # binarization
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # crop blank space
    img = crop(img)

    print(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y + h, x:x + w]

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == '__main__':
    main()
