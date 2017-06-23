import cv2
import os
import numpy as np

def main():
    print('OpenCV version: '+ cv2.__version__)

    dir = os.path.dirname(__file__)

    inputFolder = os.path.join(dir, '../data')
    inputFile = '001001_000.png'
    img = cv2.imread(os.path.join(inputFolder, inputFile), 0)
    
    # constrast adjust
    # img = np.float32(img) * (1.2 / 255)

    print(np.float32(img))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == '__main__':
    main()
