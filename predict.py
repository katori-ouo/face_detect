from PIL import Image

from frcnn import FRCNN
import numpy
import cv2

frcnn = FRCNN()


def process(path):
    origin = cv2.imread(path)
    filter = cv2.medianBlur(origin, 5)  # 中值滤波器
    # cv2.imshow("gamma", filter)
    # cv2.waitKey(0)
    gamma = numpy.power(origin / float(numpy.max(filter)), 1.5)  # gamma校正
    # cv2.imshow("filter", gamma)
    # cv2.waitKey(0)
    cv2.imwrite(path, gamma*255)
    # print("ok")
    return


while True:
    img = input('Input image filename:')
    try:
        # process(img)  # 图像预处理操作
        image = Image.open(img)

        image = image.convert("RGB")  # 转换成RGB图片，可以用于灰度图预测。

    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()

