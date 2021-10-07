# noinspection PyUnresolvedReferences
import sys
# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import cv2
# noinspection PyUnresolvedReferences
import dlib
# noinspection PyUnresolvedReferences
from scipy.spatial import Delaunay

predictor_model = 'shape_predictor_68_face_landmarks.dat'


def read_file(path):
    faces = []
    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename)
        faces.append(img)
    return faces


def mask_points(file):
    points = np.loadtxt(file)
    points = points.astype(np.float32)
    return points


def get_points(image):  # 用 dlib 来得到人脸的特征点

    face_detector = dlib.get_frontal_face_detector()  # 正向人脸检测器，进行人脸检测，提取人脸外部矩形框
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    try:
        detected_face = face_detector(image, 1)[0]
    except:
        # print('No face detected in image {}'.format(image))
        print('No face detected in image')
    pose_landmarks = face_pose_predictor(image, detected_face)  # 获取landmark
    points = []
    for p in pose_landmarks.parts()[1:16]:
        points.append([p.x, p.y])
    for p in pose_landmarks.parts()[28:36]:
        points.append([p.x, p.y])
    points.append([pose_landmarks.parts()[48].x, pose_landmarks.parts()[1].y])
    points.append([pose_landmarks.parts()[51].x, pose_landmarks.parts()[1].y])
    points.append([pose_landmarks.parts()[54].x, pose_landmarks.parts()[1].y])
    points.append([pose_landmarks.parts()[57].x, pose_landmarks.parts()[1].y])
    points.append([pose_landmarks.parts()[62].x, pose_landmarks.parts()[1].y])
    return np.array(points)


def getDelaunay(points):
    return Delaunay(points).simplices


# Apply affine transform calculated using srcTri and dstTri to src and output an image of size
def applyAffineTransform(src, srcTri, dstTri, size):
    # Givern a pair of triangles , find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Tranform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []
    for i in range(0, 3):
        t1Rect.append([(t1[i][0] - r1[0]), (t1[i][1] - r1[1])])
        t2Rect.append([(t2[i][0] - r2[0]), (t2[i][1] - r2[1])])
        t2RectInt.append([(t2[i][0] - r2[0]), (t2[i][1] - r2[1])])

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


def mask_generation(img1, m_points, img2, r_path, num, filename):  # img1为口罩，img2为人脸，m_points为手工注释的点坐标文件路径
    names = os.listdir(filename)
    # 读取图片特征点
    points1 = mask_points(m_points)
    points2 = get_points(img2)
    img1Warped = np.copy(img2)

    # 找到人脸轮廓
    hull1 = []
    hull2 = []
    img_corp = img1.copy()
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
    hullIndex1 = cv2.convexHull(np.array(points1))

    # 画图
    # for i in range(len(hullIndex1)):
    #     cv2.line(img_corp, tuple(hullIndex1[i][0]), tuple(hullIndex1[(i + 1) % len(hullIndex1)][0]), (255, 0, 0), 2)
    # img_point = img1.copy()
    # for i in points1:
    #     cv2.circle(img_point, tuple(i), 2, (0, 255, 0), 5)
    # fillbox = np.hstack((img1, img_point, img_corp))
    # cv2.imwrite("imgs/fillbox.png", fillbox)

    # 边界框
    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    # 基于边界框内的点，找到三角形
    sizeImg2 = img1.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    Tri = getDelaunay(hull2)
    # if len(Tri) == 0:
    #     quit()

    img2_con = img1.copy()
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(Tri)):
        t1 = []
        t2 = []

        # Get points for img1,img2 corresponding to triangles
        for j in range(0, 3):
            t1.append(hull1[Tri[i][j]])  # t1为图1第i个三角形的第j个顶点
            t2.append(hull2[Tri[i][j]])  # t2为图2第i个三角形的第j个顶点

        # for j in range(0, 3):
        #     cv2.line(img2_con, t1[j], t1[(j + 1) % 3], (255, 0, 0), 2)

        warpTriangle(img1, img1Warped, t1, t2)

    # out_convex = np.hstack((img1, img2_con))
    # cv2.imwrite("imgs/fillbox_tri.png", out_convex)
    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    # cv2.imwrite("imgs/mask3.png", mask)
    cv2.imwrite(r_path + "/" + '%s' % names[num], img1Warped)


face_path = "F:/Pycharm Files/faces"
mp_path = "F:/Pycharm Files/points.txt"
r_path = "F:/Pycharm Files/results"
mask = cv2.imread('imgs/mask.png')
faces = read_file(face_path)
print(len(faces))
for i in range(0, len(faces)):
    print(i)
    mask_generation(mask, mp_path, faces[i], r_path, i, face_path)
# 备份F:\Faster_Rcnn\VOCdevkit\VOC2007以及2007_***.txt
# 选取F:\Faster_Rcnn\VOCdevkit\VOC2007\JPEGImages中单个人脸的图像约1000张
# 复制到C:/Users/fjj/Desktop/JPEGImages
# 运行mask_generation，结果替换JPEGImages的图像。至此，JPEGImages替换完成
# ImageSets不变
# 找到F:\Faster_Rcnn\VOCdevkit\VOC2007\Annotations中对应的xml文件，替换object_name为masked_face
# 生成的xml替换Annotation的文件，至此，数据集更新完成
# voc_annotation.py中classes增加masked_face
# F:\Faster_Rcnn\model_data\voc_classes添加masked_face
