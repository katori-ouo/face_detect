import dlib
import cv2
import cv2
import numpy as np
import random


def get_points(pic, txt, p_path):
    f = open(txt, 'w+')
    detector = dlib.get_frontal_face_detector()
    # 相撞
    predicator = dlib.shape_predictor(p_path)
    win = dlib.image_window()
    img1 = cv2.imread(pic)

    dets = detector(img1, 1)
    print("Number of faces detected : {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}  left:{}  Top: {} Right {}  Bottom {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()
        ))
        lanmarks = [[p.x, p.y] for p in predicator(img1, d).parts()]
        for idx, point in enumerate(lanmarks):
            f.write(str(point[0]))
            f.write("\t")
            f.write(str(point[1]))
            f.write('\n')


# Check if a point is insied a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color)


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    trangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in trangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if (rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3)):
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)


# Draw voronoi diagram
def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.fillConvexPoly(img, ifacet, color)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0))


if __name__ == '__main__':
    # Define window names;
    win_delaunary = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"

    # Turn on animations while drawing triangles
    animate = True

    # Define colors for drawing
    delaunary_color = (255, 255, 255)
    points_color = (0, 0, 255)

    # Read in the image
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    png_path = "E:/py_srcipts/faces/face3.png"

    txt_path = "E:/py_srcipts/faces/points3.txt"

    get_points(png_path, txt_path, predictor_path)

    img = cv2.imread(png_path)

    # Keep a copy   around
    img_orig = img.copy()

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2d
    subdiv = cv2.Subdiv2D(rect)
    # Create an array of points
    points = []
    # Read in the points from a text file
    with open(txt_path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

        # Show animate
        if animate:
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay(img_copy, subdiv, (255, 255, 255))
            cv2.imshow(win_delaunary, img_copy)
            cv2.waitKey(100)

    # Draw delaunary triangles
    draw_delaunay(img, subdiv, (255, 255, 255))

    # Draw points
    for p in points:
        draw_point(img, p, (0, 0, 255))

    # Allocate space for Voroni Diagram
    img_voronoi = np.zeros(img.shape, dtype=img.dtype)

    # Draw Voonoi diagram
    draw_voronoi(img_voronoi, subdiv)

    # Show results
    cv2.imshow(win_delaunary, img)
    cv2.imshow(win_voronoi, img_voronoi)
    cv2.waitKey(0)
