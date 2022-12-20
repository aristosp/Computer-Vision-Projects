import cv2
import numpy as np

hotel3 = cv2.imread("hotel/hotel-03.png")
hotel2 = cv2.imread("hotel/hotel-02.png")
hotel1 = cv2.imread("hotel/hotel-01.png")
hotel = cv2.imread("hotel/hotel-00.png")
sift = cv2.xfeatures2d_SIFT.create(300)
keys, desc = sift.detectAndCompute(hotel, None)
keys1, desc1 = sift.detectAndCompute(hotel1, None)
keys2, desc2 = sift.detectAndCompute(hotel2, None)
keys3, desc3 = sift.detectAndCompute(hotel3, None)


def matchcross(d, d1):
    matches = []
    for i in range(d.shape[0]):
        vector = d[i, :]
        euclid_dis = np.linalg.norm(d1 - vector, axis=1)
        idx = np.argmin(euclid_dis)
        min_dist = euclid_dis[idx]
        for j in range(d1.shape[0]):
            vector1 = d1[j, :]
            euclid_dis1 = np.linalg.norm(d - vector1, axis=1)
            idx1 = np.argmin(euclid_dis1)
            if idx == idx1:
                matches.append(cv2.DMatch(i, idx1, min_dist))
    return matches


def points(k, k1, match):
    pt1 = []
    pt2 = []
    for x in match:
        pt1.append(k[x.queryIdx].pt)
        pt2.append(k1[x.trainIdx].pt)
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return pt1, pt2


def creating_pan(pt1, pt2, img1, img2):
    M, _ = cv2.findHomography(pt1, pt2, cv2.RANSAC)
    pan_out = cv2.warpPerspective(img1, M, (img1.shape[1]+img2.shape[1], img1.shape[0] + img2.shape[0]))
    pan_out[0:img2.shape[0], 0:img2.shape[1]] = img2
    return pan_out


match1 = matchcross(desc, desc1)
points0, points1 = points(keys, keys1, match1)
pan01 = creating_pan(points0, points1, hotel, hotel1)
keys01, desc01 = sift.detectAndCompute(pan01, None)


match2 = matchcross(desc01, desc2)
points01, points2 = points(keys01, keys2, match2)
pan012 = creating_pan(points01, points2, pan01, hotel2)
keys012, desc012 = sift.detectAndCompute(pan012, None)


match3 = matchcross(desc012, desc3)
points012, points3 = points(keys012, keys3, match3)
panorama = creating_pan(points012, points3, pan012, hotel3)
cv2.namedWindow('panorama', cv2.WINDOW_NORMAL)
cv2.imshow('panorama', panorama)
cv2.waitKey()
