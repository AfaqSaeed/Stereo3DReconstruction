import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def filterbadkpoints(ptsL,ptsR,vertuncert=5,horiuncert=5,rneg=False):
    
    # remove negative disparity
    if rneg:
        disp = ptsL[:,0]-ptsR[:,0]
        ptsL = ptsL[disp>0]
        ptsR = ptsR[disp>0]
    # remove any points that violate epipolar constraint
    yL = ptsL[:,1]
    yR = ptsR[:,1]
    
    diff = yL-yR
    print(diff)
    vcondition = np.bitwise_and((diff > (-1*vertuncert)),((diff < vertuncert)))
    print(vcondition)
    
    ptsL = ptsL[np.bitwise_and((diff < vertuncert) , diff > (-1*vertuncert))]
    ptsR = ptsR[np.bitwise_and((diff < vertuncert) , diff > (-1*vertuncert))]
    diff = diff[(diff < vertuncert) & diff > (-1*vertuncert)]
    print(np.unique(diff))
    # print("Before removing Duplicates",ptsL.shape,ptsL[0])
    # comptsLR=np.hstack((ptsL,ptsR))
    # _,indices = np.unique(comptsLR,axis=0,return_index=True)
 
    # ptsL = ptsL[indices]
    
    # ptsR = ptsR[indices]
    # print("After removing Duplicates",ptsL.shape,ptsL[0])
    
    return ptsL,ptsR


def siftkpts(rectL,rectR,fthresh):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(rectL, None)
    kp2, des2 = sift.detectAndCompute(rectR, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    matches = flann.knnMatch(des1, des2, k=2)

    # Keep good matches: calculate distinctive image features
    # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
    # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    matchesMask = [[0, 0] for i in range(len(matches))]
    ptsL = []
    ptsR = []

    for i, (m, n) in enumerate(matches):
        if m.distance < fthresh*n.distance:
            # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            ptsR.append(kp2[m.trainIdx].pt)
            ptsL.append(kp1[m.queryIdx].pt)

    ptsL = np.int32(ptsL)
    ptsR = np.int32(ptsR)
    return ptsL,ptsR
imgR = cv2.imread("E:/Stereo-3D-Reconstruction/captured/rectified/8_L_.png")
imgL  = cv2.imread("E:/Stereo-3D-Reconstruction/captured/rectified/8_R_.png")

ptsL,ptsR   = siftkpts(imgL,imgR,0.6)#customdisparity(dmLeft,dmRight,BS,NOD,0,WS)
print(ptsL[0:5],ptsR[0:5])
print("OrigKPTS",ptsL.shape)
ptsL,ptsR = filterbadkpoints(ptsL,ptsR)
print("FilteredKPTS",ptsL.shape)
print(ptsL[0:5],ptsR[0:5])
