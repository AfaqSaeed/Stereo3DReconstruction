from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
pts = np.array([[1,2],[3,4],[5,6],[1,8],[4,3],[8,5]])
pt = np.array([4,5])
# fig=plt.figure()
# ax = fig.subplots(2,2,sharex="row",sharey ="row")
        
            
rands = np.random.randint(0,255,size=(10,10))
def get_nearest(pts,pt):
    diff = pts-pt
    
    dist = np.linalg.norm(diff,axis=1)
    indx = np.argsort(dist)
    return indx[0]

ind = get_nearest(pts,pt)
ptx = pts[ind]

# ax.scatter(pts[:,0],pts[:,1],label="List Data")
# ax.scatter(pt[0],pt[1],label="Point")
# ax.scatter(ptx[0],ptx[1],label="Nearest Point")

# ax.legend()
# plt.show()
#########################################################################
############ Disparity Calculation with Restricted Range#################
#########################################################################   
correct_d = 217
win_size = 7
i = 939
j = 950
d = 216
searchrange = 5

imgR = cv2.imread("E:/Stereo-3D-Reconstruction/captured/rectified/8_L_.png")
imgL  = cv2.imread("E:/Stereo-3D-Reconstruction/captured/rectified/8_R_.png")
# https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
# mask = np.zeros_like(imgL)
# imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
# imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

# # Sets image saturation to maximum
# # Sts image saturation to maximum
# mask[..., 1] = 255
# mask[..., 1] = 255    
# flow = cv2.calcOpticalFlowFarneback(imgL, imgR, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# # Computes the magnitude and angle of the 2D vectors
# # Computes the magnitude and angle of the 2D vectors
# magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# # Sets image hue according to the optical flow direction
# # Sets image hue according to the optical flow direction
# mask[..., 0] = angle * 180 / np.pi / 2

# # Sets image value according to the optical flow magnitude (normalized)
# # Sets image value according to the optical flow magnitude (normalized)
# mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
# mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
# # Converts HSV to RGB (BGR) color representation
# # Converts HSV to RGB (BGR) color representation
# rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

# # Opens a new window and displays the Soutput frame
# # Opens a new window and displays the output frame
# rgb = cv2.resize(rgb,(800,600))
# cv2.imshow("dense optical flow", rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
def matchtemplate(imgL,imgR,pt,d,win_size,searchrange):
    (j,i) = pt
    searcharea = searchrange + win_size 
    template = imgL[i-win_size:i+win_size,j-win_size:j+win_size]
    searchstrip = imgR[i-win_size:i+win_size,(j-d)-searcharea:(j-d)+(searcharea)]
    result  = cv2.matchTemplate(np.uint8(searchstrip),np.uint8(template),cv2.TM_CCOEFF_NORMED)
    
    result = np.reshape(result,result.shape[1])
    # print(result.shape)
    sort = np.argsort(-1*result)
    top3 = sort[:3] 
    # print("Top3",top3)
    topdisp = (d - searchrange)+ top3
    # print("TopDisp",topdisp)
    
    distfromd = abs(topdisp-d)
    # print("DisttfrmD",distfromd)
    bestindx = np.argsort(distfromd)
    # print("bestindx",bestindx)
    best = topdisp[bestindx[0]]
    
    return best
# bottom_rightR = (p[0] + win_size,p[1] + win_size) 
# top_leftL = (j-int(win_size/2),i-int(win_size/2)) 
# bottom_rightL = (j + int(win_size/2),i + int(win_size/2)) 
# leftstrip = imgL[i-win_size:i+win_size,:]
# searchstrip = cv2.rectangle(searchstrip.copy(),p,(p[0]+2*win_size,p[1]+2*win_size),(0,0,255),2)
# leftstrip = cv2.rectangle(leftstrip.copy(),(j-win_size,0),(j+win_size,2*win_size),(0,255,0),2)
# crop = searchstrip[0:p[1]+2*win_size,p[0]:p[0]+2*win_size] 
# comb = cv2.vconcat([template,crop])  
# searchstrip = cv2.vconcat([leftstrip,searchstrip]) 

# rectL = cv2.rectangle(imgL.copy(),top_leftL, bottom_rightL, 255, 2)
# rectR = cv2.rectangle(imgR.copy(),top_leftR, bottom_rightR, 255, 2)

# ax[0][0].imshow(rectL)
# ax[0][1].imshow(rectR)

# 
# distance = cv2.magnitude(ptsL-(j,i))
#    index = np.argsort(distance)          
#             index = index[np.sort(distance)!= 0]
            
#             total = np.sum((1/distance[index[0:5]]))
#             ratios = 1/distance[index[0:5]] 
            
#             try:
#                 d = int(np.dot(disp[index[0:5]],ratios.T)/total )
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
# To
def filterbadkpoints(ptsL,ptsR,epsilon=5,rneg=False):
    
    # remove negative disparity
    if rneg:
        disp = ptsL[:,0]-ptsR[:,0]
        ptsL = ptsL[disp>0]
        ptsR = ptsR[disp>0]
    # remove any points that violate epipolar constraint
    yL = ptsL[:,1]
    yR = ptsR[:,1]
    
    diff = yL-yR
    ptsL = ptsL[(diff < epsilon) & diff > (-1*epsilon)]
    ptsR = ptsR[(diff < epsilon) & diff > (-1*epsilon)]
    # print("Before removing Duplicates",ptsL.shape,ptsL[0])
    # comptsLR=np.hstack((ptsL,ptsR))
    # _,indices = np.unique(comptsLR,axis=0,return_index=True)
 
    # ptsL = ptsL[indices]
    
    # ptsR = ptsR[indices]
    # print("After removing Duplicates",ptsL.shape,ptsL[0])
    
    return ptsL,ptsR

def matchtemplate(imgL,imgR,pt,d,win_size,searchrange):
    (j,i) = pt
    searcharea = searchrange + win_size 
    template = imgL[i-win_size:i+win_size,j-win_size:j+win_size]
    searchstrip = imgR[i-win_size:i+win_size,(j-d)-searcharea:(j-d)+(searcharea)]
    result  = cv2.matchTemplate(np.uint8(searchstrip),np.uint8(template),cv2.TM_CCOEFF)

    result = np.reshape(result,result.shape[1])
    # print(result.shape)
    sort = np.argsort(-1*result)
    top3 = sort[:3] 
    # print("Top3",top3)
    topdisp = (d - searchrange)+ top3
    # print("TopDisp",topdisp)
    
    distfromd = abs(topdisp-d)
    # print("DisttfrmD",distfromd)
    bestindx = np.argsort(distfromd)
    # print("bestindx",bestindx)
    best = topdisp[bestindx[0]]
    
    return best

def keydisparity(imgL,imgR,box_size, ptsL, ptsR, win_size, searchrange):
    h,w = imgL.shape[:2]
    disparity = np.zeros((h,w),dtype=np.int32)
 
    disp = ptsL[:,0]-ptsR[:,0]
    ptsL = ptsL[disp>0]
    ptsR = ptsR[disp>0]
    disp = disp[disp>0]
    

    width = np.arange(box_size,w-box_size,box_size,int)
    height = np.arange(box_size,h-box_size,box_size,int)
    
    for i in tqdm(height):
        for j in  width:
            point = np.array([j,i])
        
            index = get_nearest(ptsL,point)
            d = disp[index]
            try:
                d = matchtemplate(imgL,imgR,point,d,win_size,searchrange)
            except:
                pass 
            disparity[i-box_size:i+box_size,j-box_size:j+box_size] = d
            
    return disparity 
def bound(var,limit,max):
    if max:
        if var>limit:
            var=limit
    else:
        if var<limit:
            var = limit
    return var               

def getmorekeypoints(rectL,rectR,ptsL,ptsR,win_size,K,I):
    nptsL = []
    nptsR = []
    for ptsl,ptsr in tqdm(zip(ptsL,ptsR),total =ptsL.shape[0]):
             a,b = ptsl[1] - win_size,ptsl[1] + win_size
             c,d = ptsl[0] - win_size,ptsl[0] + win_size
             e,f = ptsr[1] - win_size,ptsr[1] + win_size
             g,h = ptsr[0] - win_size,ptsr[0] + win_size
             a,b = bound(a,0,False),bound(b,1200,True)
             c,d = bound(c,0,False),bound(d,1600,True)
             e,f = bound(e,0,False),bound(f,1200,True)
             g,h = bound(g,0,False),bound(h,1600,True)
             left  = rectL[a:b,c:d]
             
             right = rectR[e:f,g:h]
            #  print(left.shape)
            #  print(right.shape)
             if left.shape != right.shape:
                 continue
             try:
                zptsl,zptsr = siftkpts(left,right,K+I)
                fptsl,fptsr = filterbadkpoints(zptsl,zptsr,rneg=False)
                
             except:
                 print("Getting more Kpts sift failed")
                 continue
             
             sleft  = cv2.circle(left.copy(),fptsl[0],2,(255,0,0),1)
             sright = cv2.circle(right.copy(),fptsr[0],2,(255,0,0),1)
             plt.subplot(1,2,1)
             plt.imshow(sleft)             
             plt.subplot(1,2,2)
             plt.imshow(sright)
             plt.show()
             
             zptsl,zptsr = zptsl-win_size,zptsr-win_size
             
             nptsL.extend(ptsl+zptsl)
             nptsR.extend(ptsr+zptsr)
    
    nptsL = np.array(nptsL) 
    nptsR = np.array(nptsR)

    return nptsL,nptsR

ptsL,ptsR   = siftkpts(imgL,imgR,0.6)#customdisparity(dmLeft,dmRight,BS,NOD,0,WS)
ptsL,ptsR = filterbadkpoints(ptsL,ptsR)
ptsL,ptsR   = getmorekeypoints(imgL,imgR,ptsL,ptsR,30,0.6,0.3)#customdisparity(dmLeft,dmRight,BS,NOD,0,WS)

# disp = keydisparity(imgL,imgR,3,ptsL,ptsR,30,0)
# plt.imshow(disp,cmap = "rainbow")
# plt.show()
