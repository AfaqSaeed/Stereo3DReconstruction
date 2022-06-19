from this import d
import cv2
import numpy as np
cimport numpy as np
from tqdm import tqdm
import os

cpdef int[:,:] customdisparity(unsigned char[:,:]imgL,unsigned char[:,:]imgR,int box_size,int range,int direction,int interval):
    cdef int h,w,d,i,j,x
    cdef (float,float,int*,int*) maxval
    cdef int [:,:] disparity 
    cdef unsigned char[:,:] template
    cdef unsigned char[:,:] searchstrip
    cdef np.ndarray[np.float32_t, ndim=2] result
    cdef np.ndarray[np.double_t,ndim = 1] height,width 
    
    h,w = imgL.shape[:2]
    disparity = np.zeros((h,w),dtype=np.int32)
    
    width = np.arange(box_size,w-interval,interval,dtype=np.double)
    height = np.arange(box_size,h-interval,box_size,dtype=np.double)

    for i in tqdm(height):
        for j in  width:
            if direction ==0:
    
                if (j+box_size<range):
                    x=0
                else:
                    x = (j+box_size) - range
                
                template = imgL[i-box_size:i+box_size,j-box_size:j+box_size]
                searchstrip = imgR[i-box_size:i+box_size,:] # (j-box_size):(j+box_size)+x]
                result = cv2.matchTemplate(np.uint8(searchstrip),np.uint8(template),cv2.TM_CCOEFF_NORMED)
                m,n,o,p=cv2.minMaxLoc(result)
    #           print(template.shape)
    #           print(searchstrip.shape)

                
                d = j-(x+p[0]+box_size)
                disparity[i-box_size:i+box_size,j-box_size:j+box_size]=d
            # disparity from Right to Left image
            else:
                if (j+box_size<range):
                    x=0
                else:
                    x = (j+box_size) - range
                
                # L and R are reversed 
                template = imgR[i-box_size:i+box_size,j-box_size:j+box_size]
                searchstrip = imgL[i-box_size:i+box_size,:]#(j-box_size)-x:(j+box_size)]
                result = cv2.matchTemplate(np.uint8(searchstrip),np.uint8(template),cv2.TM_CCOEFF_NORMED)
                m,n,o,p=cv2.minMaxLoc(result)
                

                d = (x+p[0]+box_size)-j
                disparity[i-box_size:i+box_size,j-box_size:j+box_size] = d

    return disparity    
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


cpdef int get_nearest(np.ndarray[np.int32_t ,ndim=2] pts, np.ndarray[np.int32_t ,ndim=1] pt):
    cdef np.ndarray[np.int32_t ,ndim=2] diff
    cdef np.ndarray[np.double_t ,ndim=1] dist
    cdef np.ndarray[np.longlong_t ,ndim=1] indx
    diff = pts-pt
    
    dist = np.linalg.norm(diff,axis=1)
    indx = np.argsort(dist)
    return int(indx[0])



cpdef np.ndarray[np.int32_t ,ndim=2] keydisparity(unsigned char[:,:]imgL,unsigned char[:,:]imgR,int box_size,np.ndarray[np.int32_t ,ndim=2] ptsL, np.ndarray[np.int32_t ,ndim=2]ptsR,int win_size,int searchrange,int yshift,int minclip,int maxclip):
    cdef int h,w,d,i,j
    cdef float total
    cdef np.ndarray[np.int32_t ,ndim=2] disparity 
    cdef np.ndarray[np.double_t,ndim = 1] height,width 
    cdef np.ndarray[np.int32_t,ndim = 1] disp
    h,w = imgL.shape[:2]
    disparity = np.zeros((h,w),dtype=np.int32)
    cdef np.ndarray[np.int32_t,ndim = 1] point
    cdef int index 
    cdef np.ndarray[np.double_t,ndim = 1] distance,ratios

    disp = ptsL[:,0]-ptsR[:,0]
    ptsL = ptsL[disp>0]
    ptsR = ptsR[disp>0]
    disp = disp[disp>0]
    

    width = np.arange(box_size,w-box_size,box_size,dtype=np.double)
    height = np.arange(box_size,h-box_size,box_size,dtype=np.double)
    
    for i in tqdm(height):
        for j in  width:
            point = np.array([j,i])
            index = get_nearest(ptsL,point)
            d = disp[index]
            try:
                d = matchtemplate(imgL,imgR,point,d,win_size,searchrange)
            except:
                pass 
            if d>minclip and d<maxclip:

                disparity[i-(yshift+box_size):i+(yshift+box_size),j-box_size:j+box_size] = d
            
    return disparity 




