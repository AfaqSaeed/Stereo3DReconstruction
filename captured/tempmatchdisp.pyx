from this import d
import cv2
import numpy as np
cimport numpy as np
from tqdm import tqdm
import os

cpdef int[:,:] customdisparity(unsigned char[:,:]imgL,unsigned char[:,:]imgR,int box_size,int range,int direction):
    cdef int h,w,d,i,j,x
    cdef (float,float,int*,int*) maxval
    cdef int [:,:] disparityL,disparityR 
    cdef unsigned char[:,:] template
    cdef unsigned char[:,:] searchstrip
    cdef np.ndarray[np.float32_t, ndim=2] result
    cdef np.ndarray[np.double_t,ndim = 1] height,width 
    
    h,w = imgL.shape[:2]
    disparityL = np.zeros((h,w),dtype=np.int32)
    disparityR = np.zeros((h,w),dtype=np.int32)
    #disparity = np.zeros((h,w,2),dtype=np.int32)
    
    width = np.arange(box_size,w-box_size,box_size,dtype=np.double)
    height = np.arange(box_size,h-box_size,box_size,dtype=np.double)

    #print(imgL.shape)
    for i in tqdm(height):
        for j in  width:

            if (j+box_size<range):
                x=0
            else:
                x = (j+box_size) - range
            
            template = imgL[i-box_size:i+box_size,j-box_size:j+box_size]
            searchstrip = imgR[i-box_size:i+box_size,(j-box_size):(j+box_size)+x]
            result = cv2.matchTemplate(np.uint8(searchstrip),np.uint8(template),cv2.TM_CCOEFF_NORMED)
            m,n,o,p=cv2.minMaxLoc(result)
#           print(template.shape)
#           print(searchstrip.shape)

            
            d = j-(x+p[0]+box_size)
            disparityL[i-box_size:i+box_size,j-box_size:j+box_size]=d
            # disparity from Right to Left image
            
     #       j = (x+p[0]+box_size) # j becomes the mathed point here and we search back from there i is same
            #print(j,p[0])
            if (j+box_size<range):
                x=0
            else:
                x = (j+box_size) - range
            
            # L and R are reversed 
            template = imgR[i-box_size:i+box_size,j-box_size:j+box_size]
            searchstrip = imgL[i-box_size:i+box_size,(j-box_size)-x:(j+box_size)]
           #print(template.shape)
           #Print(searchstrip.shape)
            result = cv2.matchTemplate(np.uint8(searchstrip),np.uint8(template),cv2.TM_CCOEFF_NORMED)
            m,n,o,p=cv2.minMaxLoc(result)
            

            d = j-(x+p[0]+box_size)
            disparityR[i-box_size:i+box_size,j-box_size:j+box_size] = d

            #disparity[:,:,0] = disparityL
            #disparity[:,:,1] = disparityR
    if direction ==0:
        return disparityL
    else:
        return disparityR    
