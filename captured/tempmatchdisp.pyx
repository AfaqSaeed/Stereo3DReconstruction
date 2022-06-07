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
        
cpdef np.ndarray[np.int32_t ,ndim=2] keydisparity(unsigned char[:,:]imgL,unsigned char[:,:]imgR,int box_size,np.ndarray[np.int32_t ,ndim=2] ptsL, np.ndarray[np.int32_t ,ndim=2]ptsR):
    cdef int h,w,d,i,j
    cdef float total
    cdef np.ndarray[np.int32_t ,ndim=2] disparity 
    cdef np.ndarray[np.double_t,ndim = 1] height,width 
    cdef np.ndarray[np.int32_t,ndim = 1] disp
    h,w = imgL.shape[:2]
    disparity = np.zeros((h,w),dtype=np.int32)
    cdef np.ndarray[np.int64_t,ndim = 1] index
    cdef np.ndarray[np.double_t,ndim = 1] distance,ratios

    disp = ptsL[:,0]-ptsR[:,0]
    ptsL = ptsL[disp>0]
    ptsR = ptsR[disp>0]
    disp = disp[disp>0]
    

    width = np.arange(box_size,w-box_size,box_size,dtype=np.double)
    height = np.arange(box_size,h-box_size,box_size,dtype=np.double)
    
    for i in tqdm(height):
        for j in  width:

            distance = np.sqrt(np.square(ptsL[:,1]-i)+np.square(ptsL[:,0]-j))
        
            index = np.argsort(distance)
            
            index = index[np.sort(distance)!= 0]
            
            total = np.sum((1/distance[index[0:5]]))
            ratios = 1/distance[index[0:5]] 
            
            try:
                d = int(np.dot(disp[index[0:5]],ratios.T)/total )
                disparity[i-box_size:i+box_size,j-box_size:j+box_size]=d
            
            except:
                disparity[i-box_size:i+box_size,j-box_size:j+box_size]=d
            
        
            

            
    return disparity 




