
def mapfunc(i,j):
    
    global imgL,imgR,box_size
    print(i,j)
    template = imgL[i:i+2*box_size,j:j+2*box_size]
    searchstrip = imgR[i:i+2*box_size,:]
    # print(searchstrip.shape)
    # print(template.shape)
    result = cv2.matchTemplate(searchstrip,template,cv2.TM_CCOEFF_NORMED)
    minval,maxval,minLoc,maxLoc=cv2.minMaxLoc(result)
 


def customdispari(imgL,imgR):
    x,y = np.indices((imgL.shape[0],imgL.shape[1]))
    vecfunc = np.vectorize(mapfunc)
    ufunc = np.frompyfunc(mapfunc,2,1)
    disparity = np.zeros(imgL.shape,dtype=np.uint8)
    disparity = ufunc(x,y)
    print(disparity)
    cv2.imshow("N",disparity.astype(np.uint8))    


