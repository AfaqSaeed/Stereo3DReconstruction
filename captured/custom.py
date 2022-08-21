import cv2
import os

from kiwisolver import Constraint

from tempmatchdisp import keydisparity 
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button,TextBox, RectangleSelector
import numpy as np
import json
import stset as stset
import struct
from tempmatchdisp import customdisparity
from tqdm import tqdm

def write_pointcloud(output_points_sgbm_points,rgb_points,filename):

    """ creates a .pkl file of the point clouds generated
    """

    assert output_points_sgbm_points.shape[1] == 3,'Input output_points_sgbm points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(output_points_sgbm_points.shape).astype(np.uint8)*255
    assert output_points_sgbm_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input output_points_sgbm points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%output_points_sgbm_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(output_points_sgbm_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",output_points_sgbm_points[i,0],output_points_sgbm_points[i,1],output_points_sgbm_points[i,2],
                                        rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,2].tobytes())))
    fid.close()
scat  = None
svat = None
lgend = None
redpts = None
bluepts = None
BS = 5
WS = 30
NOD = 192
YS = 10
highf = 270
lowf = 190
K = 0.6
I = 0.0
SR = 10
Qscale=0.01
ptsL =[[]]
ptsR = [[]]
nptsL =[[]]
nptsR = [[]]

img_no = 8
imgL, imgR = cv2.imread(f"./test/left/{str(img_no)}_L_.png"), cv2.imread(f"./test/right/{str(img_no)}_R_.png")
print(imgL.shape[:2])
print('IMAGES LOADED')
print(100*'#')
vert, hori = imgL.shape[:2]
left_stereo_map, right_stereo_map, Q ,lcam_mtx,rcam_mtx= stset.st_maps("./calibrators/calibParams/", (hori, vert))
print('MAPS COMPUTED')
print(100*'#')
rectL, rectR = stset.st_rectify(imgL, imgR, left_stereo_map, right_stereo_map)
#rectL, rectR = cv2.imread("./singlerun/rectL2.png"), cv2.imread("./singlerun/rectR2.png")
print('RECTIFIED')
print(100*'#')
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
rectified_pair = (grayR, grayL)
axcolor = 'lightgoldenrodyellow'
fig = plt.figure(1)
fig.subplots_adjust(left=0.15, bottom=0.5)
axs = fig.subplots(1,3)

axs[0].imshow(rectR)

####################################################################
# Code For Rectangle Zoom inspector currently not working correctly     


# selector =[]

# def get_nearest(pts,pt):
#     diff = pts-pt
    
#     dist = np.linalg.norm(diff,axis=1)
#     indx = np.argsort(dist)
#     return indx[0]

# def select_callback(eclick, erelease):
#     """
#     Callback for line selection.

#     *eclick* and *erelease* are the press and release events.
#     """
#     global ptsL,ptsR,nptsL,nptsR,scat,svat,lgend,redpts,bluepts
#     x1, y1 = eclick.xdata, eclick.ydata
#     x2, y2 = erelease.xdata, erelease.ydata
#     x1, y1 = int(x1), int(y1)
#     x2, y2 = int(x2), int(y2)
#     print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
#     # Find Center of box
#     center = np.array([int((x1+x2)/2),int((y1+y2)/2)])
#     # find width and height of box
#     wby2,hby2 = int((x2-x1)/2),int((y2-y1)/2) # halved for conveinence in top-left point calculation from center in right image
#     print("Width , Height = " ,(wby2,hby2))
#     # Find Points that arre inside the box

#     checkptsL = (ptsL[:,0]>x1) & (ptsL[:,0]<x2) & (ptsL[:,1]>y1) & (ptsL[:,1]<y2)
#     redptsL = ptsL[checkptsL]
#     redptsR = ptsR[checkptsL]
#     # Find Point that is nearest to the center

#     rindx  = get_nearest(redptsL,center)
#     # calculate centers of box location in left and right image
#     Lcenter = redptsL[rindx]
#     Rcenter = redptsR[rindx]
#     print("Lcenter,Rcenter",Lcenter,Rcenter)

#     # crop in left image and right image of size (h,w)
#     Lcrop = rectL[Lcenter[1] - hby2:Lcenter[1] + hby2,Lcenter[0] - wby2:Lcenter[0] + wby2]
#     Rcrop = rectR[Rcenter[1] - hby2:Rcenter[1] + hby2,Rcenter[0] - wby2:Rcenter[0] + wby2]
#     # calcualte topleft points in order to change coordinate frames
#     topleftL = (x1,y1)
#     topleftR = Rcenter - (wby2,hby2)
#     # change cordinate frames
#     cropptsL = redptsL-topleftL
#     cropptsR = redptsR-(topleftR)+(wby2*2,0)
#     #combine two crops
#     comb = cv2.hconcat([Lcrop,Rcrop]) 
#     for ptl,ptr in zip(cropptsL,cropptsR):
#         color = list(np.random.random(size=3) * 256)
#         comb = cv2.circle(comb,ptl,1,color,1)
#         comb = cv2.circle(comb,ptr,1,color,1)
#         comb = cv2.line(comb ,ptl,ptr,color,1)
#     cv2.imshow("new",comb)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

        








# sfig = plt.figure(2)
# axs2 = sfig.subplots(2)

# selector_class = RectangleSelector
# axs2[0].imshow(rectL)
# axs2[1].imshow(rectR)  # plot something
# # axs[0].set_title(f"Click and drag to draw a {selector_class.__name__}.")
# selector.append(selector_class(
# axs2[0], select_callback,
#         useblit=True,
#         button=[1, 3],  # disable middle button
#         minspanx=5, minspany=5,
#         spancoords='pixels',
#         interactive=False))


# find the keypoints and descriptors with SIFT
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
# To keep variables in a specified range
def filterbadkpoints(ptsL,ptsR,epsilon=10,xrangel=None,xrangeh=None):
    
    # if low and high range provided remove negative disparity
    if xrangeh and xrangel:
        disp = ptsL[:,0]-ptsR[:,0]
        ptsL = ptsL[np.bitwise_and(disp<xrangeh,disp>xrangel)]
        ptsR = ptsR[np.bitwise_and(disp<xrangeh,disp>xrangel)]
    # remove any points that violate epipolar constraint
    yL = ptsL[:,1]
    yR = ptsR[:,1]
    
    diff = yL-yR
    ptsL = ptsL[np.bitwise_and((diff < epsilon) , diff > (-1*epsilon))]
    ptsR = ptsR[np.bitwise_and((diff < epsilon) , diff > (-1*epsilon))]
    # print("Before removing Duplicates",ptsL.shape,ptsL[0])
    comptsLR=np.hstack((ptsL,ptsR))
    _,indices = np.unique(comptsLR,axis=0,return_index=True)
 
    ptsL = ptsL[indices]
    
    ptsR = ptsR[indices]
    # print("After removing Duplicates",ptsL.shape,ptsL[0])
    
    return ptsL,ptsR
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
                filterbadkpoints(zptsl,zptsr,epsilon=5)
             except:
                 print("Getting more Kpts sift failed")
                 continue
             zptsl,zptsr = zptsl-win_size,zptsr-win_size
             
             nptsL.extend(ptsl+zptsl)
             nptsR.extend(ptsr+zptsr)
    
    nptsL = np.array(nptsL) 
    nptsR = np.array(nptsR)
     
    
    return nptsL,nptsR
def drawdisparity(imshape,nptsL,nptsR):
    disparity = np.zeros(imshape,dtype=np.int16)
    blob_size = BS
    for ptsl,ptsr in zip(nptsL,nptsR):
        d = ptsl[0]-ptsr[0]
        # if d>190 and d<270: 
        disparity[ptsl[1]-blob_size:ptsl[1]+blob_size,ptsl[0]-blob_size:ptsl[0]+blob_size] =  d#+disparity[ptsl[1]-blob_size:ptsl[1]+blob_size,ptsl[0]-blob_size:ptsl[0]+blob_size]
    return disparity

def kpdisp2ndpass(rectL,rectR,K):
    global ptsL,ptsR,nptsL,nptsR
    ptsL,ptsR = siftkpts(rectL,rectR,K)
    
    print("Points Before Filtering",ptsR.shape) 
    ptsL,ptsR = filterbadkpoints(ptsL,ptsR)
    print("Points After Filtering",ptsR.shape ) 
    nptsL,nptsR = getmorekeypoints(rectL,rectR,ptsL,ptsR,win_size=50,K=K,I=I)
    print("Additional Keypoints added",nptsR.shape) 
    nptsL,nptsR = filterbadkpoints(nptsL,nptsR,xrangel=190,xrangeh=270)
    print("Points After Filtering",nptsR.shape ) 
    
    nptsL = np.append(ptsL,nptsL,axis = 0)
    nptsR = np.append(ptsR,nptsR,axis = 0)
    print(nptsR.shape )                                          
    disparity = drawdisparity(rectL.shape,nptsL,nptsR)
    return disparity

def kpdisp1stpass(rectL,rectR,K):
    global ptsL,ptsR
    ptsL,ptsR = siftkpts(rectL,rectR,K)
    print("Points Before Filtering",ptsR.shape) 
    ptsL,ptsR = filterbadkpoints(ptsL,ptsR)
    print("Points After Filtering",ptsR.shape ) 
    # axs2[0].scatter(ptsL[:,0],ptsL[:,1],c="b",s=0.1,label="First")
    # axs2[1].scatter(ptsR[:,0],ptsR[:,1],c="r",s=0.1,label="First")
    
    disparity = drawdisparity(rectL.shape,ptsL,ptsR)
    return disparity

def submit(text):
    ydata = eval(text)
    global img_no,imgL,imgR,rectified_pair,tmdispL,tmdispR,rectL,rectR
    img_no = ydata
    imgL, imgR = cv2.imread(f"./test/left/{str(img_no)}_L_.png"), cv2.imread(f"./test/right/{str(img_no)}_R_.png")
    print(imgL.shape[:2])
    print('IMAGES LOADED')


    vert, hori = imgL.shape[:2]
    left_stereo_map, right_stereo_map, Q ,lcam_mtx,rcam_mtx = stset.st_maps("./calibrators/calibParams/", (hori, vert))
    print('MAPS COMPUTED')


    rectL, rectR = stset.st_rectify(imgL, imgR, left_stereo_map, right_stereo_map)
    print('RECTIFIED')
    print(100*'#')
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    rectified_pair = (grayR, grayL)
    axs[0].imshow(rectified_pair[0], 'gray')

    tmdispL,tmdispR= stereo_depth_map(rectified_pair)
    axs[1].imshow(tmdispL, aspect='equal', cmap='gray')
    axs[2].imshow(tmdispR, aspect='equal', cmap='jet')
    

axbox = fig.add_axes([0.27, 0.95, 0.15, 0.04])
text_box = TextBox(axbox, 'Image # ', initial="100")
text_box.on_submit(submit)


def stereo_depth_map(rectified_pair):

    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]

    print ('BS='+str(BS)+' WS='+str(WS)+' NOD='+str(NOD)+' YS='+\
           str(YS)+' High='+str(highf)+' LowFilt='+str(lowf))
    print (' K='+str(K)+' I='+str(I)+' SR='+str(SR))
    

    print("Keypoint matching Disparity")
    tmdispL   = kpdisp2ndpass(dmLeft,dmRight,K)#customdisparity(dmLeft,dmRight,BS,NOD,0,WS)

    tmdispR   = keydisparity(dmLeft,dmRight,BS,nptsL,nptsR)
    tmdispR   = cv2.blur(tmdispR,(5,5))
    # tmdispR  = np.zeros(dmLeft.shape[:2])

    return tmdispL,tmdispR

# Set up and draw interface

# Draw left image and depth map

tmdispL,tmdispR = stereo_depth_map(rectified_pair)



#  

axs[1].imshow(tmdispL, aspect='equal', cmap='jet')
axs[2].imshow(tmdispR, aspect='equal', cmap='jet')


saveax = fig.add_axes([0.3, 0.42, 0.15, 0.04]) #stepX stepY width height
buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')


def save_map_settings( event ): 
    buttons.label.set_text ("Saving...")
    print('Saving to file...') 
    result = json.dumps({'ImageNo':img_no,'blockSize':BS, 'WindowSize':WS, 'numDisparities':NOD, \
             'Yshift':YS, 'HighFilter':highf, 'LowFilter':lowf, \
             'Kvalue':K, 'I':I, 'SR':SR,'Qscale':Qscale},\
             sort_keys=True, indent=4, separators=(',',':'))
    fName = 'TMDisp3D.txt'
    f = open (str(fName), 'w') 
    f.write(result)
    f.close()
    buttons.label.set_text ("Save to file")
    print ('Settings saved to file '+fName)

buttons.on_clicked(save_map_settings)

loadax = fig.add_axes([0.1, 0.42, 0.15, 0.04]) #stepX stepY width height
buttonl = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')
def load_map_settings( event ):
    global loading_settings, BS, WS, NOD, YS, highf, lowf, K, I, SR
    loading_settings = 1
    fName = '3dmap_set.txt'
    print('Loading parameters from file...')
    buttonl.label.set_text ("Loading...")
    f=open(fName, 'r')
    data = json.load(f)
    sBS.set_val(data['blockSize'])
    sWS.set_val(data['WindowSize'])
    sNOD.set_val(data['numDisparities'])
    sYS.set_val(data['Yshift'])
    shighf.set_val(data['HighFilter'])
    slowf.set_val(data['LowFilter'])
    sK.set_val(data['Kvalue'])
    sI.set_val(data['I'])
    sSR.set_val(data['SR'])
    sQscale.set_val(data['Qscale'])
    f.close()
    buttonl.label.set_text ("Load settings")
    print ('Parameters loaded from file '+fName)
    print ('Redrawing depth map with loaded parameters...')
    loading_settings = 0
    update(0)
    print ('Done!')

buttonl.on_clicked(load_map_settings)


# Draw interface for adjusting parameters
print('Start interface creation (it takes up to 30 seconds)...')

SWSaxe = fig.add_axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
PFSaxe = fig.add_axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
PFCaxe = fig.add_axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
WSaxe = fig.add_axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
NODaxe = fig.add_axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
TTHaxe = fig.add_axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
YSaxe = fig.add_axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
SRaxe = fig.add_axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
SPWSaxe = fig.add_axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
Qscaleaxe = fig.add_axes([0.15, 0.37, 0.7, 0.025], facecolor=axcolor)
sBS = Slider(SWSaxe, 'BlockSize', 0.0, 100.0, valinit=5)
sWS = Slider(PFSaxe, 'WindowSize', 0, 100.0, valinit=30)
sNOD = Slider(PFCaxe, 'NumOfDisp', 0.0, 120, valinit=30)
sYS = Slider(WSaxe, 'Yrange', 1.0, 20.0, valinit=2)
slowf = Slider(TTHaxe, 'FLowThresh', 0, 200, valinit=190)
shighf = Slider(NODaxe, 'FHighThresh', int(slowf.val),400 , valinit=270)
sK = Slider(YSaxe, 'KPOrgThresh', 0.0, 1.0, valinit=0.6)
sI = Slider(SRaxe, 'KpIncThresh', 0.0, 1.0, valinit=0.0)
sSR = Slider(SPWSaxe, 'SR', 0.0, 120.0, valinit=10)
sQscale = Slider(Qscaleaxe, 'Q', 0.0000, 1.0000, valinit=0.01)

# Produce the colormap disparity image
def color_disparity_map(disparity):
    local_max = disparity.max()
    local_min = disparity.min()
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

    norm_coeff = 255 / disparity.max()
    return disparity * norm_coeff / 255, disparity_color

uax= fig.add_axes([0.5, 0.42, 0.15, 0.04])
buttonu = Button(uax, 'Update Map', color=axcolor, hovercolor='0.975')
def updatedepthmap(event):
        global dmObject,tmdispL,tmdispR
        print ('Rebuilding depth map')
        buttonu.label.set_text("Updating")
        update(0)
        tmdispL,tmdispR= stereo_depth_map(rectified_pair)
        
        
        axs[1].imshow(tmdispL, aspect='equal', cmap='gray')
        axs[2].imshow(tmdispR, aspect='equal', cmap='jet')


        print('Saving disp map!')
        # disp, cdisp = color_disparity_map(disparity)
        # cv2.imwrite('./disp.png', cdisp)
        print ('Redraw depth map')

        plt.draw()
        buttonu.label.set_text("Updated")

buttonu.on_clicked(updatedepthmap)
root = "./calibrators/calibParams/"
R = np.load(root+"R.npy")
T = np.load(root+"T.npy")
D1 = np.load(root+"dLS.npy")
D2 = np.load(root+"dRS.npy")
def reconstruct3D(event):
    # ax = plt.axes(projection='3d')
    global tmdispL,tmdispR
    
    plt.figure(3,constrained_layout=True)
    button3D.label.set_text("Reconstructing")
    
    
    realdisp = np.float32(tmdispR)
    disp2nd =  np.float32(tmdispL)
    plt.subplot(1,2,1)
    plt.imshow(realdisp)
    otherpoints3d = cv2.reprojectImageTo3D(disp2nd, Q.astype(np.float32),handleMissingValues=False,ddepth=-1)
    other_points = otherpoints3d[np.bitwise_and(disp2nd!=0,disp2nd>190)]#otherpoints[realdisp!=0]
    
    print(Q)
    points_3D_sgbm = cv2.reprojectImageTo3D(realdisp, Q.astype(np.float32),handleMissingValues=False,ddepth=-1)
    z =  points_3D_sgbm[:,:,2]
   
    plt.subplot(1,2,2)
    plt.imshow(z)
    plt.show()
    
    output_points_sgbm = points_3D_sgbm[np.bitwise_and(realdisp!=0,realdisp>190)]#otherpoints[realdisp!=0]
    z=output_points_sgbm[:,2]
    print("Disparity Average",(np.average(realdisp[realdisp!=0])))
    print("Average",(np.average(z)))
    print("Median",(np.median(z))) 
    print("STD",(np.std(z))) 
    
    colors = cv2.cvtColor(rectR, cv2.COLOR_BGR2RGB)
    
    output_colors = colors[np.bitwise_and(realdisp!=0,realdisp>190)]
    other_colors = colors[np.bitwise_and(disp2nd!=0,disp2nd>190)]
   
    other_file = f'2ndPass {str(img_no)}.ply'

    output_file_sgbm = f'1stPass {str(img_no)}.ply'
    
    print(output_points_sgbm)
    print (" Creating the output file... ")
    write_pointcloud(output_points_sgbm, output_colors, output_file_sgbm)
    print (f"\n  output file {output_file_sgbm} created. \n")
    write_pointcloud(other_points, other_colors, other_file)
    print (f"\n  output file {other_file} created. \n")
    
    button3D.label.set_text("Reconstructed")



ax3D= fig.add_axes([0.7, 0.42, 0.15, 0.04])
button3D = Button(ax3D, '3DRecon', color=axcolor, hovercolor='0.975')

button3D.on_clicked(reconstruct3D)


# Update depth map parameters and redraw
def update(val):
    global loading_settings, BS, WS, NOD, YS, highf, lowf, K, I, SR,Qscale
    BS = int(sBS.val/2)*2+1 #convert to ODD   
    WS = int(sWS.val)    
    NOD = int(sNOD.val/16)*16
    YS = int(sYS.val)  
    highf= int(shighf.val)
    lowf = int(slowf.val)
    I = float(sI.val)
    SR = int(sSR.val)
    Qscale = float(sQscale.val)
    K =float(sK.val) 


    
# Connect update actions to control elements
sBS.on_changed(update)
sWS.on_changed(update)
sNOD.on_changed(update)
sYS.on_changed(update)
shighf.on_changed(update)
slowf.on_changed(update)
sI.on_changed(update)
sSR.on_changed(update)
sK.on_changed(update)
sQscale.on_changed(update)

print('Show interface to user')
plt.show()
