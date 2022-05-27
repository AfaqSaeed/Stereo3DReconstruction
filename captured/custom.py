import cv2
import os
from tempmatchdisp import customdisparity 
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button,TextBox
import numpy as np
import json
import stereo_setting as stset
import struct
# from tempmatchdisp import customdisparity
from tqdm import tqdm
box_size=30
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


# Rectifying the images

# Depth map function
'''
cv2.StereoSGBM_create(
    minDisparity=None, 
    numDisparities=None, 
    blockSize=None, 
    P1=None, 
    P2=None, 
    disp12MaxDiff=None, 
    preFilterCap=None, 
    uniquenessRatio=None, 
    speckleWindowSize=None, 
    speckleRange=None, 
    mode=None)
'''
BS = 7
MDS = 1
NOD = 192
UR = 10
SPWS = 100
SR = 32
DMD = 5
P1 = 8*3*BS**2
P2 = 32*3*BS**2 
Qscale=0.01

img_no = 119
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
fig = plt.subplots(1,2)
plt.subplots_adjust(left=0.15, bottom=0.5)
plt.subplot(1,4,1)

cyan = np.zeros((grayL.shape[0],grayL.shape[1],3),dtype=np.uint8)
red = np.zeros((grayL.shape[0],grayL.shape[1],3),dtype=np.uint8)
cyan[:,:,0] = np.zeros((grayL.shape),dtype=np.uint8)
cyan[:,:,1] = grayR
cyan[:,:,2] = grayR

red[:,:,0] = grayL
red[:,:,1] = np.zeros((grayL.shape[:2]),dtype=np.uint8)
red[:,:,2] = np.zeros((grayL.shape[:2]),dtype=np.uint8)


Anaglyph = red+cyan 

dmObject = plt.imshow(Anaglyph)


def keypointdisparity(rectL,rectR):
    sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(rectL, None)
    kp2, des2 = sift.detectAndCompute(rectR, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=15)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Keep good matches: calculate distinctive image features
    # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
    # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    ptsL = []
    ptsR = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            good.append(m)
            ptsR.append(kp2[m.trainIdx].pt)
            ptsL.append(kp1[m.queryIdx].pt)

    kp_img = cv2.drawMatches(rectL, kp1, rectR, kp2, good, None,matchColor=[0,255,0], flags=2)
   
    ptsL = np.int32(ptsL)
    ptsR = np.int32(ptsR)
    kp_img = cv2.resize(kp_img,(16))
    cv2.imshow("w",kp_img)
    # remove any points that are not nearly horizontal
    yL = ptsL[:,1]
    yR = ptsR[:,1]
    
    diff = yL-yR
    # ptsL = ptsL[diff < 10]
    # ptsR = ptsR[diff < 10]
                                               
    disparity = np.zeros(imgL.shape[:2],dtype=np.int16)
    blob_size = 7
    for ptsl,ptsr in zip(ptsL,ptsR):
        d = ptsr[0]-ptsl[0]
        disparity[ptsl[0]-blob_size:ptsl[0]+blob_size,ptsl[1]-blob_size:ptsl[1]+blob_size] =  d
    
    return disparity


def submit(text):
    ydata = eval(text)
    global img_no,imgL,imgR,rectified_pair,dmObject,actual
    img_no = ydata
    imgL, imgR = cv2.imread(f"./test/left/{str(img_no)}_L_.png"), cv2.imread(f"./test/right/{str(img_no)}_R_.png")
    print(imgL.shape[:2])
    print('IMAGES LOADED')
    print(100*'#')

    vert, hori = imgL.shape[:2]
    left_stereo_map, right_stereo_map, Q ,lcam_mtx,rcam_mtx = stset.st_maps("./calibrators/calibParams/", (hori, vert))
    print('MAPS COMPUTED')
    print(100*'#')

    rectL, rectR = stset.st_rectify(imgL, imgR, left_stereo_map, right_stereo_map)
    #rectL, rectR = cv2.imread("./singlerun/rectL2.png"), cv2.imread("./singlerun/rectR2.png")
    print('RECTIFIED')
    print(100*'#')
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    rectified_pair = (grayR, grayL)
    plt.subplot(1,4,1)
    dmObject = plt.imshow(rectified_pair[0], 'gray')

    tmdispL,tmdispR= stereo_depth_map(rectified_pair)
    #,tmdisp,tmdisp_visual 
    plt.subplot(1,4,2)
    dmObject = plt.imshow(tmdispL, aspect='equal', cmap='jet')
    plt.subplot(1,4,3)
    dmObject = plt.imshow(tmdispR, aspect='equal', cmap='jet')
    

axbox = plt.axes([0.27, 0.92, 0.15, 0.04])
text_box = TextBox(axbox, 'Image # ', initial="100")
text_box.on_submit(submit)


def stereo_depth_map(rectified_pair):

    dmRight = rectified_pair[0]
    dmLeft = rectified_pair[1]

    print("Keypoint matching Disparity")
    tmdispL   = keypointdisparity(dmLeft,dmRight)

    tmdispL  = np.asarray(tmdispL)

    tmdispR   = customdisparity(dmLeft,dmRight,5,400,0)
    tmdispR  = np.asarray(tmdispR)

    return tmdispL,tmdispR

# Set up and draw interface

# Draw left image and depth map

tmdispL,tmdispR = stereo_depth_map(rectified_pair)
#  
plt.subplot(1,3,2)
dmObject = plt.imshow(tmdispL, aspect='equal', cmap='jet')
plt.subplot(1,3,3)
dmObject = plt.imshow(tmdispR, aspect='equal', cmap='jet')


saveax = plt.axes([0.3, 0.42, 0.15, 0.04]) #stepX stepY width height
buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')


def save_map_settings( event ): 
    buttons.label.set_text ("Saving...")
    print('Saving to file...') 
    result = json.dumps({'ImageNo':img_no,'blockSize':BS, 'minDisparity':MDS, 'numDisparities':NOD, \
             'uniquenessRatio':UR, 'speckleWindowSize':SPWS, 'speckleRange':SR, \
             'disp12MaxDiff':DMD, 'P1':P1, 'P2':P2,'Qscale':Qscale},\
             sort_keys=True, indent=4, separators=(',',':'))
    fName = 'TMDisp3D.txt'
    f = open (str(fName), 'w') 
    f.write(result)
    f.close()
    buttons.label.set_text ("Save to file")
    print ('Settings saved to file '+fName)

buttons.on_clicked(save_map_settings)

loadax = plt.axes([0.1, 0.42, 0.15, 0.04]) #stepX stepY width height
buttonl = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')
def load_map_settings( event ):
    global loading_settings, BS, MDS, NOD, UR, SPWS, SR, DMD, P1, P2
    loading_settings = 1
    fName = '3dmap_set.txt'
    print('Loading parameters from file...')
    buttonl.label.set_text ("Loading...")
    f=open(fName, 'r')
    data = json.load(f)
    sBS.set_val(data['blockSize'])
    sMDS.set_val(data['minDisparity'])
    sNOD.set_val(data['numDisparities'])
    sUR.set_val(data['uniquenessRatio'])
    sSPWS.set_val(data['speckleWindowSize'])
    sSR.set_val(data['speckleRange'])
    sDMD.set_val(data['disp12MaxDiff'])
    sP1.set_val(data['P1'])
    sP2.set_val(data['P2'])
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

SWSaxe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
PFSaxe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
PFCaxe = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
MDSaxe = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
NODaxe = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
TTHaxe = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
URaxe = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
SRaxe = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
Qscaleaxe = plt.axes([0.15, 0.37, 0.7, 0.025], facecolor=axcolor)
sBS = Slider(SWSaxe, 'BlockSize', 5.0, 255.0, valinit=5)
sMDS = Slider(PFSaxe, 'MinDisp', -600.0, 100.0, valinit=5)
sNOD = Slider(PFCaxe, 'NumOfDisp', 16.0, 640.0, valinit=16)
sUR = Slider(MDSaxe, 'UnicRatio', 1.0, 20.0, valinit=2)
sSPWS = Slider(NODaxe, 'SpklWinSze', 0.0, 1.0, valinit=0.95)
sSR = Slider(TTHaxe, 'SpcklRng', 0.0, 1.0, valinit=0.15)
sDMD = Slider(URaxe, 'DispMaxDiff', 1.0, 20.0, valinit=10)
sP1 = Slider(SRaxe, 'P_1', 0.0, 5000.0, valinit=15)
sP2 = Slider(SPWSaxe, 'P_2', 0.0, 5000.0, valinit=15)
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
#enddef

uax= plt.axes([0.5, 0.42, 0.15, 0.04])
buttonu = Button(uax, 'Update Map', color=axcolor, hovercolor='0.975')
def updatedepthmap(event):
        global dmObject,actual
        print ('Rebuilding depth map')
        buttonu.label.set_text("Updating")
        update(0)
        actual,tmdispL,tmdispR= stereo_depth_map(rectified_pair)
        depth = 300*4/np.float32(np.divide(actual,16))
        
        plt.subplot(1,4,2)
        dmObject = plt.imshow(actual, aspect='equal', cmap='jet')
        plt.subplot(1,4,3)
        dmObject = plt.imshow(tmdispL, aspect='equal', cmap='jet')
        plt.subplot(1,4,4)
        dmObject = plt.imshow(tmdispR, aspect='equal', cmap='jet')


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
    button3D.label.set_text("Reconstructing")
    # output_points_sgbm = cv2.omnidir.stereoReconstruct(imgL,imgR,lcam_mtx,rcam_mtx)
    focal_length = 4
    Q2 = np.float32([[1,0,0,0],
    [0,-1,0,0],
    [0,0,focal_length*Qscale,0], #Focal length multiplication obtained experimentally. 
    [0,0,0,1]])
    # mask_map = np.ones(actual.shape[:2],dtype=np.bool)
    
    realdisp = np.float32(np.divide(actual,16))
    
    # Homo = np.ones((actual.shape[0],actual.shape[1]),dtype=np.float32)  
    fx,fy = lcam_mtx[0,0],lcam_mtx[1,1]

    f = np.sqrt(np.square(fx)+np.square(fy))
    Z = 300*f/-realdisp
    Z = np.where(np.isinf(Z),0,Z)
    XY=np.indices((Z.shape))
    yl,xl=XY[0],XY[1]
    X = xl*Z/fx
    Y = yl*Z/fy 
    otherpoints = np.zeros((realdisp.shape[0],realdisp.shape[1],3),dtype=np.float32)
    otherpoints[:,:,0] = X
    otherpoints[:,:,1] = Y
    otherpoints[:,:,2] = Z     
    print(lcam_mtx)
    # print(rcam_mtx)
    # Q[2][3] = 0.004
    # Q[3][2]=1/-0.03
    # Q[3,3] =lcam_mtx[0][2]-rcam_mtx[0][2]/-0.03 
    print(Q)
    points_3D_sgbm = cv2.reprojectImageTo3D(realdisp, Q.astype(np.float32),handleMissingValues=False,ddepth=-1)
    # print (points_3D_sgbm.shape,points_3D_sgbm.dtype)
    output_points_sgbm = points_3D_sgbm[realdisp != 0]#otherpoints[realdisp!=0]
    
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    # output_points_sgbm = otherpoints[mask_map]
    
    output_colors = colors[realdisp!=0]
    # output_points = np.where(np.isinf(output_points_sgbm),0,output_points_sgbm)
    # output_points_sgbm = output_points_sgbm[np.invert(np.isinf(output_points_sgbm))]
    # print(output_points_sgbm.shape)
    # # (output_points[:,2]<-6.5e+3)&
    # # (output_points[:,2]<-6.5e+3)

    # z = output_points[:,2]
    # y = sorted(z)
    # x = np.arange(0,len(y))
    # plt.figure()
    # plt.hist(y,bins=50)
    # plt.show()
    
    output_file_sgbm = f'SGBM{str(img_no)}.ply'
    
    print(output_points_sgbm)
    # ax.scatter(output_points_sgbm[:,0], output_points_sgbm[:,1], output_points_sgbm[:,2], c = output_colors/255, s=0.01)
    print (" Creating the output file... ")
    write_pointcloud(output_points_sgbm, output_colors, output_file_sgbm)
    print (f"\n  output file {output_file_sgbm} created. \n")
    button3D.label.set_text("Reconstructed")

    # plt.figure(2)
    # ax = plt.axes(projection='3d')
    # ax.scatter(output_points_sgbm[:,0], output_points_sgbm[:,1], output_points_sgbm[:,2], c = output_colors/255, s=0.01)




ax3D= plt.axes([0.7, 0.42, 0.15, 0.04])
button3D = Button(ax3D, '3DRecon', color=axcolor, hovercolor='0.975')

button3D.on_clicked(reconstruct3D)


# Update depth map parameters and redraw
def update(val):
    global loading_settings, BS, MDS, NOD, UR, SPWS, SR, DMD, P1, P2,Qscale
    BS = int(sBS.val/2)*2+1 #convert to ODD   
    MDS = int(sMDS.val)    
    NOD = int(sNOD.val/16)*16
    UR = int(sUR.val)  
    SPWS= float(sSPWS.val)
    SR = float(sSR.val)
    P1 = 8*3*BS**2#int(sP1.val)
    P2 = 32*3*BS**2#int(sP2.val)
    Qscale = float(sQscale.val)

    
# Connect update actions to control elements
sBS.on_changed(update)
sMDS.on_changed(update)
sNOD.on_changed(update)
sUR.on_changed(update)
sSPWS.on_changed(update)
sSR.on_changed(update)
sP1.on_changed(update)
sP2.on_changed(update)
sSPWS.on_changed(update)
sQscale.on_changed(update)

print('Show interface to user')
plt.show()

# plt.imshow(tmdisp_visual)
# plt.show()
