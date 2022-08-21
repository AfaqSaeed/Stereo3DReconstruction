import cv2
from plyfile import PlyData as ply
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from StereoRecon import D3Recon 

calibfolderpath = "./calibrators/calibParams/"
root = "D:/Downloads/Random Downloads/sequence_images_stereo"
recon = D3Recon(calibfolderpath,block_size=5)



def read_ply_xyzrgb(filename):
    """ read XYZRGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = ply.read(f)
        num_verts = plydata['vertex'].count
        rgb = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        xyz = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        
        xyz[:,0] = plydata['vertex'].data['x']
        xyz[:,1] = plydata['vertex'].data['y']
        xyz[:,2] = plydata['vertex'].data['z']
        rgb[:,0] = plydata['vertex'].data['red']
        rgb[:,1] = plydata['vertex'].data['green']
        rgb[:,2] = plydata['vertex'].data['blue']
    return xyz,rgb
i=0
H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

for img_no in tqdm(range(12,21,3)):
    
    if i==0:
        beforeL, beforeR = cv2.imread(f"{root}/left/{str(img_no)}_L_.png",cv2.IMREAD_COLOR), cv2.imread(f"{root}/right/{str(img_no)}_R_.png",cv2.IMREAD_COLOR)
        print(beforeL.shape)
        i=i+1
        continue
    if i > 0:
        
        afterL,afterR = cv2.imread(f"{root}/left/{str(img_no)}_L_.png",cv2.IMREAD_COLOR), cv2.imread(f"{root}/right/{str(img_no)}_R_.png",cv2.IMREAD_COLOR)
    
    # rectL,rectR = recon.rectify(imgL,imgR)
    # rectimagepath = f"./3D Reconstructions/{str(img_no)}_R_.png"  
    # cv2.imwrite(rectimagepath,rectR)
    # plypath = f"./3D Reconstructions/{str(img_no)}.ply"
    # dispmappath = f"./3D Reconstructions/{str(img_no)}.png"
    if i>1:
        xyz,rgb = read_ply_xyzrgb('allpts.ply')
        print("XYZ Shape:",xyz.shape)
        H = recon.deadreckoning(beforeL,beforeR,afterL,afterR,H,xyz,rgb,viz=True)
    else:
        H = recon.deadreckoning(beforeL,beforeR,afterL,afterR,H,viz=True)
    i=i+1    
    # recon.reconstruct(imgL,imgR,plypath,dispmappath)


    beforeL,beforeR = afterL,afterR
