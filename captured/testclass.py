import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from StereoRecon import D3Recon 

calibfolderpath = "./calibrators/calibParams/"
recon = D3Recon(calibfolderpath)

for img_no in tqdm(range(8,32)):
    
    imgL, imgR = cv2.imread(f"./test/left/{str(img_no)}_L_.png"), cv2.imread(f"./test/right/{str(img_no)}_R_.png")
    plypath= f"./3D Reconstructions/{str(img_no)}.ply"
    recon.reconstruct(imgL,imgR,plypath)

