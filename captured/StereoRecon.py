import numpy as np
import cv2
import struct
import stset
from tempmatchdisp import keydisparity

class D3Recon:
    
    def __init__(self,calibfolderpath,K=0.6,block_size=3,tmwin_size=30,searchrange=0):
        self.left_stereo_map, self.right_stereo_map, self.Q ,self.lcam_mtx,self.rcam_mtx= stset.st_maps(calibfolderpath, (1600, 1200))
        self.K = K
        self.block_size = block_size
        self.tmwin_size = tmwin_size
        self.searchrange = searchrange
        print('MAPS COMPUTED')

    def reconstruct(self,imgL,imgR,plypath):
        
        rectL, rectR = stset.st_rectify(imgL, imgR, self.left_stereo_map, self.right_stereo_map)
        dispmap = self.stereo_depth_map(rectL,rectR)
        points3D = cv2.reprojectImageTo3D(dispmap, self.Q.astype(np.float32),handleMissingValues=False,ddepth=-1)
        
        points2D = points3D[dispmap != 0 ]
        colors = imgR
        colors2D = colors[dispmap!=0]
        self.write_pointcloud(points2D,colors2D,plypath)
         
    def siftkpts(self,rectL,rectR):
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
            if m.distance < self.K*n.distance:
                # Keep this keypoint pair
                matchesMask[i] = [1, 0]
                ptsR.append(kp2[m.trainIdx].pt)
                ptsL.append(kp1[m.queryIdx].pt)

        ptsL = np.int32(ptsL)
        ptsR = np.int32(ptsR)
        return ptsL,ptsR
    def filterbadkpoints(self,ptsL,ptsR,epsilon=5,rneg=False):
        
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
        comptsLR=np.hstack((ptsL,ptsR))
        _,indices = np.unique(comptsLR,axis=0,return_index=True)
    
        ptsL = ptsL[indices]
        
        ptsR = ptsR[indices]
        # print("After removing Duplicates",ptsL.shape,ptsL[0])
        
        return ptsL,ptsR

    def stereo_depth_map(self,rectL,rectR):
    ##!Note: Inverted Left and Right intentionally as setup gives inverted pictures change when setup changes    
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        rectified_pair = (grayR, grayL)
        
        dmLeft = rectified_pair[0]
        dmRight = rectified_pair[1]

        # print("Keypoint matching Disparity")
        ptsL,ptsR   = self.siftkpts(dmLeft,dmRight)
        ptsL,ptsR = self.filterbadkpoints(ptsL,ptsR)
        disp   = keydisparity(dmLeft,dmRight,self.block_size,ptsL,ptsR,self.tmwin_size,self.searchrange)
      
        return disp
    
    def write_pointcloud(self,points,colors,filename):

        """ creates a .pkl file of the point clouds generated
        """

        assert points.shape[1] == 3,'Input output_points_sgbm points should be Nx3 float array'
        if colors is None:
            colors = np.ones(points.shape).astype(np.uint8)*255
        assert points.shape == colors.shape,'Input RGB colors should be Nx3 float array and have same size as input output_points_sgbm points'

        # Write header of .ply file
        fid = open(filename,'wb')
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n'%points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(points.shape[0]):
            fid.write(bytearray(struct.pack("fffccc",points[i,0],points[i,1],points[i,2],
                                            colors[i,0].tobytes(),colors[i,1].tobytes(),
                                            colors[i,2].tobytes())))
        fid.close()



    