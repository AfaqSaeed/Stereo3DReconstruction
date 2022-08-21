from scipy.sparse import csr_matrix
import numpy as np
import cv2
import struct
import stset
from tempmatchdisp import keydisparity
from tqdm import tqdm
import matplotlib.pyplot as plt
   
class D3Recon:
    

    def __init__(self,calibfolderpath,K=0.6,block_size=3,I=0.0,searchrange=0):
        self.left_stereo_map, self.right_stereo_map, self.Q ,self.lcam_mtx,self.rcam_mtx,self.RL,self.RR= stset.st_maps(calibfolderpath, (1600, 1200))
        self.K = K
        self.block_size = block_size
        self.I = I
        self.searchrange = searchrange
        print('MAPS COMPUTED')
    def rectify(self,imgL,imgR):
        rectL,rectR = stset.st_rectify(imgL, imgR, self.left_stereo_map, self.right_stereo_map)
        return rectL,rectR
    def vizdispmap(self,dispmap):
        local_max = dispmap.max()
        local_min = dispmap.min()
        disparity_grayscale = (dispmap-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
        colordispmap = cv2.applyColorMap(np.uint8(disparity_fixtype),cv2.COLORMAP_JET)
        return colordispmap
    def reconstruct(self,imgL,imgR,plypath,dispmappath,returncoords=False):
        
        rectL, rectR = self.rectify(imgL,imgR)
        dispmap = self.stereo_depth_map(rectL,rectR)
        # plt.imshow(dispmap,"jet")
        # plt.savefig(dispmappath)
        # plt.show()
        colordispmap = self.vizdispmap(dispmap)
        if cv2.imwrite(dispmappath,colordispmap):
            print("Map written succcessfully to : ",dispmappath)    
        points3D = cv2.reprojectImageTo3D(np.float32(dispmap), self.Q.astype(np.float32),handleMissingValues=False,ddepth=-1)
        
        points2D = points3D[np.bitwise_and(dispmap != 0,dispmap>190) ]
        colors = rectR
        colors2D = colors[np.bitwise_and(dispmap != 0,dispmap>190)]
        self.write_pointcloud(points2D,colors2D,plypath)
        print("Ply created succcessfully at : ",plypath)
        if returncoords:
            return points3D,dispmap
        

    def bound(self,var,limit,max):
        if max:
            if var>limit:
                var=limit
        else:
            if var<limit:
                var = limit
        return var
    def siftkpts(self,rectL,rectR,K,returndesc=False):
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
        ptsL = []
        ptsR = []
        descL = []
        descR = []

        for i, (m, n) in enumerate(matches):
            if m.distance < K*n.distance:
                ptsR.append(kp2[m.trainIdx].pt)
                ptsL.append(kp1[m.queryIdx].pt)
                if returndesc==True:
                    descL.append(des1[m.queryIdx])
                    descR.append(des2[m.trainIdx])

        ptsL = np.int32(ptsL)
        ptsR = np.int32(ptsR)
        if returndesc:
            return ptsL,ptsR,descL,descR
        else:
            return ptsL,ptsR
    
    def getmorekeypoints(self,rectL,rectR,ptsL,ptsR,win_size,K,I):
        nptsL = []
        nptsR = []
        for ptsl,ptsr in tqdm(zip(ptsL,ptsR),total =ptsL.shape[0]):
                a,b = ptsl[1] - win_size,ptsl[1] + win_size
                c,d = ptsl[0] - win_size,ptsl[0] + win_size
                e,f = ptsr[1] - win_size,ptsr[1] + win_size
                g,h = ptsr[0] - win_size,ptsr[0] + win_size
                a,b = self.bound(a,0,False),self.bound(b,1200,True)
                c,d = self.bound(c,0,False),self.bound(d,1600,True)
                e,f = self.bound(e,0,False),self.bound(f,1200,True)
                g,h = self.bound(g,0,False),self.bound(h,1600,True)
                left  = rectL[a:b,c:d]
                
                right = rectR[e:f,g:h]
                #  print(left.shape)
                #  print(right.shape)
                if left.shape != right.shape:
                    continue
                try:
                    zptsl,zptsr = self.siftkpts(left,right,K+I)
                    self.filterbadkpoints(zptsl,zptsr,epsilon=5)
                except:
                    print("No Kpts found",zptsl.shape)
                    continue
                zptsl,zptsr = zptsl-win_size,zptsr-win_size
                
                nptsL.extend(ptsl+zptsl)
                nptsR.extend(ptsr+zptsr)
        
        nptsL = np.array(nptsL) 
        nptsR = np.array(nptsR)
        
        
        return nptsL,nptsR    
    def filterbadkpoints(self,ptsL,ptsR,epsilon=10,xrangel=None,xrangeh=None):
    
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

    def stereo_depth_map(self,rectL,rectR):
    ##!Note: Inverted Left and Right intentionally as setup gives inverted pictures change when setup changes    
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        rectified_pair = (grayR, grayL)
        
        dmLeft = rectified_pair[0]
        dmRight = rectified_pair[1]

        # print("Keypoint matching Disparity")
        ptsL,ptsR   = self.siftkpts(dmLeft,dmRight,self.K)
        ptsL,ptsR = self.filterbadkpoints(ptsL,ptsR)
        
        nptsL,nptsR = self.getmorekeypoints(dmLeft,dmRight,ptsL,ptsR,win_size=50,K=self.K,I=self.I)
        print("Additional Keypoints added",nptsR.shape) 
        nptsL,nptsR = self.filterbadkpoints(nptsL,nptsR,xrangel=0,xrangeh=270)
        print("Points After Filtering",nptsR.shape ) 
        
        nptsL = np.append(ptsL,nptsL,axis = 0)
        nptsR = np.append(ptsR,nptsR,axis = 0)
        disp   = keydisparity(dmLeft,dmRight,self.block_size,nptsL,nptsR)
        disp   = cv2.blur(disp,(5,5))
    
        return disp
    def reverse_xy(self,kpts):
        x = kpts[:,0].copy()
        y = kpts[:,1].copy()
        
        kpts[:,0] = y
        kpts[:,1] = x
        # print(kpts)
        return kpts

    def rigid_transform_3D(self,A, B):
        assert A.shape == B.shape

        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

        num_rows, num_cols = B.shape
        if num_rows != 3:
            raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(-1, 1)
        centroid_B = centroid_B.reshape(-1, 1)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = Am @ np.transpose(Bm)

        # sanity check
        #if linalg.matrix_rank(H) < 3:
        #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            print("det(R) < R, reflection detected!, correcting for it ...")
            Vt[2,:] *= -1
            R = Vt.T @ U.T

        T = -R @ centroid_A + centroid_B
        I = np.array([0,0,0,1]).reshape(1,4)
        Rt = np.concatenate([R.T,np.dot(-1*R.T,T)],axis=-1)
        M  = np.concatenate([Rt,I],axis=0)
        
        addA = np.array([[1]]*A.shape[1])
        addB = np.array([[1]]*B.shape[1])
        
        Aadd = np.concatenate([A.T,addA],axis=1)
        Badd = np.concatenate([B.T,addB],axis=1)

        BtoA = np.apply_along_axis(self.applyhomogenuos,1,Badd,M)[:,0:3]
        error = A.T-BtoA
        print("Error : ",np.average(np.linalg.norm(error,axis=1)))

        return R, T


    def merge(self,b):
        b1d = b[:,0]*10000 + b[:,1]
        return b1d


    def deadreckoning(self,beforeL,beforeR,afterL,afterR,H,xyz=None,rgb=None,viz =False):
        # Note: L and R inverted because of the camera incorrectly mounted
        #rectify images 
        beforeL, beforeR = self.rectify(beforeL,beforeR)
        afterL, afterR = self.rectify(afterL,afterR)
        beforekpts,afterkpts = self.siftkpts(beforeR,afterR,0.6)
        #get depth for both with R as the reference because of inversion
        beforedepthR = self.stereo_depth_map(beforeL,beforeR)
        afterdepthR = self.stereo_depth_map(afterL,afterR)
        # Testing code for visulaizing the correct kpts
        # beforecolor = self.vizdispmap(beforedepthR)
        # aftercolor = self.vizdispmap(afterdepthR)
        if viz:
            plt.subplot(2,2,1)
            plt.imshow(beforedepthR,cmap='jet')
            plt.subplot(2,2,2)
            plt.imshow(beforeR,cmap='jet')
            plt.subplot(2,2,3)
            plt.imshow(afterdepthR,cmap='jet')
            plt.subplot(2,2,4)
            plt.imshow(afterR,cmap='jet')  
            plt.show()
        #find keypoints b/w before and after Rs
        # print(beforekpts)
        print('beforedepthR shape:',beforedepthR.shape)
        # create epty masks and draw kpts locations as true values 
        beforemask = np.zeros(beforedepthR.shape,dtype= bool)
        aftermask = np.zeros(afterdepthR.shape,dtype=bool)
    
        # reverse the order of x y for indexing
        
        # print("Before:",beforekpts)
        beforekpts = self.reverse_xy(beforekpts)
        afterkpts = self.reverse_xy(afterkpts)
       
        data = np.array([True]*beforekpts.shape[0])
        
        # print("Unique before",np.unique(beforekpts,axis=0).shape)
        u,sameindex = np.unique(afterkpts,axis=0,return_index=True)
        
        # beforemask = csr_matrix((data[sameindex], (beforekpts[sameindex,0], beforekpts[sameindex,1])), shape=(beforemask.shape[0],beforemask.shape[1] )).toarray()
        # aftermask = csr_matrix((data[sameindex], (afterkpts[sameindex,0], afterkpts[sameindex,1])), shape=(beforemask.shape[0],beforemask.shape[1] )).toarray()

        
        print('kpts shape ',beforekpts.shape,afterkpts.shape)
        # print("AFter:",aftermask.shape)
        
        # beforedepthR = np.where(beforemask,beforedepthR,0) 
        # afterdepthR = np.where(aftermask,afterdepthR,0) 
        # print('After count-nonzero ',np.count_nonzero(aftermask)) 
        before3D = cv2.reprojectImageTo3D(np.float32(beforedepthR), self.Q.astype(np.float32),handleMissingValues=False,ddepth=-1)
        after3D = cv2.reprojectImageTo3D(np.float32(afterdepthR), self.Q.astype(np.float32),handleMissingValues=False,ddepth=-1)
        before3Dall= before3D[np.bitwise_and(beforedepthR !=0,beforedepthR>190) ]
        after3Dall= after3D[np.bitwise_and(afterdepthR !=0,afterdepthR>190)]

        before3Dsame= before3D[tuple(beforekpts[sameindex].T)]
        beforeindex = np.bitwise_not(np.apply_along_axis(lambda x:any(np.isinf(x)),1,before3Dsame))
        after3Dsame = after3D[tuple(afterkpts[sameindex].T)]
        afterindex = np.bitwise_not(np.apply_along_axis(lambda x:any(np.isinf(x)),1,after3Dsame))
        removeindex = np.bitwise_and(afterindex,beforeindex)
        print("removeindex",removeindex.shape)
        before3Dsame = before3Dsame[removeindex]
        after3Dsame = after3Dsame[removeindex]

        print("Beforesame",before3Dsame.shape)
        print("Aftersame",after3Dsame.shape)
        before3Dcolor=np.zeros(before3Dall.shape,dtype=np.uint8)
        after3Dcolor=np.zeros(after3Dall.shape,dtype=np.uint8)
        before3Dcolor[:,0]= beforeR[np.bitwise_and(beforedepthR !=0,beforedepthR>190) ][:,0]
        after3Dcolor[:,2]= afterR[np.bitwise_and(afterdepthR !=0,afterdepthR>190)][:,0]
    
        transformedafter2before,H = self.allign3D(before3Dall,after3Dall,before3Dsame,after3Dsame,H)
        transformedcolor = np.zeros(transformedafter2before.shape,dtype=np.uint8)
        transformedcolor[:,1] = afterR[np.bitwise_and(afterdepthR !=0,afterdepthR>190)][:,0]


        if rgb is None:
            allpts = np.concatenate([before3Dall,transformedafter2before],axis=0).astype(np.float32)
            allptscolor = np.concatenate([before3Dcolor,transformedcolor],axis=0).astype(np.uint8)
        else :
            allpts = np.concatenate([xyz,transformedafter2before],axis=0).astype(np.float32)
            allptscolor = np.concatenate([rgb,transformedcolor],axis=0).astype(np.uint8)
        
        # print("All pts :",allpts.shape,allpts)
        print("All pts Color:",allptscolor.shape)
        self.write_pointcloud(allpts,allptscolor,"allpts.ply")

        # self.write_pointcloud(after3Dall,after3Dcolor.astype(np.uint8),"before.ply")
        # self.write_pointcloud(before3Dall,before3Dcolor.astype(np.uint8),"after.ply")
        self.write_pointcloud(transformedafter2before,transformedcolor.astype(np.uint8),"transformed.ply")
       
        if viz :
            fig2 = plt.figure(2)
            plt.subplot(2,2,1)
            plt.imshow(beforemask)
            plt.subplot(2,2,2)
            plt.imshow(aftermask)
            plt.subplot(2,2,3)
            plt.imshow(beforedepthR)
            plt.subplot(2,2,4)
            plt.imshow(afterdepthR)
            plt.show()
        return H

    def allign3D(self,image12pts,image23pts,img12samepts,img23samepts,H):
        print(img12samepts.shape)
        R,T = self.rigid_transform_3D(img12samepts.copy().T,img23samepts.copy().T)

        I = np.array([0,0,0,1]).reshape(1,4)
        #Invert R and T in the homogenuos matrix to invert the transformation between the two points
        # R= R.T
        # T = np.dot(R.T,-1*T)
        print("R : ",R.copy().T,R.shape)
        print("T : ",np.dot(-1*R.copy().T,T),T.shape)

        Rt = np.concatenate([R,T],axis=-1)
        M  = np.concatenate([Rt,I],axis=0)
        H = np.dot(H,M)
        print("Rt : ",Rt,"\nShape : ",Rt.shape)
        print("M : ",M,"\nShape : ",M.shape)

        add12 = np.array([[1]]*image12pts.shape[0])
        add23 = np.array([[1]]*image23pts.shape[0])
        
        image12ptsadd = np.concatenate([image12pts,add12],axis=1)
        image23ptsadd = np.concatenate([image23pts,add23],axis=1)

        print("Shape : ",image23ptsadd.shape)
        
                        
        transformed23to12 = np.apply_along_axis(self.applyhomogenuos,1,image23ptsadd,H)[:,0:3]
        return transformed23to12,H
    def applyhomogenuos(self,a,H):
    # print(a)
    # print(M)
        t=np.dot(H,a)
    # print(t)
        return t  

    def find_common_pts_index(self,a,b):
        print('B: ',b.shape,"A :",a.shape)
        b1d = b[:,0]*10000 + b[:,1]
        
        # print(pts31d)
        print(b1d.shape)
        a1d = a[:,0]*10000 + a[:,1]
        print(a1d.shape)
        # print(pts21d)
        matches,a_loc,b_loc = np.intersect1d(a1d,b1d,return_indices=True)
        print(all(a1d [a_loc]==b1d[b_loc]))
        print(a_loc.shape)
        return a_loc,b_loc
    

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



    