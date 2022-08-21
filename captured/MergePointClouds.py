import cv2
import numpy as np 
import os
from tqdm import tqdm
import random
import struct
import numpy as np

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
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

    t = -R @ centroid_A + centroid_B

    return R, t

def write_pointcloud(xyz_points,rgb_points,filename):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,2].tobytes())))
    fid.close()


def avg_rep_error(rep_error):
    """
    Calculate average reprojection error for all the 3d triangulated points
    Args:
        rep_error: (list of lists [[x,y]] where x,y represent the reprojection error for a single point)
    Returns:
        average reprojection error in pixels for all the triangulated 3d points
    """
    sum_x = 0
    sum_y = 0
    for x,y in rep_error:
        sum_x = sum_x + abs(x)
        sum_y = sum_y + abs(y)
    
    return [sum_x / len(rep_error), sum_y / len(rep_error)]



def rep_error_fn(opt_variables, points_2d, num_pts):
    """
        Calculate reprojection error from the given camera projection matrices, and 2d pixels
        for each triangulated 3d point
    Args:
        opt_variables: (stacked numpy array of projection matrix and 3d points)
        points_2d: (numpy array of 2d sift keypoints)
        num_points: (int representing total number of keypoints)
    Returns:
        Reprojection error of the 3d reconstruction
    """ 
    
    P = opt_variables[0:12].reshape(3,4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]

        #print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(list(pt_2d - reprojected_pt[0:2]))
        
    avg_error = avg_rep_error(rep_error)
        
    return avg_error


def filter_points(matched_points1, matched_points2, th=50):
    """
    Filters matched points that do not specify the horizontal line criteria. Since, the frame rate is high, 
    matched points in successive frames are expected to be horizontal. 
    Args:
        Two numpy arrays showing corresponding feature matches (x,y) from the two images
    Returns:
        Filtered matched points that satisfy the horizontal criteria
    """
    new_matched_points1 = []
    new_matched_points2 = []
    
    for i in range(len(matched_points1)):
        x1,y1 = (matched_points1[i][0], matched_points1[i][1])
        x2,y2 = (matched_points2[i][0], matched_points2[i][1])
        
        if abs(y2-y1) < th:
            new_matched_points1.append([x1,y1])
            new_matched_points2.append([x2,y2])
        else:
            pass
        
    new_matched_points1 = np.array(new_matched_points1)
    new_matched_points2 = np.array(new_matched_points2)
        
    return new_matched_points1, new_matched_points2
def applyhomogenuos(a,M):
    # print(a)
    # print(M)
    t=np.dot(M,a)
    # print(t)
    return t  
def removeoutlier(points_3d,return_ind = False):
    a = points_3d[0:3].T
    mean, stdev = np.mean(a, axis=0), np.std(a, axis=0)
    outliers = ((np.abs(a[:,0] - mean[0]) > stdev[0])
                    * (np.abs(a[:,1] - mean[1]) > stdev[1])
                    * (np.abs(a[:,2] - mean[2]) > stdev[2]))
    print("Outliers",outliers)
            # print("Color",output_color.shape)
            
    points_3d = np.delete(points_3d,outliers,1)

    print("After Points3d",points_3d.shape)

    if return_ind:
        return outliers
    else:
        return points_3d

def reconstruct(pts1,pts2,mtx):
            R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
            # initialize camera pose 1 
            R_t_1 = np.empty((3,4))
            # calculate projection matrix for camera pose 0
            P1 = np.matmul(mtx, R_t_0)
            # initialize projection matrix for camera pose 1
            P2 = np.empty((3,4))
    

            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
            print("\nThe fundamental matrix \n" + str(F))
            
            # extract essential matrix
            E = np.matmul(np.matmul(np.transpose(mtx), F), mtx)
            # recover new camera pose
            retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx)
            # get extrinsic camera matrix
            R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
            R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())
            # get projection matrix
            P2 = np.matmul(mtx, R_t_1)
            

            # triangulate for 3d world po.ints
            points_3d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
            
    
            points_3d /= points_3d[3]
            

            return points_3d[0:3].T



def find_common_pts_index(a,b):
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
    

def siftkpts(rectL,rectR,K,mtx):
        
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
            if m.distance < K*n.distance:
                # Keep this keypoint pair
                matchesMask[i] = [1, 0]
                ptsR.append(kp2[m.trainIdx].pt)
                ptsL.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(ptsL)
        pts2 = np.int32(ptsR)
        R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
            # initialize camera pose 1 
        R_t_1 = np.empty((3,4))
            # calculate projection matrix for camera pose 0
        P1 = np.matmul(mtx, R_t_0)
            # initialize projection matrix for camera pose 1
        P2 = np.empty((3,4))
        #fliter based on horizontal threshold
        pts1,pts2 = filter_points(pts1,pts2,th=500)
            # find the fundamental matrix

        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
        print("\nThe fundamental matrix \n" + str(F))
            # extract essential matrix
        E = np.matmul(np.matmul(np.transpose(mtx), F), mtx)
            # recover new camera pose
        retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx)
            # get extrinsic camera matrix
        R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
        R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())
            # get projection matrix
        P2 = np.matmul(mtx, R_t_1)
            

        # triangulate for 3d world po.ints
        points_3d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
            
    
        points_3d /= points_3d[3]
        #Return index of outlier in 3d to remove them from the sift keypoits
        outliers = removeoutlier(points_3d,True)
        print("Shape before Delete: ",pts1.shape)
        pts1 = np.delete(pts1.T,outliers,1)
        pts2 = np.delete(pts2.T,outliers,1)
        
        return pts1.T,pts2.T
path = "D:/Downloads/Random Downloads/OpenSFM Depth Maps/OpenSFM/images/"
sequence = os.listdir(path)
# sequence.sort()

print(sequence)
imgvid = ['frame34.jpg','frame40.jpg','frame46.jpg']
h,w = 768,1366  
img2 =None
out = cv2.VideoWriter("trackedPoints.mp4",cv2.VideoWriter_fourcc(*"mp4v"),1,(w,h))
for i,imgpath in enumerate(imgvid):
    print("Image No: ",i,"Path",imgpath)
    if i == 1:
        img2 = img1
        img1 = cv2.imread(path + imgpath)
        continue
    
    if i ==0:  
        img1 = cv2.imread(path + imgpath)
        continue
    
    # out.write(combined)
    
    
    if i >=2:   
        img3 = img2
        img2 = img1
        img1 = cv2.imread(path + imgpath)
    
    calibParams = "D:/Downloads/GL010036/Calib_Params/"

    mtx   = np.load(calibParams+'mtx.npy')
    dist  = np.load(calibParams+'dist.npy')   
    pts1 , pts2  = siftkpts(img1,img2,0.6,mtx)
    pts3 , pts4  = siftkpts(img2,img3,0.6,mtx)
    pts2loc, pts3loc = find_common_pts_index(pts2,pts3)
    print("pts3losc = ",pts3loc.shape,"pts2losc = ",pts2loc.shape)

    image12pts = reconstruct(pts1,pts2,mtx)
    image23pts = reconstruct(pts3,pts4,mtx)
    print("Before Removal ",image23pts.shape)
    image12pts = removeoutlier(image12pts.T,False).T
    image23pts = removeoutlier(image23pts.T,False).T
    print("After Removal ",image23pts.shape)
    img12samepts = reconstruct(pts1[pts2loc],pts2[pts2loc],mtx).T
    img23samepts = reconstruct(pts3[pts3loc],pts4[pts3loc],mtx).T
    
    print("Image12 : ",img12samepts.shape,"Image23 : ",img23samepts.shape)
    R,T = rigid_transform_3D(img12samepts,img23samepts)
    
    

    
    I = np.array([0,0,0,1]).reshape(1,4)
    #Invert R and T in the homogenuos matrix to invert the transformation between the two points
    # R= R.T
    # T = np.dot(R.T,-1*T)
    print("R : ",R,R.shape)
    print("T : ",T,T.shape)

    Rt = np.concatenate([R,T],axis=-1)
    M  = np.concatenate([Rt,I],axis=0)

    print("Rt : ",Rt,"\nShape : ",Rt.shape)
    print("M : ",M,"\nShape : ",M.shape)

    add12 = np.array([[1]]*image12pts.shape[0])
    add23 = np.array([[1]]*image23pts.shape[0])
    
    image12ptsadd = np.concatenate([image12pts,add12],axis=1)
    image23ptsadd = np.concatenate([image23pts,add23],axis=1)

    print("Shape : ",image23ptsadd.shape)
    
                    
    transformed = np.apply_along_axis(applyhomogenuos,1,image12ptsadd,M)[:,0:3]
    
    print(transformed[0])
    transformedcolor = np.array([[255,0,0]]*transformed.shape[0])
    img23sameptscolor = np.array([[0,255,255]]*img23samepts.T.shape[0])
    image12ptscolor = np.array([[255,255,255]]*image12pts.shape[0])
    image23ptscolor = np.array([[0,255,0]]*image23pts.shape[0])
    allpts = np.concatenate([image12pts,image23pts,img23samepts.T],axis=0).astype(np.float32)
    allptscolor = np.concatenate([image12ptscolor,image23ptscolor,img23sameptscolor],axis=0).astype(np.uint8)

    print("All pts :",allpts.shape,allpts)
    print("All pts Color:",allptscolor.shape)
    write_pointcloud(allpts,allptscolor,"allpts.ply")
    write_pointcloud(image12pts,image12ptscolor.astype(np.uint8),"12pts.ply")
    write_pointcloud(image23pts,image23ptscolor.astype(np.uint8),"23pts.ply")
    write_pointcloud(transformed,transformedcolor.astype(np.uint8),"transpts.ply")
    
    # print(img12ptscolor)    
    
    print(img23samepts.shape)
    
    print(img23samepts.shape)
    
    # img2pts = pts2[pts2loc]
    # img3pts = pts4[pts3loc]

    # image1 = img1
    # image2 = img2
    # image3 = img3

    # for points1,points2,points3 in zip(img1pts,img2pts,img3pts):
    #     color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    #     image1 = cv2.circle(img1,points1,10,color,-1)
    #     image2 = cv2.circle(img2,points2,10,color,-1)
    #     image3 = cv2.circle(img3,points3,10,color,-1)
     
    # combined = cv2.hconcat([image1,image2,image3])
    
    # combined = cv2.resize(combined,(w,h))
    # cv2.imshow("s", combined)
    # cv2.waitKey(0)
    


    
    

    