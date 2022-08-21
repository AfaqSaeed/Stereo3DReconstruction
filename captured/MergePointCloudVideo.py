import cv2
import numpy as np 
import os
from tqdm import tqdm
import random
import struct
import matplotlib.pyplot as plt
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



def plot_3d_matplotlib(pts_3d,color):
    X, Y, Z = pts_3d
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for m, zlow, zhigh in [('o', -50, 50), ('o', -50, 50)]:
        xs = X
        ys = Y
        zs = Z
        ax.scatter(xs, ys, zs,c=color/255, marker=m, s=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
def find_common_pts_index(a,b):
    b1d = b[:,0]*10000 + b[:,1]
    
    # print(pts31d)
    print(b1d.shape)
    a1d = a[:,0]*10000 + a[:,1]
    print(a1d.shape)
    # print(pts21d)
    matches,a_loc,b_loc = np.intersect1d(a1d,b1d,return_indices=True)
    print(all(a1d [a_loc]==b1d[b_loc]))
    
    return a_loc,b_loc
    
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

def siftkpts(rectL,rectR,K):
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

        ptsL = np.int32(ptsL)
        ptsR = np.int32(ptsR)
        return ptsL,ptsR
# path = "E:/Labelling Policy/"
# sequence = os.listdir(path)
# videos = [file   for file in sequence if file[-4:]==".MP4" ] 
# print(videos)
# videos.sort()
# print(videos)
# h,w = 768,1366  
# cap = cv2.VideoCapture(path + videos[0])
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# img2 =None
# out = cv2.VideoWriter("trackedPoints.mp4",cv2.VideoWriter_fourcc(*"mp4v"),1,(w,h))
def reconstruct(pts1,pts2,mtx):
            R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
            # initialize camera pose 1 
            R_t_1 = np.empty((3,4))
            # calculate projection matrix for camera pose 0
            P1 = np.matmul(mtx, R_t_0)
            # initialize projection matrix for camera pose 1
            P2 = np.empty((3,4))
    
            pts1,pts2 = filter_points(pts1,pts2,th=500)
            # find the fundamental matrix

            F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
            print("\nThe fundamental matrix \n" + str(F))
            # select only inlier points
            pts1 = pts1[mask.ravel()==1]
            pts2 = pts2[mask.ravel()==1]
            # extract essential matrix
            E = np.matmul(np.matmul(np.transpose(mtx), F), mtx)
            # print("\nThe essential matrix is \n" + str(E))
            # recover new camera pose
            retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx)
            # get extrinsic camera matrix
            R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
            R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())
            # get projection matrix
            P2 = np.matmul(mtx, R_t_1)
            # output_color = resized_img[pts1.T.astype(np.int64)[1,:],pts1.T.astype(np.int64)[0,:],:]

            # transpose
            pts1 = pts1.T
            pts2 = pts2.T


            # triangulate for 3d world points
            points_3d = cv2.triangulatePoints(P1, P2, pts1, pts2)
            
            
            print(points_3d.shape,pts1.shape)
            print(pts1.T.ravel().astype(np.int64))
    
            points_3d /= points_3d[3]
            # calculate reprojection error
            opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
            num_points = len(pts2[0])
            rep_error = rep_error_fn(opt_variables, pts2, num_points)
            print("\nAverage Reprojection Error \n" + str(rep_error))
            a = points_3d[0:3].T
            mean, stdev = np.mean(a, axis=0), np.std(a, axis=0)
            outliers = ((np.abs(a[:,0] - mean[0]) > stdev[0])
                    * (np.abs(a[:,1] - mean[1]) > stdev[1])
                    * (np.abs(a[:,2] - mean[2]) > stdev[2]))
            print("Points3d",points_3d.shape)
            # print("Color",output_color.shape)
            
            points_3d = np.delete(points_3d,outliers,1)
            # output_color = np.delete(output_color,outliers,0)
            
            return points_3d[0:3]

calibParams = "D:/Downloads/GL010036/Calib_Params/"
mtx   = np.load(calibParams+'mtx.npy')
dist  = np.load(calibParams+'dist.npy')
rvecs = np.load(calibParams+'rvecs.npy'),
tvecs = np.load(calibParams+'tvecs.npy',)
newcameramtx = np.load(calibParams+'Optimalmtx.npy')
path = "D:/Downloads/Random Downloads/OpenSFM Depth Maps/three/"
images = os.listdir(path)
for i,img_path in enumerate(tqdm(images)):

    
    if i == 1:
        img2 = img1
        img1 = cv2.imread(path+img_path)
        continue
    
    if i == 0:  
        img1 = cv2.imread(path+img_path)
        continue
    
    # out.write(combined)
    
    
    if i ==2:   
        img3 = img2
        img2 = img1
        img1 = cv2.imread(path+img_path)
        rightMap = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (img1.shape[1],img1.shape[0]), cv2.CV_16SC2)
            
        img1 =  cv2.remap(img1,rightMap[0],rightMap[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        img2 =  cv2.remap(img2,rightMap[0],rightMap[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        img3 =  cv2.remap(img3,rightMap[0],rightMap[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        
        pts1 , pts2  = siftkpts(img1,img2,0.6)
        pts3 , pts4  = siftkpts(img2,img3,0.6)
        
        pts2loc, pts3loc = find_common_pts_index(pts2,pts3)
        pts1sel = pts1[pts2loc]
        pts2sel = pts2[pts2loc]
        pts3sel = pts3[pts3loc]
        pts4sel = pts4[pts3loc]

        points_3d12 = reconstruct(pts1,pts2,mtx)
        points_3d23 = reconstruct(pts3,pts4,mtx)
        points3dcommon = reconstruct(pts3sel,pts4sel,mtx)
        color_12 = np.ones((max(points_3d12.shape),3),dtype=np.uint8)*[255,0,0]
        color_23 = np.ones((max(points_3d12.shape),3),dtype=np.uint8)*[0,255,0]
        color_common = np.ones((max(points_3d12.shape),3),dtype=np.uint8)*[0,0,255]
        stack2 = np.vstack(points_3d12,points_3d23)
        stack3 = np.vstack(stack2,points3dcommon)
        cstack2 = np.vstack(color_12,color_23)
        cstack3 = np.vstack(cstack2,color_common)
        plot_3d_matplotlib(stack3,cstack3)

    
    
    # image1 = img1.copy()
    # image2 = img2.copy()
    # image3 = img3.copy()

    # for points1,points2,points3 in zip(img1pts,img2pts,img3pts):
    #     color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    #     image1 = cv2.circle(img1,points1,10,color,-1)
    #     image2 = cv2.circle(img2,points2,10,color,-1)
    #     image3 = cv2.circle(img3,points3,10,color,-1)
     
    # combined = cv2.hconcat([image1,image2,image3])
    
    # combined = cv2.resize(combined,(w,h))
    # cv2.imshow("s", combined)
    # cv2.waitKey(0)
    


    
    

    