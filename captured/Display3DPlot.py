from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import argparse 
from plyfile import PlyData as ply
import os
parser = argparse.ArgumentParser(description='Input file path.')
parser.add_argument('FilePath', metavar='FilePath', type=str,
                    help='Input filepath here for the 3d pointcloud')
args = parser.parse_args()

def read_ply_xyzrgb(filename):
    """ read XYZRGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = ply.read(f)
        num_verts = plydata['vertex'].count
        rgb = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        x = plydata['vertex'].data['x']
        y = plydata['vertex'].data['y']
        z = plydata['vertex'].data['z']
        rgb[:,0] = plydata['vertex'].data['red']
        rgb[:,1] = plydata['vertex'].data['green']
        rgb[:,2] = plydata['vertex'].data['blue']
    return x,y,z,rgb
def plot_ply(infile):
    
    fig = plt.figure()
    
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    x,y,z,rgb = read_ply_xyzrgb(infile)
    # defining all 3 axes
    
    ax.scatter(x, y, z, c=rgb/255,s=0.005)
    ax.set_title(infile)
    plt.show()
	
if __name__ == '__main__':
	infile = args.FilePath
	plot_ply(infile)
