import matplotlib.pyplot as plt
import cv2
var = "8"
image_l = cv2.imread(f"E:/Stereo-3D-Reconstruction/captured/rectified/{var}_L_.png",cv2.IMREAD_GRAYSCALE)
image_r = cv2.imread(f"E:/Stereo-3D-Reconstruction/captured/rectified/{var}_R_.png",cv2.IMREAD_GRAYSCALE)
image_l = image_l[600:1100,400:1200]
image_r = image_r[600:1100,400:1200]
plt.subplot(1,2,1)
plt.imshow(image_l)
plt.subplot(1,2,2)
plt.imshow(image_r)
cv2.imwrite(f"{var}crop_L.png",image_l)
cv2.imwrite(f"{var}crop_R.png",image_r)

plt.savefig(f"{var}.png")
plt.show()