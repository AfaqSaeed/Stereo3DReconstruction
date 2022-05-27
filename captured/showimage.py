import matplotlib.pyplot as plt
import cv2
var = "Test_disparity (8)"
image = cv2.imread(f"D:/Downloads/Random Downloads/{var}.png",cv2.IMREAD_GRAYSCALE)
plt.imshow(image,cmap="rainbow")
plt.savefig(f"{var}.png")
plt.show()