import numpy as np
box_size = 5
interval = 5
j=5
height = 1200
width = 1600
i = 5
imgL=np.zeros((height,width))
template = imgL[(i-box_size) if ((i-box_size)>0) else (i-interval) :(i+box_size) if ((i+box_size)<height) else (i-interval) ,(j-box_size) if ((j-box_size)>0) else (j-interval) :(j+box_size) if ((j+box_size)<width) else (j-interval)]
template = imgL[ (i-box_size) if ((i-box_size)>0) else (i-interval) :(i+box_size) if ((i+box_size)<height) else (i-interval) ,(j-box_size) if ((j-box_size)>0) else (j-interval) :(j+box_size) if ((j+box_size)<width) else (j-interval)]
searchstrip = imgL[((i-box_size) if ((i-box_size)>0) else (i-interval) ):((i+box_size) if ((i+box_size)<height) else (i-interval) ),:] # (j-box_size):(j+box_size)+x]
                
print(template.shape)
print(searchstrip.shape)  