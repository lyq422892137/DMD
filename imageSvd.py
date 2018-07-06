###################################################
#################
# try jpg
import numpy as np
from loadfile import loaddata
img = loaddata(1000)

###################################################

print(type(img))
print(img.shape)


calculate_svd(img, 1)
