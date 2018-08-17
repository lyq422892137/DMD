import glob
import os
from PIL import Image

for files in glob.glob('D:/input_hw/*.jpg'):
    filepath, filename = os.path.split(files)
    filterame, exts = os.path.splitext(filename)
    opfile = r'D:/input_hw/'
    if (os.path.isdir(opfile) == False):
        os.mkdir(opfile)
    im = Image.open(files)
    w, h = im.size
    im_ss = im.resize((int(w * 1.0), int(h * 1.0)))
    im_ss.save(opfile + filterame + '.jpg')