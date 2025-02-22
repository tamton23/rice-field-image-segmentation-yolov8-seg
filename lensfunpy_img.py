import cv2 # OpenCV library
import lensfunpy
import os
import glob
import numpy as np

cam_maker = 'NIKON CORPORATION'
cam_model = 'NIKON D3S'
lens_maker = 'Nikon'
lens_model = 'Nikkor 28mm f/2.8D AF'

db = lensfunpy.Database()
cam = db.find_cameras(cam_maker, cam_model)[0]
lens = db.find_lenses(cam, lens_maker, lens_model)[0]

print(cam)
# Camera(Maker: NIKON CORPORATION; Model: NIKON D3S; Variant: ;
#        Mount: Nikon F AF; Crop Factor: 1.0; Score: 0)

print(lens)
# Lens(Maker: Nikon; Model: Nikkor 28mm f/2.8D AF; Type: RECTILINEAR;
#      Focal: 28.0-28.0; Aperture: 2.79999995232-2.79999995232;
#      Crop factor: 1.0; Score: 110)
focal_length = 39.0
aperture = 4
distance = 10
file_path = 'Data_set/'
for filename in glob.glob(os.path.join(file_path, '*.JPG')):
    image_path = filename
    undistorted_image_path = filename[9:]

    img = cv2.imread(image_path).astype(np.float64)
    height, width = img.shape[0], img.shape[1]

    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, distance)

    undist_coords = mod.apply_geometry_distortion()
    img_undistorted = cv2.remap(img, undist_coords, None, cv2.INTER_LANCZOS4)
    cv2.imwrite("DataSet/" + filename[9:], img_undistorted)
