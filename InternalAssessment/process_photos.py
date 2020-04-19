"""
Go through a folder, picking up each headshot jpg
Recognize face, center the eyes, and make smaller file
Very useful if you have a bunch of pics and want to use them as avatar for websites

pip install opencv-python
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
"""

import cv2

import os
import glob
import re


FILE_EXT = '.jpg'


if __name__ == '__main__':

    # directory name
    source_dir = 'source'

    # Create the face detection object, which uses training models (already pre-made)
    # This cascade is the best for head shots
    # "cascade" is a term used to indicate the kind of algorithm it is
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # make array of all file names at source_dir
    # uses os.path.join and glob library to find all filesnames
    # that end with FILE_EXT
    photo_files = glob.glob(os.path.join(source_dir, '*' + FILE_EXT))

    # The dimensions of the thumbnail, recommended by ManageBac
    squared_size = 250


    # iterate over each file name
    for photo_file in photo_files:

        # get the name of the file, which we'll use later to save
        file_name = photo_file.split(FILE_EXT)[0].split(os.path.sep)[-1]

        # read in the contents of the image of the file
        # now img have shapes array for each shape detected
        img = cv2.imread(photo_file)

        # For math, see https://realpython.com/face-recognition-with-python/ for intro
        radius = 280.0 / img.shape[1]
        dim = (280, int(img.shape[0] * radius))

        # We want to make it smaller, so we use cv2.resize method
        # For info on what the heck his interpolation, see https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # Convert the resized image into a grayscale image, which makes
        # the below algorithm much faster
        grayscale_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Detect face(s)!
        faces = face_cascade.detectMultiScale(grayscale_img, 1.2, 5)
        if len(faces) == 0:
            print("Skipping, as no face recognized in {}".format(photo_file))
            continue

        # get the dimensions, width and height of the face detected
        x, y, w, h = faces[0]

        # now adjust our numbers so that nx and ny have the new values of the x, y coords
        t = (squared_size - w) // 2
        nx = x - t
        u = (squared_size - h) // 2
        ny = y - u

        # now we crop
        newimg = resized[ny:squared_size+ny, nx:squared_size+nx]

        # Save it by taking the newimg crop, and saving it onto disk
        # cv2.imwrite
        path_to_save = os.path.join('destination', file_name + '.png')
        if not cv2.imwrite(path_to_save, newimg):
            print("Did not save {}".format(path_to_save))
process_photos.py
Displaying process_photos.py.
