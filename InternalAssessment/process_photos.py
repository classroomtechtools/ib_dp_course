IB DP Computer Science
Stream
Classwork
People
Grades
IB DP Computer Science
Class code
gpipto2
Meet link
https://meet.google.com/lookup/b2eu3cyikm
Upcoming
Due today
Attendance: Extensibility
View all

Share something with your class…
Question: "Attendance: Extensibility"
Adam Morris posted a new question: Attendance: Extensibility
Created Apr 19Apr 19

Announcement: "This is the actual code I use to turn…"
Adam Morris
Created Apr 17Apr 17
This is the actual code I use to turn the large student photos we take every year into thumbnails. Please see this example in order to have a better understanding of what the IA is expecting you to do regarding "extensibility." You get two marks in Criterion E for being able to write code with good variable names and decent comments.
Hopefully this example gives you a clear picture of what is expected with that.

process_photos.py
Text

Question: "Potentially, what project are you thinking of doing for your IA?"
Adam Morris posted a new question: Potentially, what project are you thinking of doing for your IA?
Created Apr 11Apr 11 (Edited Apr 16)
1 class comment
Assignment: "Internal Assessment Report: Comprehension Check"
Adam Morris posted a new assignment: Internal Assessment Report: Comprehension Check
Created Apr 11Apr 11
Question: "What are specifications and why are they important in the development cycle?"
Adam Morris posted a new question: What are specifications and why are they important in the development cycle?
Created Apr 7Apr 7 (Edited Apr 8)
Assignment: "Mad Libs Game"
Adam Morris posted a new assignment: Mad Libs Game
Created Mar 29Mar 29 (Edited Apr 2)
1 class comment
Question: "What is a boot loader? When does it execute, and what is its function?"
Adam Morris posted a new question: What is a boot loader? When does it execute, and what is its function?
Created Mar 17Mar 17
Assignment: "Epidemic Simulation on Scratch"
Adam Morris posted a new assignment: Epidemic Simulation on Scratch
Created Mar 17Mar 17 (Edited Mar 17)
Material: "Operating Systems: Their role and function"
Adam Morris posted a new material: Operating Systems: Their role and function
Created Mar 16Mar 16
IB DP Computer Science
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
