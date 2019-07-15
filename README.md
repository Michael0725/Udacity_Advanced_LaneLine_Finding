## Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

The goal / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration 

Here, I use the the method offered by open cv to calculate the camera calibration matrix and the distortion coefficients.At first, I need to go through 20 chessboard pics to find the coordinate of each corner and add it to the "img points".Then, I need also add the object points cordinate to "objpoints".Secondly, I just need to put the "imgpoints" and the "objpoints" to the cv.calibrateCamera()，then I could get the camera calibration matrix "mtx" and the distortion coefficients "dst".
The code is as follow whose input is the original picture and it's out put is the calibration matrix and the distortion coefficients.
Also, in order to test whether the method is working normally, I used the calibration1.jpg to test the method and the output is as follow

```
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import sys


class Advanced_finding_lane_line:

    def __init__(self):
        self.Output_path = ''
        pass

    def file_abspath(self, path):
        current_path = os.path.dirname(sys.argv[0])
        file_abspath = current_path + path
        file_abspath = file_abspath.replace('/', '\\')
        return file_abspath


    def Calibration_Camera(self, img):
        relative_path = '/camera_cal'
        self.absolute_path = self.file_abspath(relative_path)
        images = glob.glob(self.absolute_path + '\\' + 'calibration*.jpg')
        objpoints = []
        imgpoints = []
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        for element in images:
            picture = cv2.imread(element)
            gray1 = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray1, (9, 6))
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.ret, self.mtx, self.dst, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                                                   None, None)
        Undistort = cv2.undistort(img,self.mtx,self.dst,None,self.mtx)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=14)
        ax2.imshow(Undistort)
        ax2.set_title('Undistorted Image', fontsize=14)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(self.Output_path)
        return self.mtx, self.dst

    def Pics_processing(self):
        Input_relative_path = '/test_images'
        Input_abs_path = self.file_abspath(Input_relative_path)
        Output_relative_path = '/Camera_Calibration_Output'
        Output_abs_path = self.file_abspath(Output_relative_path)
        pics = os.listdir(Input_abs_path)
        Mode = 2
        if Mode == 1:
            for element in pics:
                whole_path = Input_abs_path + '\\' + element
                self.Output_path = Output_abs_path + '\\' + element
                image = cv2.imread(whole_path)
        else:
            image_path = self.file_abspath('/camera_cal/calibration1.jpg')
            self.Output_path = Output_abs_path+'\\'+'calibration1.jpg'
            image = cv2.imread(image_path)
        Pic_processing.Calibration_Camera(image)

Pic_processing = Advanced_finding_lane_line()
Pic_processing.Pics_processing()
```
![Undistorted Image](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Camera_Calibration_Output/calibration1.jpg)

After testint the method of camera calibration, then I could use the images in the test_images folder as the input of the method.
The comparison picture is as follow:
![straight_lines1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Camera_Calibration_Output/straight_lines1.jpg)
![straight_lines2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Camera_Calibration_Output/straight_lines2.jpg)
![test1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Camera_Calibration_Output/test1.jpg)
![test2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Camera_Calibration_Output/test2.jpg)
![test3](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Camera_Calibration_Output/test3.jpg)
![test4](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Camera_Calibration_Output/test4.jpg)
![test5](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Camera_Calibration_Output/test5.jpg)
![test6](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Camera_Calibration_Output/test6.jpg)

### Threshold Method to identify the Lane-line 

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

