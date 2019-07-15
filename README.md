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
    def Calibration_Camera(self,img):

        relative_path = '/camera_cal'
        self.absolute_path = self.file_abspath(relative_path)
        images = glob.glob(self.absolute_path+'\\'+'calibration*.jpg')
        objpoints = []
        imgpoints = []
        objp = np.zeros((6*9,3),np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        for element in images:
            picture = cv2.imread(element)
            gray1 = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
            ret,corners = cv2.findChessboardCorners(gray1,(9,6))
            if ret ==True:
                objpoints.append(objp)
                imgpoints.append(corners)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.ret,self.mtx,self.dst,self.rvecs,self.tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
        Undistorted_Chessboard_pic = cv2.undistort(img,self.mtx,self.dst,None,self.mtx)
        Undistorted_pic_Outputpath = r'D:\01_Udacity_Self_Drivingcar_Program\Advanced_Camera_Calibration\CarND-Advanced-Lane-Lines-master\camera_cal\Undistorted.jpg'
        cv2.imwrite(Undistorted_pic_Outputpath,Undistorted_Chessboard_pic)
        print self.dst
        return self.mtx,self.dst
```
！[Undistorted Image]https://github.com/Michael0725/Udacity_Advanced_LaneLine_Finding/blob/master/camera_cal/Undistorted.jpg

A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

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

