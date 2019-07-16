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

#### SobelX Threshold to generate the binary pictures to identify the lane-line.

The SobelX code part method is as follow:
```
 def abs_sobel_threshold(self,img,orient='x',thresholdmin=0, thresholdmax=255):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel = np.absolute((sobel))

        scale_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        sxbinary = np.zeros_like(scale_sobel)
        sxbinary[(scale_sobel>= thresholdmin)& (scale_sobel<=thresholdmax)] = 1
        return sxbinary
 ```

the generated comparison picture is as follow:
left is the Undistorted pic and the rightone is the sobelX binary pic.
![straight_lines1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Sobel_X_Threshold/straight_lines1.jpg)
![straight_lines2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Sobel_X_Threshold/straight_lines2.jpg)
![test1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Sobel_X_Threshold/test1.jpg)
![test2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Sobel_X_Threshold/test2.jpg)
![test3](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Sobel_X_Threshold/test3.jpg)
![test4](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Sobel_X_Threshold/test4.jpg)
![test5](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Sobel_X_Threshold/test5.jpg)
![test6](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Sobel_X_Threshold/test6.jpg)

The code of HLS_S Threshold method is as follow:
```
   def HSL_S_threshold(self,img,thresholdmin=90,thresholdmax=255):
        hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
        HLS_Binary = np.zeros_like(H)
        HLS_Binary[(S>thresholdmin)&(S<thresholdmax)] = 1
        return  HLS_Binary
```
The comparison image of Undistorted pic and HSL_S_Threshold binary pic is as follow:
![straight_lines1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_S_Binary/straight_lines1.jpg)
![straight_lines2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_S_Binary/straight_lines2.jpg)
![test1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_S_Binary/test1.jpg)
![test2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_S_Binary/test2.jpg)
![test3](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_S_Binary/test3.jpg)
![test4](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_S_Binary/test4.jpg)
![test5](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_S_Binary/test5.jpg)
![test6](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_S_Binary/test6.jpg)

The code of HLS_H Threshold method is as follow:
```
  def HSL_H_threshold(self, img, thresholdmin=15, thresholdmax=100):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H = hls[:, :, 0]
        L = hls[:, :, 1]
        S = hls[:, :, 2]
        HLS_Binary = np.zeros_like(H)
        HLS_Binary[(H > thresholdmin) & (H < thresholdmax)] = 1
        return HLS_Binary
  ```
 The comparison image of Undistorted pic and HSL_H_Threshold binary pic is as follow:
![straight_lines1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_H_Binary/straight_lines1.jpg)
![straight_lines2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_H_Binary/straight_lines2.jpg)
![test1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_H_Binary/test1.jpg)
![test2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_H_Binary/test2.jpg)
![test3](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_H_Binary/test3.jpg)
![test4](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_H_Binary/test4.jpg)
![test5](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_H_Binary/test5.jpg)
![test6](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/HLS_H_Binary/test6.jpg) 

The combination code of different throshold method is as follow:
```
        Sobel_binary = self.abs_sobel_threshold(undistort, 'x', 35, 100)
        HLS_Binary = self.HSL_S_threshold(undistort, 90, 255)
        HLS_H_Binary = self.HSL_H_threshold(undistort, 25, 100)
        Threshold = np.zeros_like(Sobel_binary)
        Threshold[(Sobel_binary == 1) | (HLS_Binary == 1) | (HLS_H_Binary == 1)] = 1
```
The comparison image of Undistorted pic and Combine threshold binary pic is as follow:
![straight_lines1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Threshold_Combine_Image/straight_lines1.jpg)
![straight_lines2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Threshold_Combine_Image/straight_lines2.jpg)
![test1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Threshold_Combine_Image/test1.jpg)
![test2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Threshold_Combine_Image/test2.jpg)
![test3](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Threshold_Combine_Image/test3.jpg)
![test4](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Threshold_Combine_Image/test4.jpg)
![test5](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Threshold_Combine_Image/test5.jpg)
![test6](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Threshold_Combine_Image/test6.jpg) 


### Perspective view of image:
The code for the perspective transform is as follow:
```
    def Perspective(self,img):

        self.img_size = (img.shape[1],img.shape[0])
        src_corners = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
        dist_corners = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
        self.M= cv2.getPerspectiveTransform(src_corners,dist_corners)
        self.Min = cv2.getPerspectiveTransform(dist_corners,src_corners)
        return self.M, self.Min
        # warped = cv2.warpPerspective(img,self.M,img_size)
        # print self.M
        # return warped
```
#### Original pics and perspective pics
![straight_lines1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Out_Original/straight_lines1.jpg)
![straight_lines2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Out_Original/straight_lines2.jpg)
![test1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Out_Original/test1.jpg)
![test2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Out_Original/test2.jpg)
![test3](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Out_Original/test3.jpg)
![test4](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Out_Original/test4.jpg)
![test5](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Out_Original/test5.jpg)
![test6](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Out_Original/test6.jpg) 
#### Combined threshold pics and perspective pics
![straight_lines1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Output/straight_lines1.jpg)
![straight_lines2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Output/straight_lines2.jpg)
![test1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Output/test1.jpg)
![test2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Output/test2.jpg)
![test3](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Output/test3.jpg)
![test4](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Output/test4.jpg)
![test5](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Output/test5.jpg)
![test6](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/Perspective_Output/test6.jpg) 

### Find the line points in the Threshold-Combined binary pic with the bird-eye view:
The code is as follow:
```
histogram = np.sum(Pespective_image[Pespective_image.shape[0] // 2:, :], axis=0)
out_img = np.dstack((Pespective_image, Pespective_image, Pespective_image)) * 255
midpoint = np.int(histogram.shape[0] // 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
nwindows = 9
margin = 100
minpix = 50
window_height = np.int(Pespective_image.shape[0] // nwindows)
nonzeros = Pespective_image.nonzero()
nonzeroy = np.array(nonzeros[0])
nonzerox = np.array(nonzeros[1])
leftx_current = leftx_base
rightx_current = rightx_base
left_lane_inds = []
right_lane_inds = []
for window in range(nwindows):
    win_y_low = Pespective_image.shape[0] - (window + 1) * window_height
    win_y_high = Pespective_image.shape[0] - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)
    # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox <= win_xleft_high) & (
                nonzerox > win_xleft_low)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox <= win_xright_high) & (
                nonzerox > win_xright_low)).nonzero()[0]
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
try:
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
except:
    pass
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
ploty = np.linspace(0, Pespective_image.shape[0] - 1, Pespective_image.shape[0])
try:
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
except TypeError:
    # Avoids an error if `left` and `right_fit` are still none or incorrect
    print('The function failed to fit a line!')
    left_fitx = 1 * ploty ** 2 + 1 * ploty
    right_fitx = 1 * ploty ** 2 + 1 * ploty
out_img[lefty, leftx] = [255, 0, 0]
out_img[righty, rightx] = [0, 0, 255]
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.imshow(out_img)
plt.show()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))
cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))
plt.imshow(out_img)
plt.show()
```
The related pics is as follow:
![image1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/processing_image/Figure_1.png)
![image2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/processing_image/window_figure.png)
![image3](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/processing_image/fill_the_lane.png)
### Calculate the radius of the lane and the vehicle position:
```
ym_per_pix = 30. / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
y_eval = np.max(ploty)
left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
curvature = ((left_curverad + right_curverad) / 2)
lane_width = np.absolute(left_fitx[719] - right_fitx[719])
lane_xm_per_pix = 3.7 / lane_width
veh_pos = (((left_fitx[719] + right_fitx[719]) * lane_xm_per_pix) / 2.)
cen_pos = ((out_img.shape[1] * lane_xm_per_pix) / 2.)
distance_from_center = veh_pos - cen_pos
```

### Change the image view back to normal view and put the radius info , radius info to the image
```
newwarp = cv2.warpPerspective(out_img, self.Min, (out_img.shape[1], out_img.shape[0]))
combine_pic = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
# plt.imshow(combine_pic)
# plt.show()
font = cv2.FONT_HERSHEY_SIMPLEX
radius_text = "Radius of Curvature: %sm" % (round(curvature))
if distance_from_center > 0:
    pos_flag = 'right'
else:
    pos_flag = 'left'
cv2.putText(combine_pic, radius_text, (100, 100), font, 1, (255, 255, 255), 2)
center_text = "Vehicle is %.3fm %s of center" % (abs(distance_from_center), pos_flag)
cv2.putText(combine_pic, center_text, (100, 150), font, 1, (255, 255, 255), 2)
# plt.imshow(combine_pic)
# plt.show()
cv2.imwrite(self.Output_path, combine_pic)
```
The related pics is as follow:
![image1](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/processing_image/back_perspective.png)
![image2](https://raw.githubusercontent.com/Michael0725/Udacity_Advanced_LaneLine_Finding/master/processing_image/write_the_info.png)

### The final output image is as follow:

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

