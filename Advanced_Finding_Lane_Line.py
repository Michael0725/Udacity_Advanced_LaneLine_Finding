import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import sys
from moviepy.editor import VideoFileClip
from IPython.display import HTML

class Advanced_finding_lane_line:

    def __init__(self):
        pass


    def file_abspath(self,path):

        current_path = os.path.dirname(sys.argv[0])
        file_abspath = current_path + path
        file_abspath = file_abspath.replace('/', '\\')
        return file_abspath

    def Calibration_Camera(self,img):

        relative_path = '/CarND-Advanced-Lane-Lines-master/camera_cal'
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
        return self.mtx,self.dst

        # distorted_image = cv2.undistort(img,mtx,dst,None,mtx)
        # return  distorted_image


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

    def Magnitude_Grad(self,img,thresholdmin = 0,thresholdmax =255):
        gray_Grad = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_Grad,cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_Grad,cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.square(sobelx**2+sobely**2)
        scale_factor = np.max(grad_mag)/255
        grad_mag = (grad_mag/scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(grad_mag)
        binary_output[(grad_mag>=thresholdmin)&(grad_mag<=thresholdmax)] = 1
        return binary_output

    def dir_threshold(self,img,thresholdmin =0.7,thresholdmax =1.3 ):
        gray_Grad = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_Grad, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_Grad, cv2.CV_64F, 0, 1, ksize=3)
        absgrad_dir = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
        binary = np.zeros_like(absgrad_dir)
        binary[(absgrad_dir>=thresholdmin) &(absgrad_dir<=thresholdmax)] =1
        return  binary

    def HSL_S_threshold(self,img,thresholdmin=90,thresholdmax=255):
        hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
        HLS_Binary = np.zeros_like(H)
        HLS_Binary[(S>thresholdmin)&(S<thresholdmax)] = 1
        return  HLS_Binary

    def HSL_H_threshold(self, img, thresholdmin=15, thresholdmax=100):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H = hls[:, :, 0]
        L = hls[:, :, 1]
        S = hls[:, :, 2]
        HLS_Binary = np.zeros_like(H)
        HLS_Binary[(H > thresholdmin) & (H < thresholdmax)] = 1
        return HLS_Binary

    def Single_pic_processing(self,image):


        # abs_path = r'D:\01_Udacity_Self_Drivingcar_Program\Advanced_Camera_Calibration\CarND-Advanced-Lane-Lines-master\test_images\straight_lines1.jpg'
        # image = cv2.imread(abs_path)
        # plt.imshow(image)
        # plt.show()

        # undistort = Advanced_finding_lane_line().Calibration_Camera(image)
        undistort = cv2.undistort(image,self.mtx,self.dst,None,self.mtx)
        # plt.imshow(undistort)
        # plt.show()
        Sobel_binary = self.abs_sobel_threshold(undistort, 'x', 35, 100)
        # plt.imshow(Sobel_binary)
        # plt.show()
        HLS_Binary = self.HSL_S_threshold(undistort, 90, 255)
        # plt.imshow(HLS_Binary)
        # plt.show()
        HLS_H_Binary = self.HSL_H_threshold(undistort, 25, 100)
        # plt.imshow(HLS_H_Binary)
        # plt.show()
        Magnitude_binary = self.Magnitude_Grad(undistort)
        Dir_binary = self.dir_threshold(undistort)
        Threshold = np.zeros_like(Sobel_binary)
        Threshold[(Sobel_binary == 1) | (HLS_Binary == 1) | (HLS_H_Binary == 1)] = 1
        Pespective_image = cv2.warpPerspective(Threshold,self.M,self.img_size)
        # plt.imshow(Pespective_image)
        # plt.show()

        histogram = np.sum(Pespective_image[Pespective_image.shape[0] // 2:, :], axis=0)
        # plt.plot(histogram)
        # plt.show()
        out_img = np.dstack((Pespective_image, Pespective_image, Pespective_image)) * 255
        # print out_img
        # print out_img.shape
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
        # print leftx
        # print lefty

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
        # plt.imshow(out_img)
        # plt.show()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))
        # plt.imshow(out_img)
        # plt.show()
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
        # print image.shape[0]
        # print image.shape[1]
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
        return combine_pic

    def Pics_processing(self):
        Input_relative_path = '/CarND-Advanced-Lane-Lines-master/test_images'
        Input_abs_path = self.file_abspath(Input_relative_path)
        Output_relative_path = '/CarND-Advanced-Lane-Lines-master/output_images'
        Output_abs_path = self.file_abspath(Output_relative_path)
        pics = os.listdir(Input_abs_path)
        for element in pics:
            whole_path = Input_abs_path + '\\' + element
            Advanced_finding_lane_line.Output_path = Output_abs_path + '\\' + element
            image = cv2.imread(whole_path)
            Pic_processing.Calibration_Camera(image)
            Pic_processing.Perspective(image)
            Pic_processing.Single_pic_processing(image)


Pic_processing = Advanced_finding_lane_line()
Pic_processing.Pics_processing()
Video_Output_path = '/CarND-Advanced-Lane-Lines-master/Output_Video'
Video_Input_path = '/CarND-Advanced-Lane-Lines-master'
white_output =Pic_processing.file_abspath(Video_Output_path)+'\\'+'Output1.mp4'
Video_input = Pic_processing.file_abspath(Video_Input_path)+'\\'+'project_video.mp4'
clip1 = VideoFileClip(Video_input)
white_clip = clip1.fl_image(Pic_processing.Single_pic_processing)
white_clip.write_videofile(white_output,audio=False)
