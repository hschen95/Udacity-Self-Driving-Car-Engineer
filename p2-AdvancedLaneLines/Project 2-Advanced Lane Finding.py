#!/usr/bin/env python
# coding: utf-8

# ## Project : Advanced Lane Finding
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# ## Import Package

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Camera Calibration

# In[2]:


#Prepare object points
nx = 9
ny = 6
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)


def cal_undistort(img, gray, objpoints, imgpoints):
    # performs the camera calibration, image distortion correction and returns the undistorted image

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    
    return undist, mtx, dist


def CameraCalibrationPipeline(img):
    # Array to store object points and image points from all the image
    objpoints = []   #3D points in real world
    imgpoints = []   #2D points in image plane
    
    #Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    
    
    #Find the chessboard conrners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    #If corners are found, add obj points, img points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

    undist, mtx, dist = cal_undistort(img, gray, objpoints, imgpoints)
    return undist, mtx, dist

    
img = 'camera_cal/calibration2.jpg'
image = mpimg.imread(img)

undistorted, mtx, dist= CameraCalibrationPipeline(image)

#Plot original image and undistorted image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=35)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=35)

f.savefig('output_images/camera_calibration.jpg')


pickle.dump( { 'mtx': mtx, 'dist': dist }, open('./saved_data/camera_calibration.p', 'wb'))


# In[3]:


# test camera calibration pipline using test_images

test_images = os.listdir('test_images/')

for test_images in os.listdir('test_images/'):
    img = mpimg.imread('test_images/' + test_images)

    undist = cv2.undistort(img, mtx, dist, None, mtx)    
    #Plot original image and undistorted image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=35)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=35)

    f.savefig('output_images/camera_calibration_test/camera_calibration_' + test_images)



    


# ### Perspective transform

# In[4]:


#camera matrix and distortion coefficients
dist_pickle = pickle.load( open( './saved_data/camera_calibration.p', 'rb' ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def PerspectiveTransformPipeline(img, src, dst):
    img_siz = (img.shape[1], img.shape[0])
    
    #Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    
    #Compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    #wrap the image using perspective transform
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M


img =  mpimg.imread('test_images/straight_lines1.jpg')

#Undistort using mtx and dist
undist = cv2.undistort(img, mtx, dist, None, mtx)
img_size = (img.shape[1], img.shape[0])

#Four source coordinates
src = np.float32([[580,455],[710,455],[1100,710],[190,710]])

#Four desired coordinates
offset = 200
dst = np.float32([[offset, 0],[img_size[0]-offset, 0],
                   [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])  
 

img_copy = img.copy()

cv2.line(img_copy, (580,455),(710,455), (255, 0, 0), 2)
cv2.line(img_copy, (1100,710),(190,710), (255, 0, 0), 2)
cv2.line(img_copy, (580,455), (190,710), (255, 0, 0), 2)
cv2.line(img_copy, (1100,710), (710,455), (255, 0, 0), 2)


top_down, perspective_M = PerspectiveTransformPipeline(img, src, dst) 
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=40)
ax2.imshow(img_copy)
ax2.set_title('Region of Interest', fontsize=40)
ax3.imshow(top_down)
ax3.set_title('Undistorted and Warped Image', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
f.savefig('output_images/perspective_transform.jpg')



pickle.dump( { 'perspective_M': perspective_M }, open('./saved_data/perspective_transform.p', 'wb'))


# 
# ### Gradient and color space

# #### Sobel thresholding and magnitude thresholding

# In[5]:


#Sobel Absolute Threshold, a function that applies Sobel x or y
def abs_sobel_thresh(img, orient, thresh):
    #grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    
    #Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    #Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    
    return binary_output



#Sobel Magnitude Threshold, a function that applies Sobel x and y
def mag_thresh(img, sobel_kernel, mag_thresh):
  
    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    #Calculate the magnitude 
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    
    #Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    #Return this mask as your binary_output image
    return binary_output


# Applies Sobel magnitude threshold and computes the direction of the gradient and applies a threshold.
def dir_threshold(img, sobel_kernel, thresh):
        # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    

    return binary_output


# In[6]:


#Absolute Sobel Thresholding
for test_images in os.listdir('test_images/'):
    img = mpimg.imread('test_images/' + test_images)
    sobelx_output = abs_sobel_thresh(img, orient = 'x', thresh = (20, 100))
    sobely_output = abs_sobel_thresh(img, orient = 'y', thresh = (20, 100))
    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(sobelx_output, cmap='gray')
    ax2.set_title('Thresholded x-derivative', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    ax3.imshow(sobely_output, cmap='gray')
    ax3.set_title('Thresholded y-derivative', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig('output_images/absolute_sobel_test/' + test_images)




 


# In[7]:


#Sobel Magnitude Thresholding
for test_images in os.listdir('test_images/'):
    img = mpimg.imread('test_images/' + test_images)
    sobel_output = mag_thresh(img, sobel_kernel = 3, mag_thresh = (20, 100))    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(sobel_output, cmap='gray')
    ax2.set_title('Thresholded Magnitude', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig('output_images/sobel_magnitude_thresholding_test/' + test_images)

 


# In[25]:




#Direction of magnitude Thresholding
for test_images in os.listdir('test_images/'):
    img = mpimg.imread('test_images/' + test_images)
    dir_binary = dir_threshold(image, sobel_kernel=3, thresh=(0, 1.4))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dir_binary, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig('output_images/direction_thresholding_test/' + test_images)


# In[26]:


#Combine Thresholds and compare all
for test_images in os.listdir('test_images/'):
    img = mpimg.imread('test_images/' + test_images)
    gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(20, 100))
    dir_binary = dir_threshold(image, sobel_kernel=3, thresh=(0,1.4))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(sobelx_output, cmap='gray')
    ax2.set_title('Thresholded x-derivative', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    ax3.imshow(sobely_output, cmap='gray')
    ax3.set_title('Thresholded y-derivative', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    ax4.imshow(sobel_output, cmap='gray')
    ax4.set_title('Thresholded Magnitude', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    ax5.imshow(dir_binary, cmap='gray')
    ax5.set_title('Thresholded Grad. Dir.', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    ax6.imshow(combined, cmap='gray')
    ax6.set_title('Combined Thresholds', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    f.savefig('output_images/sobel_direction_threshold_test/' + test_images)
    


# #### Color space

# In[10]:


image = mpimg.imread('test_images/test1.jpg')
#RGB
R = image[:,:,0]
G = image[:,:,1]
B = image[:,:,2]
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image_test1' , fontsize=45)
ax2.imshow(R, cmap='gray')
ax2.set_title('R', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
ax3.imshow(G, cmap='gray')
ax3.set_title('G', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
ax4.imshow(B, cmap='gray')
ax4.set_title('B', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


#HLS
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image_test1', fontsize=45)
ax2.imshow(H, cmap='gray')
ax2.set_title('H', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
ax3.imshow(L, cmap='gray')
ax3.set_title('L', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
ax4.imshow(S, cmap='gray')
ax4.set_title('S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


#LAB
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
L = hls[:,:,0]
A = hls[:,:,1]
B = hls[:,:,2]
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image_test1', fontsize=45)
ax2.imshow(H, cmap='gray')
ax2.set_title('L', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
ax3.imshow(L, cmap='gray')
ax3.set_title('A', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
ax4.imshow(S, cmap='gray')
ax4.set_title('B', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# #### Color thresholds

# In[31]:


def color_thres(img, colorSpace, colorChannel, thresh):
    if colorSpace == 'HLS':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if colorChannel == 'H':
            channel = img[:,:,0]
        elif  colorChannel == 'L':
            channel = img[:,:,1]
        else: 
            channel = img[:,:,2]
    elif colorSpace == 'RGB':
        if colorChannel == 'R':
            channel = img[:,:,0]
        elif colorChannel == 'G':
            channel = img[:,:,1]
        else:
            channel = img[:,:,2]
    elif colorSpace == 'LAB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        if colorChannel == 'L':
            channel = img[:,:,0]
        elif colorChannel == 'A':
            channel = img[:,:,1]
        else:
            channel = img[:,:,2]
    elif colorSpace == 'LUV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        if colorChannel == 'L':
            channel = img[:,:,0]
        elif colorChannel == 'U':
            channel = img[:,:,1]
        else:
            channel = img[:,:,2]
    
    #Apply threshold to selected channel
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel < thresh[1])] = 1
    return binary_output


# In[12]:


for test_images in os.listdir('test_images/'):
    img = mpimg.imread('test_images/' + test_images)
    thresh = (120,255)
    h_threh = color_thres(img, 'HLS', 'H', thresh)
    l_threh = color_thres(img, 'HLS', 'L', thresh)
    s_threh = color_thres(img, 'HLS', 'S', thresh)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(h_threh, cmap='gray')
    ax2.set_title('Thresholded H', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    ax3.imshow(l_threh, cmap='gray')
    ax3.set_title('Thresholded L', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    ax4.imshow(s_threh, cmap='gray')
    ax4.set_title('Thresholded S', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig('output_images/color_thresh_test/' + test_images)


# #### Combine color and gradient 

# In[13]:


#Convert to HLS space and use S channel
#Sobel x and threshold x gradient

for test_images in os.listdir('test_images/'):
    img = mpimg.imread('test_images/' + test_images)
    thresh = (160,255)
    s_threh = color_thres(img, 'HLS', 'S', thresh)
    gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_threh)
    combined_binary[(gradx == 1) | (s_threh == 1)] = 1
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('Original Image')
    ax1.imshow(img)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    ax2.set_title('Combined S channel and x-gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig('output_images/combined_gradient_color_test/' + test_images)





# ### Perspective transform on binary image

# In[34]:


dist_pickle = pickle.load( open( './saved_data/camera_calibration.p', 'rb' ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


img =  mpimg.imread('test_images/straight_lines1.jpg')

#Undistort using mtx and dist
undist = cv2.undistort(img, mtx, dist, None, mtx)
img_size = (img.shape[1], img.shape[0])

#Convert image to binary image using color and gradient threshold 

l_threh = color_thres(img, 'LUV', 'L', thresh=(215,255))
b_threh = color_thres(img, 'LAB', 'B', thresh=(145,200))
gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 100))

# Combine the two binary thresholds
combined_binary = np.zeros_like(s_threh)
combined_binary[(gradx == 1) | (l_threh == 1) | (b_threh == 1)] = 1


#Four source coordinates
src = np.float32([[580,455],[710,455],[1100,710],[190,710]])
#Four desired coordinates

offset = 200
dst = np.float32([[offset, 0],[img_size[0]-offset, 0],
                   [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])  
            
top_down, perspective_M = PerspectiveTransformPipeline(combined_binary, src, dst) 


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down, cmap = 'gray')
cv2.imwrite('output_images/perspective_transform_binary_solo.jpg', top_down)

ax2.set_title('Undistorted and Warped Binary Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
f.savefig('output_images/perspective_transform_binary.jpg')


# ### Finding the Lanes

# In[35]:



#Find peaks in histogram
def hist(binary_wraped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    return histogram
binary_warped = mpimg.imread('output_images/perspective_transform_binary_solo.jpg')
histogram = hist(binary_warped)
plt.plot(histogram)
cv2.imwrite('output_images/histogram.jpg', histogram)


# In[36]:


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows 
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)


    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img, left_lane_inds, right_lane_inds


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img, left_lane_inds, right_lane_inds = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    ## Visualization ##

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, ploty, left_lane_inds, right_lane_inds

binary_warped = top_down
out_img, left_fit, right_fit, ploty, left_lane_inds, right_lane_inds = fit_polynomial(binary_warped)
plt.imshow(out_img)
cv2.imwrite('output_images/slidingWindows.jpg', out_img)
pickle.dump( { 'left_fit': left_fit, 'right_fit': left_fit, 'ploty':ploty }, open('./saved_data/poly_fit_slidingwindow.p', 'wb'))


# In[37]:


def fit_poly(binary_warped, leftx, lefty, rightx, righty):
    img_shape = binary_warped.shape

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit

#Use previous polynomial to skip the sliding windows
def search_around_poly(binary_warped, left_fit, right_fit):
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #Set the area of search based on activated x-values 
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
                    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped, leftx, lefty, rightx, righty)
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    
    return result, left_fit, right_fit, ploty, left_lane_inds, right_lane_inds


result, left_fit, right_fit, ploty, left_lane_inds, right_lane_inds = search_around_poly(binary_warped, left_fit, right_fit)

plt.imshow(result)
cv2.imwrite('output_images/skip_slidingWindows.jpg', result)
pickle.dump( { 'left_fit': left_fit, 'right_fit': left_fit, 'ploty':ploty }, open('./saved_data/poly_fit_skip_slidingwindow.p', 'wb'))



# ### Determine the curvature of the lane and vehicle position with respect to center

# In[38]:


#Calculates the curvature of polynomial functions in meters
def measure_curvature_pixels(left_fit, right_fit, ploty):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty+left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty+right_fit[2]
    
    leftx *= xm_per_pix
    rightx *= xm_per_pix
    ploty *= ym_per_pix
    
    
    #Fit bew polynoimal to x, y in world space
    left_fit_cr = np.polyfit(ploty, leftx, 2)
    right_fit_cr = np.polyfit(ploty, rightx, 2)
    
    y_eval = np.max(ploty)
    #Calcaulte the radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    
    
    return left_curverad, right_curverad
    
left_curverad, right_curverad = measure_curvature_pixels(left_fit, right_fit, ploty) 
print('left curvered radius:', left_curverad, 'm;')
print('right curvered radius:',right_curverad, 'm')
    


# In[39]:


#Calculate vehicle position with respect to center
def vehicle_pos(img, left_fit, right_fit, ploty):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)


    leftx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval+left_fit[2]
    rightx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval+right_fit[2]
    
    car_position = img.shape[1]/2
    lane_center_position = (rightx + leftx) /2
    car_pos_repect_center = (car_position - lane_center_position) * xm_per_pix
    return car_pos_repect_center
car_pos_repect_center = vehicle_pos(top_down, left_fit, right_fit, ploty)
print('Vehicle position respect to center: ', car_pos_repect_center, 'm')


# ### Warp the detected lane boundaries back onto the original image

# In[40]:


def DrawLine(original_image,binary_warped, left_fit, right_fit):
    
    h,w= binary_warped.shape
    Minv = cv2.getPerspectiveTransform(dst, src)
    ploty = np.linspace(0, h-1, num=h)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return result

original_image =  mpimg.imread('test_images/straight_lines1.jpg')
originalImage=cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
binary_wraped = mpimg.imread('output_images/perspective_transform_binary_solo.jpg')
result = DrawLine(original_image,binary_warped, left_fit, right_fit)
plt.imshow(result)
cv2.imwrite('output_images/backtoOriginalImage.jpg', result)


# ### Write Curvature Radius and Distance from Center Data onto the Original Image

# In[41]:


def draw_cirvature(original_img, curv, vehicle_pos):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    text = 'Curve radius: ' + '{:2f}'.format(curv) + 'm'
    cv2.putText(new_img, text, (40,70), cv2.FONT_HERSHEY_DUPLEX,1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if vehicle_pos > 0:
        direction = 'right'
    elif vehicle_pos < 0:
        direction = 'left'
    abs_center_dist = abs(vehicle_pos)
    text = 'Vehicle is '+ '{:2f}'.format(abs_center_dist) + 'm '+ direction + ' of center'
    cv2.putText(new_img, text, (40,120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return new_img

curv = (left_curverad+right_curverad)/2
new_img = draw_cirvature(mpimg.imread('output_images/backtoOriginalImage.jpg'), curv, car_pos_repect_center)
plt.imshow(new_img)
cv2.imwrite('output_images/FinalOutputImage.jpg', new_img)





# ### Video

# In[42]:


# Define a class to store properties of each line 
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.is_detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #number of detected pixels
        self.px_count = None
    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or                self.diffs[1] > 1.0 or                self.diffs[2] > 100.) and                len(self.current_fit) > 0:
                # bad fit! 
                self.is_detected = False
            else:
                self.is_detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.is_detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)
                
                


# In[44]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def Pipeline(img):
    dist_pickle = pickle.load( open( './saved_data/camera_calibration.p', 'rb' ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    undist = cv2.undistort(img, mtx, dist, None, mtx) 
    
    img_size = (img.shape[1], img.shape[0])
    #Convert image to binary image using color and gradient threshold 
    l_threh = color_thres(img, 'LUV', 'L', thresh=(215,255))
    b_threh = color_thres(img, 'LAB', 'B', thresh=(145,200))
    gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 100))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_threh)
    combined_binary[(gradx == 1) | (b_threh == 1) | (l_threh == 1)] = 1


    #Four source coordinates
    src = np.float32([[580,455],[710,455],[1100,710],[190,710]])

    #Four desired coordinates
    offset = 200
    dst = np.float32([[offset, 0],[img_size[0]-offset, 0],
                       [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])  

    top_down, perspective_M = PerspectiveTransformPipeline(combined_binary, src, dst) 
    
    
     # if both left and right lines were detected last frame, 
    # use polyfit_using_prev_fit, otherwise use sliding window fit
    if not l_line.is_detected or not r_line.is_detected:
        out_img, left_fit, right_fit, ploty, left_lane_inds, right_lane_inds = fit_polynomial(top_down)
    else:
        out_img, left_fit, right_fit, ploty, left_lane_inds, right_lane_inds = search_around_poly(top_down, l_line.best_fit, r_line.best_fit)
    
   
    
    l_line.add_fit(left_fit, left_lane_inds)
    r_line.add_fit(right_fit, right_lane_inds)
    
        
    left_curverad, right_curverad = measure_curvature_pixels(l_line.best_fit, r_line.best_fit, ploty) 
    curv = (left_curverad+right_curverad)/2
    car_pos_repect_center = vehicle_pos(top_down, l_line.best_fit, r_line.best_fit, ploty)
    result = DrawLine(img , top_down, l_line.best_fit, r_line.best_fit)
    new_output = draw_cirvature(result, curv, car_pos_repect_center)

        
    return new_output
    
    
    
    
    
    
        


# In[45]:


l_line = Line()
r_line = Line()
output = 'project_video_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(Pipeline) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(output, audio=False)')


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))


# In[ ]:




