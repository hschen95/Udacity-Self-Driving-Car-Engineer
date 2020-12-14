## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in ["./Project 2-Advanced Lane Finding.py"][/Project 2-Advanced Lane Finding.py].  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/camera_calibration.jpg" alt="camera_calibration" />


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the camera calibration and distortion coefficients computed above, distortion corrections can be applied to test images like this one:
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/camera_calibration_test/camera_calibration_straight_lines1.jpg"  alt="distortion_correction" />


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the `Gradient and color space` section in the IPython notebook `/Project 2-Advanced Lane Finding.ipynb`, I tried various color and gradient thresholds to choose how I wanted to perform thresholding to generate a binary image. 
##### Gradient thresholding
* Sobel absolute threshold: I applied Sobel x or y operators to the image with a threshold of between 20 and 100, the binary images are shown below:
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/absolute_sobel_test/straight_lines1.jpg" alt="sobel_abs" />
*Sobel magnitude threshold: This used the square root of combined squares of Soble x and y. By setting the kernel size to be 3 and threshold between 20 and 100, the following binary image was generated:
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/sobel_magnitude_thresholding_test/straight_lines1.jpg" alt="sobel_mag" />
*Direction of the gradient: With the Sobel magnitude threshold, I explored the the direction of the gradient. By setting the kernel size to be 3 and angle of the gradient away from horizontal limited between 0 and 1.4, the following binary image was generated:
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/direction_thresholding_test/straight_lines1.jpg" alt="dir_threshold" />
*Combining gradient threshold: I combined all the gradient thresholds(Sobel x, Sobel y, Sobel magnitude thresholding, and direction thresholding) and compare with other gradient thresholding methods:
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/sobel_direction_threshold_test/straight_lines1.jpg" alt="combine_gradient" />
By comparing all graient thresholding method, Sobel x operator can indicate the lane lines most clearly. 

##### Color space
The image was converted into HLS and threshold of between 120 and 255 was applied to H, L, and S channel respectively. 
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/color_thresh_test/straight_lines1.jpg" alt="color_thres" />
S channel was selected to be used in the later section, becasue it showed more contracts on the lane lines. 
##### Combined gradient and color thresholding
To create the binary image for detecting lane lines, I chose to use Sobel x operator and S channel thresholding. Threshold was set between 20 and 100 for Sobel x operator and S channel threshold was set to be between 160 and 255. The resulting binary image is shown below:
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/combined_gradient_color_test/straight_lines1.jpg" alt="combined_grad_color" />

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `PerspectiveTransformPipeline()`, which appears in the fourth code block in the file  `Project 2-Advanced Lane Finding.ipynb`.  The `PerspectiveTransformPipeline()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[580,455],[710,455],[1100,710],[190,710]])

offset = 200
dst = np.float32([[offset, 0],[img_size[0]-offset, 0],
                   [img_size[0]-offset, img_size[1]], [offset, img_size[1]]]) 
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 455      | 200, 0        | 
| 710, 455      | 1080, 0       |
| 1100, 710     | 1080, 720     |
| 190, 710      | 200, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/perspective_transform.jpg" alt="perspective_transform" />

By doing perspective transformation on the binary image generated from the previous step, the resulting wraped binary image was generated:
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/perspective_transform_binary.jpg" alt="wrap" />


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the `Finding the Lanes` section in the IPython notebook `Project 2-Advanced Lane Finding.ipynb`. After applying calibration, thresholding, and a perspective transform to a road image, a binary image was generated with the lane lines stand out clearly. In order to decide which pixels were part of the lines and which belong to the left line and which belong to the right line, a histogram was plotted and the two highest peaks in the histogram determined the position of the lines. 
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/histogram.png" alt="hist" />
With the position of two highest peaks show in the histogram, sliding windows technique was used to move upward along the road to determine where the lane lines go. The number of window used was 10, the widthe of each window was set to be 100, and the minimum number of pixels found to recenter window was 50. With sliding window methods, all pixels belong to each line could be found and then a polynomial can be fit to the line by using `np.polyfit()` method. Sliding windown output can be visualized as following:
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/slidingWindows.jpg" alt="sliding window" />
Since I have found the lines, sliding window was not necessary in the next frames of video. Instead of blind searching, previous lines can be used as a reference and just search in a margin around the previous lane line position. The green shaded area shows below is where we searched for the lines this time.
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/skip_slidingWindows.jpg" alt="skip_sliding_window" />



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `Determine the curvature of the lane and vehicle position with respect to center` section in `Project 2-Advanced Lane Finding.ipynb`. The radius of curvature for lane lines was calculated in meter. The left curvered radius is 2240.79m and right curvered radius is 935.42m for the lane lines in the test image.
The vehicle position with respect to center is calculated as car_position - lane_center_position. If the result was negative, then the vehicle was on the left; and if the result is positive, then the vehicle was on the right. For the vehicle in the test image, vehicle is 0.35m left of center. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `Warp the detected lane boundaries back onto the original image` and `Write Curvature Radius and Distance from Center Data onto the Original Image` section in `Project 2-Advanced Lane Finding.ipynb`. The dispalyed curvature was calculated by averaging the left curvature and right curvature. Here is an example of my result on a test image:
<div style="text-align: left"><img src="/p2-AdvancedLaneLines/output_images/FinalOutputImage.jpg" alt="final_output" />


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the problem is to find the correct source and destination locations for perspective transform, especially for the video stream. I have set the source and destination points to be fixed for all video, this make the video pipeline more likely to fail because a little difference in the source and desintation points can generate a big difference in the result. Maybe tune the source and destination points for different video stream to make the pipeline more robust. Also, more information could be used from frame to frame in the video stream to make the process more robust. 
