# **Finding Lane Lines on the Road** 


The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report




### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

This project used Python 3, OpenCV, Gaussian smoothing, grayscaling, Canny detection, region of interest, and Hough transofrmation to detect lane lines on the road. 
My lane finding pipeline consisted of 5 steps. The code can be found [here](https://github.com/hschen95/Udacity-Self-Driving-Car-Engineer/blob/main/p1-FindingLaneLines/Project%201-Finding%20Lane%20Lines%20on%20the%20Road.ipynb).

#### a. Grayscale the image

Since the image was in RGB format, converted the image into grayscale can speed up the precessing time. This is becasue processing a single channel image is faster than processing a three channel image. 
<div style="text-align: left"><img src="/p1-FindingLaneLines/write_up_images/grayscale_img.jpg" width="480" alt="Grayscale Image"/>

#### b. Reduce noise

With a kernel size of 3, Gauissan filter was applied on grayscaled image in order to smooth out the image and reduce noise.
<div style="text-align: left"><img src="/p1-FindingLaneLines/write_up_images/blur_img.jpg" width="480" alt="Blurred Image" />


#### c. Canny edge detector

Canny edge detection algorithm was applied to the blurred image for computing gradient in all directions of the image and tracing the edges with large change in intensity. 
<div style="text-align: left"><img src="/p1-FindingLaneLines/write_up_images/edges_img.jpg" width="480" alt="Edges Image" />

#### d. Mask the canny image

This step was to apply region of interest mask to filter out detected line segments in the image except the lane lines.
<div style="text-align: left"><img src="/p1-FindingLaneLines/write_up_images/masked_edges_img.jpg" width="480" alt="Masked Edge Image" />

#### e. Hough transformation

Hough transformation was used to detect straight lines in the masked image. Detected lane lines were drew on the original images.
<div style="text-align: left"><img src="/p1-FindingLaneLines/test_images_output/solidWhiteRight.jpg" width="480" alt="Line Detection" />

 
In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first differentiating left and right road lanes with the help of slopes of each line segements. Postive slope indicated the line belongs to right lane, and negative slope indicated the line belongs to the left lane. The lines were then averaged to get the best representation of lane position. 



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is current pipeline is not accurate when the road lane is curved. By implementing this pipeline using the challenge video, the pipeline could not accurately detect the road lane lines.
Another shortcoming could be parameters tuning. In the gaussian smoothing, canny detection algorithm, and hough transformation sections, the seclection of parameters would impact the accutacy of the pipeline. 


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use piece-wise straight line to fit the curve lane. This might improve the accuaracy of the pipeline when using it to detect curve lane lines. 
