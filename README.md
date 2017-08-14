## Advanced lane detectin using cv

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./output_images/pers_transform.png "Road Transformed"
[image3]: ./output_images/binarize.png "Binary Example"
[image4]: ./output_images/curve_fit.png "Curve_fit Example"
[image5]: ./output_images/out_1.jpg "Output"
[video1]: ./result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Undistort the image

I used opencv library's calibrate function with provided checkboard images to undistort the camera image.  
I fixed world coordinates of the chess board corners to be every integer coordinates from 6 x 9 grid, on z=0 plane (i.e. (0,0), (1,0), (1,1), ..., (6,9)). I got image points by using `cv2.findChessboardCorners()` and used the image point and world point to calculate camera calibration and distortion coefficients with `cv2.calibrateCamera()`. The last step was to apply undistortion using `cv2.undistort()`.  The code can be found at `calibrate_from_dir` function, lines from 12 to 40 in utility.py.  

![image1]

### Pipeline (single images)

#### 1. Apply distortion-correction and transform image to bird-view.

I used camera calibration matrix and undistortion coefficients calculated using checkboard image to undistort road images.  
Then I transformed each image with `cv2.getPerspectiveTransform()`. The src and dst coordinates used were obtained by visually inspecting straight lane image. The codes can be found at `get_birdview`, lines from 43 to 69 in utility.py.  

![image2]

#### 2. Binarize the image with color mask and sobel threshold.  

While applying threshold to S channel from HLS color space worked well on clean cases, it was not robust enough to handle shadows. I tested several combinations of color mask and sobel thresholds. The best result I obtained so far was by combining white / yellow color mask, S channel mask, and sobel threshold. Then I clipped the left and right portion of the image to remove noise from other lanes. The codes for masking functions can be found at `utility.py` from line 72 to 115.

```python
def binary_mask(gray, hls):   
    white = white_mask(hls, sensitivity=50)
    yellow = yellow_mask(hls, sensitivity=150)
    color_mask = cv2.bitwise_or(white, yellow)
    color_mask[:, :300] = 0
    color_mask[:, 1100:] = 0 
    return color_mask
```

![alt text][image3]

#### 4. Identify lane pixel and fit lane curve.

I applied sliding windows to find lane pixels. I adjusted the center of the windows to be the mean of non-zero points inside the window. The code can be found at line 7 to 56 in lane_fitting.py, `sliding_window`.  
Then I fitted lane curve using `numpy.polyfit` on lane pixels. 

![image4]

#### 5. Calculate curvature of the lanes and car offset from lane center.

I assumed the conversion metrics from pixel to real-world to be 30 / 720 for y axis and 3.7 / 500 for x axis. The numbers were based US-road specification. I converted pixel coordination to world coordination then calculated lane curvature. I calculated car offset by assuming image center as car center then substracting it from center of two fitted lanes.   
The code can be found in lines 59 through 84 in my code in `lanefitting.py`

#### 6. Plot calculated lane back to original images.

I implemented this step in lines 87 through 110 in my code in `lanefitting.py` in the function `weighted_lane()`.  Here is an example of my result on a test image:

![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I wrote `continuous_pipeline()` in `lane_fitting.py`, lines 185 to 261 to process video.  
The result can be seen at:

[project_movie](https://github.com/trivus/CarND-Advanced-Lane-Lines/blob/master/result.mp4)  

[![youtube link](https://img.youtube.com/vi/szKzvvvDFxM/0.jpg)](https://www.youtube.com/watch?v=szKzvvvDFxM)  

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline worked considerably well on project video, but was not robust enough for challenge videos. I could adjust binarize function to work better on those videos, but not without sacrificing performance on project video.  
This is due to hard-coded thresholds in binarize function. The videos differ in overall brightness, shadows, noise etc. to extent that simple masking method can not handle. The problem may be mitigated by using more sophisticated CV techniques, such as shadow removal. I tested one simple shadow removal technique, which is to apply histogram equalization on YUV space's Y channel, which indeed improved pipelines'performance on shadowed images, but was not included in final pipeline as it messed up clean images.  
In conclusion, I found it hard to find an algorithm that works well on all three videos. I guess this is why convolution networks is gaining dominance today, as it can deal better with noisy sources.  


































