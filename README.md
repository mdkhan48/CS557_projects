# CS557_projects
CS 557 course codes. 
# Project 1: Camera Calibration
In this project, you will implement camera calibration as described in your textbook and lectures. You will be given
information about a calibration rig and a picture of the rig as part of the given data for this project.
# Project 2: Low level image processing
- Implement a histogram equalization program that flattens the histogram of the input image as described in lecture by creating a mapping function c(I) based on the histogram of the original image.
- Implement a function that does a log mapping of the input image (thus enhancing the detail in dark regions).
- Write a function that will take an input angle and produce an output image which is the input image rotated by around the image center. Normally is positive if the rotation is counter-clockwise (CCW) and negative otherwise. If the pixels in the output image do not correspond to a rotated pixel of the input image, then set their value to 0 (black).
- Implement the Gaussian averaging filter. You will have to design the filter mask (integer approximation or floating point samples; it’s up to you). You may want to have sigma of the Gaussian filter as an input parameter that you can easily vary. This will let you experiment with different width Gaussian filters. The filter mask size may depend on the value of sigma.
- Implement the median filter. Use 3 x 3 neighborhood.
