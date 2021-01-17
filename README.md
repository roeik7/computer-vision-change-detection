## **computer-vision-change-detection**

The algorithms gets a video and compute the change detection in differents ways. 
First i used as background the median of a set of k1 frames, and update it every k2 frames. 
The algorithm work on color images and merge the different channels (colors). 
camera is static. 
The output is a video where the pixels of the foreground objects consists of the original frame, and the other pixels are black.

After running the basic change detection algorithm (median), the algorithm execute sequence of post processing operation to improve the results, such as dilation, erosion
and connected components.

The second algorithm for compute the change detection is based on Lucas Kanade algorithm for compute optical flow.

Attached is a reference to the description of the algorithm from Wikipedia- 
https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method
