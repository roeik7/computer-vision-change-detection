## **computer-vision-change-detection**

The algorithms gets a video and compute the change detection in differents ways. 
First i used as background the median of a set of k1 frames, and update it every k2 frames. 
The algorithm work on color images and merge the different channels (colors). 
camera is static. 
The output is a video where the pixels of the foreground objects consists of the original frame, and the other pixels are black.

After running the basic change detection algorithm (median), the algorithm execute sequence of post processing operation to improve the results, such as dilation, erosion
and connected components.

After running the post processing operations:</br>
![image](https://user-images.githubusercontent.com/48287470/105815686-fe020380-5fbb-11eb-8eca-0fe4778009c4.png)

The second algorithm (LK-Change-Detection) compute the change detection between two frames and its based on Lucas Kanade algorithm for compute optical flow.

Attached is a reference to the description of the algorithm from Wikipedia- 
https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method

One of the frames:</br>
![wrap_res](https://user-images.githubusercontent.com/48287470/105816363-dfe8d300-5fbc-11eb-852c-d9230c922a3c.jpg)

After running LK Change Detection:</br>
![image](https://user-images.githubusercontent.com/48287470/105815744-107c3d00-5fbc-11eb-828d-b5fbeabc279f.png)


This is results of another algorithm for compute the change detection and its based on assumption that is affine motion: </br>
Original Frame:</br>
![image](https://user-images.githubusercontent.com/48287470/105815813-2689fd80-5fbc-11eb-9a48-212f03c3db11.png)

After compute change detection:</br>
![image](https://user-images.githubusercontent.com/48287470/105815767-183be180-5fbc-11eb-968f-230a44221f6b.png)
