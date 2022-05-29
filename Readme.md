

# Image Stitching using openCV from Scratch 🖼️

The goal of this project is to stitch two images (named “left.jpg” and “right.jpg”) together to construct a panorama image.   Image stitching is  **the combination of images with overlapping sections to create a single panoramic or high-resolution image**.

Panoramic photography is a technique that combines multiple images from the same rotating camera to form a single, wide photo. It combines images based on their matching features by the process called image stitching.

<p align="center">
<img src="https://github.com/sajmaru/Image-Stitching-OpenCV/blob/main/stitching_example.jpeg" height = 200 width = 500>
</p>

##  Steps 🪜

-   Detect and match features.
-   Compute homography (perspective transform between frames) using **RANSAC** algorithm.
-   Warp one image onto the other perspective.
-   Combine the base and warped images while keeping track of the shift in origin.
-   Given the combination pattern, stitch multiple images.  


## Results 🚀

<p align="center">
<img src="https://github.com/sajmaru/Image-Stitching-OpenCV/blob/main/Keypoints.png" height = 200 width = 500>
</p>

<p align="center">
<img src="https://github.com/sajmaru/Image-Stitching-OpenCV/blob/main/Keypoints_Mapped.png" height = 200 width = 500>
</p>

<p align="center">
<img src="https://github.com/sajmaru/Image-Stitching-OpenCV/blob/main/Stitched%20Output.jpeg" height = 200 width = 500>
</p>