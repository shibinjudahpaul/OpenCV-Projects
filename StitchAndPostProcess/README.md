## Image Stitching and Post processing to generate a Panorama

The fundamentals of a typical image stitching algorithm require four key steps,
1. Detecting keypoints (DoG, Harris, etc.) and extracting local invariant descriptors (SIFT, SURF, etc.) from the input.
2. Matching the descriptors between the images.
3. Using the RANSAC algorithm to estimate a homography matrix using our matched feature vectors.
4. Applying a warping transformation using the homography matrix obtained in the above step.

## Key OpenCV Class / Methods 

1. `cv2.Stitcher_create()`  - This initializes a **class** that uses the method proposed by Brown and Lowe in their 2017 paper, 
    **Automatic Panoramic Image Stitching with Invariant Features.** This algorithm stands out as its not sensitive to the ordering 
    or the orientation of the images, illumination changes and noise.
   * `mode` - **PANORAMA** for photo panoramas or **SCANS** for composing scans and doesn't compensate for exposure by default.
  
2. `cv2.Stitcher.stitch()` - Stitches the given images using Brown and Lowe method along with gain composition and image blending techniques.
   * `images` - Input images
   * `masks` -	Masks for each input image specifying where to look for keypoints (optional).
   * `pano`	- Final panorama.
      
3. `cv2.copyMakeBorder()` - Copies the source image into the middle of the destination image. 
    The areas to the left, to the right, above and below the copied source image will be filled with extrapolated pixels.
   * `src`	- First input 2D point set.
   * `top`, `bottom`, `left`, `right`	- borders to be filled with extrapolated pixels.

4. `cv2.findContours()` - The function retrieves contours or edges from the binary image using the algorithm [Suzuki85].
    The contours are a useful tool for edge detection.
    * `images` - Input image
    * `mode` -	Contour retrieval mode.
    * `method`	- Contour approximation method.

 
 ## Sample Input
  [Input](./Images/2/)
 
 
 ## Sample Output

  [Ouput](./stitchedOutputProcessed.png)



