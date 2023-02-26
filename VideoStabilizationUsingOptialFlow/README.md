## Video Stabilization Using Point Feature Matching  

This digital form of stabilization is acheived in three broad steps, 
1. Motion estimation - Transformation parameters between two consecutive frames are calculated (Rigid Transforms)
2. Motion smoothing - Unwanted motion is removed by calculating a windowed moving average between frames and trajectory between the now smooth frames are calculated. 
3. Image composition - The stabilized video is reconstructed in accordance to the calculated trajectory matrix. 

## Key OpenCV methods 

1. `calcOpticalFlowPyrLK()`  - Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
   * `nextPts` - output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new         positions of input features in the second image.  
   * `status` - output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the                 corresponding features has been found, otherwise, it is set to 0.  
   * `err` -  outputs vector of errors, if the flow wasn’t found then the error is not defined.  
  
2. `estimateRigidTransform()` - Computes an optimal affine transformation between two 2D point sets.
   * `src` - First input 2D point set stored in std::vector or Mat, or an image stored in Mat.
   * `dst` -	Second input 2D point set of the same size and the same type as A, or another image.
   * `fullAffine`	- If true, the function finds an optimal affine transformation with no additional restrictions (6 degrees of freedom). 
      Otherwise, the class of transformations to choose from is limited to combinations of translation, rotation, and uniform scaling (4 degrees of freedom).
      
      **NOTE:**  OpenCV below 3.0.0
      
3. `estimateAffinePartial2D()` - Computes an optimal limited affine transformation with 4 degrees of freedom between two 2D point sets.
   * `from`	- First input 2D point set.
   * `to`	- Second input 2D point set.
   * `inliers`	- Output vector indicating which points are inliers.
     
     **NOTE:** OpenCV above 3.0.0
 
 ## Sample Output

  [Ouput](./stableFootage/Recorded Output_sample.mp4)




