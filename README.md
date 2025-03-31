# My-research-data

The data used in this study, measured by a stereo vision system, were primarily processed through experiments and algorithmic computations using the Python code in high_speed_displacement_numba. This code includes camera calibration for binocular vision, stereo 
rectification, image correction, and the implementation of the Newton-Raphson method, Zero-Normalized Sum of Squared Differences (ZNSSD), and Triangulation method for Digital Image Correlation (DIC) measurements. To enhance the computational efficiency of the algorithms, 
the code incorporates the Numba library, leveraging the GPU capabilities of the computer to improve performance.

Another codebase, chessboard_photo_separate, is designed to read images captured by the camera. Since each single image contains both left and right views, this code separates the image into distinct left and right images. After the image separation is completed, the 
resulting left and right images are fed into high_speed_displacement_numba for further computation.
