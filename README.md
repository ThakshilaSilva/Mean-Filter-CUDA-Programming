# Mean-Filter-CUDA-Programming
1. Perform Vector addition using both GPU and CPU
2. Implement a mean filter:

  Mean filter is a simple method used in image processing to reduce noise in images. Here there is a CUDA program to apply a 𝑘 𝑥 𝑘 kernel  filter to a given image (Gray Scale image) and produce the result. CPU (host) program reads the image file and convert to a raw image matrix. Then pass the matrix to the GPU and GPU kernel applies the mean filter to the image and return the result image back to the CPU. 
Additionally this contains the sequentional program (in c) for the same application.
