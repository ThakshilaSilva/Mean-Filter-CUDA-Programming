#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_profiler_api.h>
#include <tuple>
#include <iostream>
#include <string.h>

double time_h = 0;
double time_d = 0;

int numOfRounds = 5;

void meanFilter_h (unsigned char* raw_image, unsigned char* filtered_image, int img_width, int img_height, int window_size)
{
	int half_window = (window_size - 1) / 2;
	
	for (int i=0; i < img_height; i++)
	{
		for(int j=0; j < img_width; j++)
		{
			int left_limit, right_limit, top_limit, bottom_limit;
			
			if(j - half_window >= 0){
				left_limit = j-half_window;
			}else{
				left_limit = 0;
			}
			
            if(j + half_window <= img_width-1){
				right_limit = j + half_window;
			}else{
				right_limit = img_width-1;
			}
			
			if(i - half_window >= 0){
				top_limit = i - half_window;
			}else{
				top_limit = 0;
			}
			
            if(i + half_window <= img_height-1){
				bottom_limit = i + half_window;
			}else{
				bottom_limit = img_height-1;
			}
			
			double sum = 0;
			for(int k = top_limit; k <= bottom_limit; k++)
			{
				for(int m = left_limit; m <= right_limit; m++)
				{
					sum += raw_image[(k * img_height) + m];
				}
			}
			int current_window_size = (bottom_limit - top_limit + 1) * (right_limit - left_limit + 1);
			filtered_image[i*img_height + j] = sum / current_window_size; 
		}
	}
}

__global__ void meanFilter_d (unsigned char* raw_image, unsigned char* filtered_image, int img_width, int img_height, int window_size)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	int half_window = (window_size - 1) / 2;
	
	if (i < img_height && j < img_width)
	{
		int left_limit, right_limit, top_limit, bottom_limit;
			
		if(j - half_window >= 0){
			left_limit = j-half_window;
		}else{
			left_limit = 0;
		}
			
        if(j + half_window <= img_width-1){
			right_limit = j + half_window;
		}else{
			right_limit = img_width-1;
		}
			
		if(i - half_window >= 0){
			top_limit = i - half_window;
		}else{
			top_limit = 0;
		}
			
        if(i + half_window <= img_height-1){
			bottom_limit = i + half_window;
		}else{
			bottom_limit = img_height-1;
		}
		
		double sum = 0;
		for(int k = top_limit; k <= bottom_limit; k++)
		{
			for(int m = left_limit; m <= right_limit; m++)
			{
				sum += raw_image[(k * img_height) + m];
			}
		}
		int current_window_size = (bottom_limit - top_limit + 1) * (right_limit - left_limit + 1);
		filtered_image[i*img_height + j] = sum / current_window_size;
	}
}

int main(int argc,char **argv)
{
    printf("Begin......\n");
    
	//get bitmap to a char array
    FILE* file = fopen(argv[1], "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, file);

    int width, height;
    memcpy(&width, info + 18, sizeof(int));
    memcpy(&height, info + 22, sizeof(int));

    int window_size = strtol(argv[2],NULL,10);
        
    int size = 3 * width * abs(height);
    unsigned char* inputImage = new unsigned char[size];
    unsigned char* result_image_data_d;
    unsigned char* result_image_data_h = new unsigned char[size];
    unsigned char* result_image_data_h1 = new unsigned char[size];

    unsigned char* image_data_d;

    fread(inputImage, sizeof(unsigned char), size, file);
    fclose(file);
 
    int block_size = 32;
    int grid_size = width/block_size;
	
    dim3 dimBlock(block_size, block_size, 1);
    dim3 dimGrid(grid_size, grid_size, 1);
    
    for(int x = 0; x < numOfRounds; x += 1)
    {
        cudaMalloc((void **)&image_data_d,size*sizeof(unsigned char));
        cudaMalloc((void **)&result_image_data_d,size*sizeof(unsigned char));

        cudaMemcpy(image_data_d,inputImage,size*sizeof(unsigned char),cudaMemcpyHostToDevice);

        clock_t start_d=clock();
		//execution of GPU code
        meanFilter_d <<< dimGrid, dimBlock >>> (image_data_d, result_image_data_d, width, height, window_size);
        cudaThreadSynchronize();

        cudaError_t error = cudaGetLastError();
        if(error!=cudaSuccess)
        {
            fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
            exit(-1);
        }
		
        clock_t end_d = clock();

        clock_t start_h = clock();
		//executing CPU code
        meanFilter_h(inputImage, result_image_data_h1, width, height, window_size);
        clock_t end_h = clock();

        time_h += (double)(end_h-start_h)/CLOCKS_PER_SEC;
        time_d += (double)(end_d-start_d)/CLOCKS_PER_SEC;

        cudaFree(image_data_d);
        cudaFree(result_image_data_d);
    }

    printf("Average GPU execution time: %f\n",(time_d/numOfRounds));
    printf("Average CPU execution time: %f\n",(time_h/numOfRounds));
    printf("CPU/GPU time: %f\n",(time_h/time_d));

    return 0;
}



































