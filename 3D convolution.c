#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define TILE_WIDTH 8
#define INPUT_TILE_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)

// Serial convolution code - for test purpose



//@@ Define constant memory for device kernel here
__constant__ float MASK[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size, const int y_size, const int x_size) {
  //@@ Insert kernel code here
    
  __shared__ float sharedMemoryTile [INPUT_TILE_WIDTH][INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];
  
  int bx = blockIdx.x;  int by = blockIdx.y; int bz = blockIdx.z;
  int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
  
  // Identify the address of the input tile element
  int z_in = bz * TILE_WIDTH + tz - MASK_RADIUS;
  int y_in = by * TILE_WIDTH + ty - MASK_RADIUS;
  int x_in = bx * TILE_WIDTH + tx - MASK_RADIUS;
  int x_out = bx * TILE_WIDTH + tx;
  int y_out = by * TILE_WIDTH + ty;
  int z_out = bz * TILE_WIDTH + tz;
      
  
  // Repeat with number of tiles
  
  
 
    // give 0 values to elements beyond the matrix range
    if (z_in >= 0 && z_in < z_size &&
                y_in >= 0 && y_in < y_size &&
                x_in >= 0 && x_in < x_size)
            
      // copy input to shared memory
      sharedMemoryTile[tz][ty][tx] = input[y_size*x_size*z_in + x_size*y_in + x_in];
    else sharedMemoryTile[tz][ty][tx] = 0;
    
    
     __syncthreads();
     float res = 0.0;
    // do convolution for every element of the output tile
   
   
  
    //sharedMemoryTile[tz+1][ty+1][tx+1];
    // find output index for resulted sum
    if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH){
      for (int x = 0; x < MASK_WIDTH; x++)
       for (int y = 0; y < MASK_WIDTH; y++)
        for (int z = 0; z < MASK_WIDTH; z++)
         res += sharedMemoryTile[tz + z][ty + y][tx + x] * MASK[z][y][x];
      
      
      
      if (y_out < y_size && x_out < x_size && z_out < z_size)
        output[x_size * y_size * z_out + x_size * y_out + x_out] = res;
    }
    __syncthreads();
  
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;
  //float *deviceMask;
    
  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbCheck(cudaMalloc((void **) &deviceInput, z_size * y_size * x_size * sizeof(float))); // allocate the value in the gpu memory and print an error code
  wbCheck(cudaMalloc((void **) &deviceOutput, z_size * y_size * x_size * sizeof(float))); // allocate the value in the gpu memory and print an error code
  //wbCheck(cudaMalloc((void **) &deviceMask, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float))); // allocate the value in the gpu memory and print an error code
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpyToSymbol(MASK,hostKernel, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float));
  wbCheck(cudaMemcpy(deviceInput, &hostInput[3], z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice)); // copy the variables into gpu memory
 // wbCheck(cudaMemcpy(deviceMask, hostKernel, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float),cudaMemcpyHostToDevice));
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((float)x_size/TILE_WIDTH), ceil((float)y_size/TILE_WIDTH),ceil((float)z_size/TILE_WIDTH));
  dim3 dimBlock(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH, INPUT_TILE_WIDTH);
    
  //@@ Launch the GPU kernel here
  //serialConvolution(hostInput+3, hostOutput+3, hostKernel, z_size, y_size, x_size);
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(cudaMemcpy(&hostOutput[3], deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  wbLog(TRACE, "The result is: ", hostInput[0], "x", hostInput[1], "x", hostInput[2], "x", hostInput[3]);
  //wbLog(TRACE, "The result is: ", hostKernel[0], "x", hostKernel[1], "x", hostKernel[2], "x", hostKernel[3]);
  wbLog(TRACE, "The result is: ", hostOutput[4], "x", hostOutput[5], "x", hostOutput[6], "x", hostOutput[7]);
  
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
