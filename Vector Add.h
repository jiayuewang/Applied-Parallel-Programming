// MP 1
#include <wb.h>


__global__ void vectorAddKernel(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  if (i<len) out[i] =in1[i]+in2[i]; 
}

int main(int argc, char **argv) {
  wbArg_t args;
  int n;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &n);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &n);
  hostOutput = (float *)malloc(n * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");


  wbLog(TRACE, "The input length is ", n);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int sizeBytes,errCode; 
  sizeBytes = n * sizeof(float);
  
  errCode = cudaMalloc((void **) &deviceInput1, sizeBytes); 
  if (errCode) wbLog(TRACE, "Allocating GPU memory 1 is done error:", errCode); 
  errCode = cudaMalloc((void **) &deviceInput2, sizeBytes);
  if (errCode) wbLog(TRACE, "Allocating GPU memory 2 is done error:", errCode);
  errCode = cudaMalloc((void **) &deviceOutput, sizeBytes);
  if (errCode) wbLog(TRACE, "Allocating GPU memory 3 is done error:", errCode);
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  errCode = cudaMemcpy(deviceInput1, hostInput1, sizeBytes, cudaMemcpyHostToDevice); 
  if (errCode) wbLog(TRACE, "Copying input memory 1 is done error:", errCode);
  errCode = cudaMemcpy(deviceInput2, hostInput2, sizeBytes, cudaMemcpyHostToDevice);
  if (errCode) wbLog(TRACE, "Copying input memory 2 is done error:", errCode);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  dim3 DimGrid(n/256,1,1);
  if (n%256) DimGrid.x++;
  dim3 DimBlock(256,1,1);

  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  vectorAddKernel<<<DimGrid,DimBlock>>>(deviceInput1,deviceInput2,deviceOutput, n);
  if (cudaGetLastError) wbLog(TRACE, "Performing CUDA computation is done error:", cudaGetLastError);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
 
  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  errCode = cudaMemcpy(hostOutput, deviceOutput, sizeBytes, cudaMemcpyDeviceToHost);
  if (errCode) wbLog(TRACE, "Copying output memory to the CPU is done error:", errCode);  
                  
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  

  
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, n);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
