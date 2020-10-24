/*
 * Uloha pro cviceni 3 - CUDA - B4M39GPU (zima 2020/2021):
 *
 *  Napiste kernel, ktery otoci pole celych cisel:
 * 
 *  a) pro pripad kdy je vstupni pole i vystupni pole ulozeno v globalni pameti
 *     -> kernel reverseArrayI(int *devIn, int *devOut)
 *     pouzijte pouze jednorozmernou mrizku
 *
 *  b) to same jako a), ale nyni pouzijte dvourozmernou mrizku
 *     -> kernel reverseArrayII(int *devIn, int *devOut)
 *
 *  c) kazdy blok otoci svuj kus vstupniho pole ve sdilene pameti a vysledek zapise do globalni pameti sekvencne
 *     -> reverseArraySM(int *devIn, int *devOut)
 *
 * based on the code published in Dr Dobb's Online Journal:
 * - by Rob Farber , May 13, 2008
 * - CUDA, Supercomputing for the Masses: Part 3
 * - http://drdobbs.com/high-performance-computing/207603131
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h> 
#include <stdio.h>
#include <assert.h>
 

// Simple function to check for CUDA runtime errors.
static void handleCUDAError(
	cudaError_t error,		// error code
	const char *file,		// file within error was generated
	int line )			// line where error occurs
{
  if (error != cudaSuccess) {	// any error -> display error message and terminate application
    printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
    exit( EXIT_FAILURE );
  }
}

#define CHECK_ERROR( error ) ( handleCUDAError( error, __FILE__, __LINE__ ) )

 
// Kernel to reverse array directly in the global memory (1D grid).
__global__ void reverseArrayI(int *devIn, int *devOut) {
 // number of elements in preceding blocks in input array
 int inOffset  = blockDim.x * blockIdx.x;					
 // number of elements in preceding blocks in output array
 int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
 // element index in input array
 int inIndex  = inOffset + threadIdx.x;							
 // element index in output array
 int outIndex = outOffset + (blockDim.x - 1 - threadIdx.x);		

  devOut[outIndex] = devIn[inIndex];		// Non-coalesced write
}


// Kernel to reverse array directly in the global memory (2D grid).
__global__ void reverseArrayII(int *devIn, int *devOut) {
 // number of elements in preceding blocks in input array
 int inOffset = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y);
 // number of elements in preceding blocks in output array
 int outOffset = ((gridDim.y - 1 - blockIdx.y) * gridDim.x + (gridDim.x - 1 - blockIdx.x)) * (blockDim.x * blockDim.y);
 // element index in input array
 int inIndex = inOffset + threadIdx.y * blockDim.x + threadIdx.x;
 // element index in output array
 int outIndex = outOffset + (blockDim.y - 1 - threadIdx.y) * blockDim.x + (blockDim.x - 1 - threadIdx.x);

  devOut[outIndex] = devIn[inIndex];		// Non-coalesced write
}


// Kernel to reverse array using shared memory.
__global__ void reverseArraySM(int *devIn, int *devOut) {
 extern __shared__ int shData[];   // shared memory - its amount is given by third parameter during the kernel invocation kernelName<<<gridDim, blockDim, sharedMemPerBlock>>>()
 
 int inOffset  = blockDim.x * blockIdx.x;		// number of elements in preceding blocks in input array
 int inIndex  = inOffset + threadIdx.x;			// element index in input array
 
  // load one element per thread from device/global memory and store it 
  // in reversed order into temporary shared memory -> Coalesced Read
  shData[blockDim.x - 1 - threadIdx.x] = devIn[inIndex];
 
  // wait until all threads in the block have written their data to shared memory
  __syncthreads();
 
  // write the data from shared memory in forward order, 
  // but to the reversed block offset as before -> Coalesced Write
  int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);	// number of elements in preceding blocks in output array
  int outIndex = outOffset + threadIdx.x;						// element index in output array

  devOut[outIndex] = shData[threadIdx.x];
}



int main( int argc, char** argv) {
 // pointer for host memory and size
 int *hostArray;
 int arraySize = 256 * 1024; // 256K elements (1MB total) 
 int *devIn, *devOut;		 // pointers for device memory
 int numThreadsPerBlock = 256;	 // define grid and block size
 
 // compute number of blocks needed based on array size and desired block size
 int numBlocks = arraySize / numThreadsPerBlock;  
 
  // allocate host memory
  size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
  hostArray = (int *) malloc(memSize);

  // allocate device memory
  CHECK_ERROR( cudaMalloc( (void **) &devIn, memSize ) );
  CHECK_ERROR( cudaMalloc( (void **) &devOut, memSize ) );
 
  // initialize input array on host
  for (int i = 0; i < arraySize; i++)
    hostArray[i] = i;
 
  // copy host array to device array
  CHECK_ERROR( cudaMemcpy( devIn, hostArray, memSize, cudaMemcpyHostToDevice ) );
 
  // grid configuration
  dim3 gridRes(numBlocks, 1, 1);
  dim3 blockRes(numThreadsPerBlock, 1, 1);

  // launch kernel - reverse array in global memory
  reverseArrayI<<< gridRes, blockRes >>>( devIn, devOut );

  //dim3 gridResII(numBlocks/16, 16, 1);
  //dim3 blockResII(16, numThreadsPerBlock/16, 1);
  //reverseArrayII <<< gridResII, blockResII >> > (devIn, devOut);

  CHECK_ERROR( cudaGetLastError() );

  // block until the device has completed
  cudaDeviceSynchronize();
 
  // device to host copy
  CHECK_ERROR( cudaMemcpy( hostArray, devOut, memSize, cudaMemcpyDeviceToHost ) );
  
  // verify the data returned to the host is correct
  for (int i = 0; i < arraySize; i++)
    assert( hostArray[i] == arraySize - 1 - i );
 
 // compute number of bytes of shared memory needed per block
 int sharedMemSize = numThreadsPerBlock * sizeof(int);

  // launch kernel - reverse array using shared memory
  reverseArraySM<<< gridRes, blockRes, sharedMemSize >>>( devOut, devIn );
  
  CHECK_ERROR( cudaGetLastError() );

  // block until the device has completed
  cudaDeviceSynchronize();
  
  // device to host copy
  CHECK_ERROR( cudaMemcpy( hostArray, devIn, memSize, cudaMemcpyDeviceToHost ) );
 
  // verify the data returned to the host is correct
  for (int i = 0; i < arraySize; i++)
    assert( hostArray[i] == i );

  // free device memory
  CHECK_ERROR( cudaFree(devIn) );
  CHECK_ERROR( cudaFree(devOut) );
 
  // free host memory
  free(hostArray);
  
  return 0;
}
