#include <iostream>
#include "dnn.hpp"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>

using namespace std; 

__global__ void classifier_layer(const VTYPE *neurons_i, const VTYPE *synapses, VTYPE *neurons_n, const int Ni, const int Nn) {
  __shared__ VTYPE psum[1024];
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i == 31)
    for(int n=0; n < Nn; n++)
      psum[n] = 0;
  __syncthreads();
  for(int n=0; n < Nn; n++)
    psum[n] += neurons_i[i] * synapses[Ni * n + i];
  __syncthreads();
  for(int n=0; n < Nn; n++)
    neurons_n[n] = psum[n]>0? psum[n]:psum[n]/4;
}


int main(int argc, char** argv) {
  int Nn = 1024, Ni = 4096;
  //int Tn = 32, Ti = 32;
  cudaError_t err = cudaSuccess;

  cout << "Allocating space for host and device matrix/vectors\n";

  //Allocate host memory for input layer
  VTYPE *h_neurons_i = (VTYPE *)malloc(Ni * sizeof(VTYPE));

  //Allocate host memory for synapses
  VTYPE *h_synapses = (VTYPE *)malloc(Nn * Ni * sizeof(VTYPE));

  //Allocate host memory for output layer
  VTYPE *h_neurons_n = (VTYPE *)malloc(Nn * sizeof(VTYPE));


  // Verify that allocations succeeded
  if (h_neurons_i == NULL || h_synapses == NULL || h_neurons_n == NULL)
  {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
  }

  for(int i = 0; i < Ni; i++)
    h_neurons_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  for(int n = 0; n < Nn ; n++) 
    for(int i = 0; i < Ni ; i++)
      h_synapses[n * Ni + i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;

  //Allocate device memory for input layer
  VTYPE *d_neurons_i = NULL;
  err = cudaMalloc((void **)&d_neurons_i , Ni * sizeof(VTYPE));
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device input layer (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  //Allocate device memory for synapses layer
  VTYPE *d_synapses = NULL;
  err = cudaMalloc((void **)&d_synapses , Nn * Ni * sizeof(VTYPE));
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device synapes (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  //Allocate device memory for output layer
  VTYPE *d_neurons_n = NULL;
  err = cudaMalloc((void **)&d_neurons_n , Nn * sizeof(VTYPE));
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device output layer (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  //Copy host input layer + synapses to device
  err = cudaMemcpy(d_neurons_i, h_neurons_i, Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy input layer from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  } 

  err = cudaMemcpy(d_synapses, h_synapses, Nn * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy input layer from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cout << "starting computation\n";

  // Launch the Vector Add CUDA Kernel
  begin_roi();
  classifier_layer<<<4096/32,32>>>(d_neurons_i,d_synapses,d_neurons_n,Ni,Nn);
  cudaDeviceSynchronize();
  end_roi();

  //Copy device output layer to host
  err = cudaMemcpy(h_neurons_n, d_neurons_n, Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy output layer from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

 /* 
  for(int n = 0; n < Nn ; n++)
    cout << h_neurons_n[n] << "\n";
*/  
  cout << "simple version complete!\n";  

  //Deallocating memory on host and device

  free(h_neurons_i);
  free(h_synapses);
  free(h_neurons_n);

  cudaFree(d_neurons_i);
  cudaFree(d_synapses);
  cudaFree(d_neurons_n);

  //compare(h_neurons_n,h_neurons_n2,Nn);

  cout << "done\n";
}

