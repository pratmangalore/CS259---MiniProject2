#include <iostream>
#include <string>
#include "dnn.hpp"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>

using namespace std;

__global__ void convolution(const VTYPE *neurons_i, const VTYPE *synapses, VTYPE *neurons_n, const int Ny,
                               const int Nx, const int Ni, const int Nn, const int Ky, const int Kx,
                               const int NYSCL, const int NXSCL, const int NYPAD, const int NXPAD) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int tn = blockIdx.z;
  int c = threadIdx.x;
  
  for(int ky = 0; ky < Ky; ky++)
    for(int kx = 0; kx < Kx; kx++)
      neurons_n[y * NXSCL * Nn + x * Nn + tn] += neurons_i[(y+ky) * NXPAD * Ni + (x+kx)*Ni + c] * synapses[ky * Kx * Nn * Ni + kx * Nn * Ni + tn * Ni + c]; 
  __syncthreads();
  neurons_n[y * NXSCL * Nn + x * Nn + tn] = neurons_n[y * NXSCL * Nn + x * Nn + tn]? neurons_n[y * NXSCL * Nn + x * Nn + tn] : neurons_n[y * NXSCL * Nn + x * Nn + tn]/4;
}

int main(const int argc, const char** argv) {
  //Setting parameter values
  const int Kx = 3, Ky = 3;
  const int Nn = 128, Ni = 128;
  const int Nx = 224, Ny = 224;
  //const int Sx = 1, Sy = 1;
  //int Tnn = 32, Tn = 16, Ti = 16, Ty = 8, Tx = 8;
  const int NYPAD = 226, NXPAD = 226;
  const int NYSCL = 224, NXSCL = 224;

  cudaError_t err = cudaSuccess;

  cout << "Allocating space for host and device matrix/vectors\n";

  //Allocating host memory for input layer
  VTYPE *h_neurons_i = (VTYPE *)malloc(Ni * NYPAD * NXPAD * sizeof(VTYPE));
  //Allocating host memory for synapses
  VTYPE *h_synapses = (VTYPE *)malloc(Nn * Ni * Ky * Kx * sizeof(VTYPE));
  //Allocating host memory for output layer
  VTYPE *h_neurons_n = (VTYPE *)malloc(Nn * NYSCL * NXSCL * sizeof(VTYPE));
  
  //Verify that host allocations succeeded
  if(h_neurons_i == NULL || h_synapses == NULL || h_neurons_n == NULL) {
    fprintf(stderr, "Failed to allocate host vectors\n");
    exit(EXIT_FAILURE);
  }
  
  //Allocate device memory for input layer
  VTYPE *d_neurons_i = NULL;
  err = cudaMalloc((void **)&d_neurons_i, Ni * NYPAD * NXPAD * sizeof(VTYPE));
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device input layer (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  //Allocate device memory for synapses
  VTYPE *d_synapses = NULL;
  err = cudaMalloc((void **)&d_synapses, Nn * Ni * Ky * Kx * sizeof(VTYPE));
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device input layer (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  //Allocate device memory for output
  VTYPE *d_neurons_n = NULL;
  err = cudaMalloc((void **)&d_neurons_n, Nn * NYSCL * NXSCL * sizeof(VTYPE));
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device input layer (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  cout << "initializing arrays\n";
  
  //Initializing host input layer
  for(int i = 0; i < Ni; i++)
    for(int y = 0; y < NYPAD; y++)
      for(int x = 0; x < NXPAD; x++)
        h_neurons_i[i * NYPAD * NXPAD + y * NXPAD + x] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  //Initializing host synapses
  for(int n = 0; n < Nn; n++)
    for(int i = 0; i < Ni; i++)
      for(int ky = 0; ky < Ky; ky++)
        for(int kx = 0; kx < Kx; kx++)
          h_synapses[n * Ni * Ky * Kx + i * Ky * Kx + ky * Kx + kx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  //Initializing host output layer
  for(int n = 0; n < Nn; n++)
    for(int y = 0; y < NYSCL; y++)
      for(int x = 0; x < NXSCL; x++)
        h_neurons_n[n * NYSCL * NXSCL + y * NXSCL + x] = static_cast <float> (0);

  //Copy host input layer + synapses + output to device
  err = cudaMemcpy(d_neurons_i, h_neurons_i, Ni * NYPAD * NXPAD * sizeof(VTYPE), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy input layer from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  } 

  err = cudaMemcpy(d_synapses, h_synapses, Nn * Ni * Ky * Kx * sizeof(VTYPE), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy input layer from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  err = cudaMemcpy(d_neurons_n, h_neurons_n, Nn * NYSCL * NXSCL * sizeof(VTYPE), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy input layer from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  } 
  cout << "starting computation\n";

  //Simple Version
  begin_roi();

  // — Original code — (excluding nn, ii loops)
  dim3 dimBlock(Ni);
  dim3 dimGrid(Nx,Ny,Nn);
  convolution<<<dimGrid,dimBlock>>>(d_neurons_i,d_synapses,d_neurons_n,Ny,Nx,Ni,Nn,Ky,Kx,NYSCL,NXSCL,NYPAD,NXPAD);
  cudaDeviceSynchronize();
  end_roi();

  err = cudaMemcpy(h_neurons_n, d_neurons_n, Nn * NYSCL * NXSCL * sizeof(VTYPE), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy input layer from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  printf("%f\n",h_neurons_n[13400]);
  cout << "simple version complete!\n";  

  //Deallocating device and host memory
  cudaFree(d_neurons_i);
  cudaFree(d_synapses);
  cudaFree(d_neurons_n);
  free(h_neurons_i);
  free(h_synapses);
  free(h_neurons_n);
}


