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
                               const int NYSCL, const int NXSCL, const int NYPAD, const int NXPAD, int Ti) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int tn = blockIdx.z;
  int c = threadIdx.x;
  for(int ky = 0; ky < Ky; ky++)
    for(int kx = 0; kx < Kx; kx++)
      for(int i = Ti*c; i < Ti*c + Ti; i++)
        neurons_n[y * NXSCL * Nn + x * Nn + tn] += neurons_i[(y+ky) * NXPAD * Ni + (x+kx)*Ni + c] * synapses[ky * Kx * Nn * Ni + kx * Nn * Ni + tn * Ni + i];
  __syncthreads();
  neurons_n[y * NXSCL * Nn + x * Nn + tn] = neurons_n[y * NXSCL * Nn + x * Nn + tn]? neurons_n[y * NXSCL * Nn + x * Nn + tn] : neurons_n[y * NXSCL * Nn + x * Nn + tn]/4;
}

int main(const int argc, const char** argv) {
  //Setting parameter values
  const int Kx = 3, Ky = 3;
  const int Nn = 512, Ni = 512;
  const int Nx = 14, Ny = 14;
  //const int Sx = 1, Sy = 1;
  //int Tnn = 32, Tn = 16, Ti = 16, Ty = 8, Tx = 8;
  const int NYPAD = Ny + Ky - 1, NXPAD = Nx + Kx - 1;
  const int NYSCL = Ny, NXSCL = Nx;

  cudaError_t err = cudaSuccess;
  //cout << sizeof(VTYPE);
  cout << "Allocating space for host and device matrix/vectors\n";

  //Allocating host memory for input layer
  VTYPE *h_neurons_i = (VTYPE *)malloc(NYPAD * NXPAD * Ni * sizeof(VTYPE));
  //Allocating host memory for synapses
  VTYPE *h_synapses = (VTYPE *)malloc(Ky * Kx * Nn * Ni * sizeof(VTYPE));
  //Allocating host memory for output layer
  VTYPE *h_neurons_n = (VTYPE *)malloc(NYSCL * NXSCL * Nn * sizeof(VTYPE));
  
  //Verify that host allocations succeeded
  if(h_neurons_i == NULL || h_synapses == NULL || h_neurons_n == NULL) {
    fprintf(stderr, "Failed to allocate host vectors\n");
    exit(EXIT_FAILURE);
  }
  
  //Allocate device memory for input layer
  VTYPE *d_neurons_i = NULL;
  err = cudaMalloc((void **)&d_neurons_i, NYPAD * NXPAD * Ni * sizeof(VTYPE));
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device input layer (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  //Allocate device memory for synapses
  VTYPE *d_synapses = NULL;
  err = cudaMalloc((void **)&d_synapses, Ky * Kx * Nn * Ni * sizeof(VTYPE));
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device input layer (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  //Allocate device memory for output
  VTYPE *d_neurons_n = NULL;
  err = cudaMalloc((void **)&d_neurons_n, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device input layer (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  cout << "initializing arrays\n";
  
  //Initializing host input layer
  for(int y = 0; y < NYPAD; y++)
    for(int x = 0; x < NXPAD; x++)
      for(int i = 0; i < Ni; i++)
        h_neurons_i[y * NXPAD * Ni + x * Ni + i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  //Initializing host synapses
  for(int ky = 0; ky < Ky; ky++)
    for(int kx = 0; kx < Kx; kx++)
      for(int n = 0; n < Nn; n++)
        for(int i = 0; i < Ni; i++)
          h_synapses[ky * Kx * Nn * Ni + kx * Nn * Ni + n * Ni + i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  //Initializing host output layer
  for(int y = 0; y < NYSCL; y++)
    for(int x = 0; x < NXSCL; x++)
      for(int n = 0; n < Nn; n++)
        h_neurons_n[y * NYSCL * Nn + x * Nn + n] = static_cast <float> (0);

  //Copy host input layer + synapses + output to device
  err = cudaMemcpy(d_neurons_i, h_neurons_i, NYPAD * NXPAD * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy input layer from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  } 

  err = cudaMemcpy(d_synapses, h_synapses, Ky * Kx * Nn * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy input layer from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  err = cudaMemcpy(d_neurons_n, h_neurons_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    fprintf(stderr, "Failed to copy input layer from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  } 
  cout << "starting computation\n";

  //Simple Version
  begin_roi();

  // — Original code — (excluding nn, ii loops)
  dim3 dimBlock(Ni);  //Max = 1024
  dim3 dimGrid(Nx,Ny,Nn);
  int Ti = Ni/dimBlock.x;
  convolution<<<dimGrid,dimBlock>>>(d_neurons_i,d_synapses,d_neurons_n,Ny,Nx,Ni,Nn,Ky,Kx,NYSCL,NXSCL,NYPAD,NXPAD,Ti);
  cudaDeviceSynchronize();
  end_roi();

  err = cudaMemcpy(h_neurons_n, d_neurons_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
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


