#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "definition.h"
#include <string.h>

__global__  void getTheScore(int *scores, char* dna, char* rna, int numElements);
__device__ void compare(int *scores, char dna, char rna);
__device__ int checkColon(char a, char b);
__device__ int checkPoint(char a, char b);
__global__  void arrayOfScores(double *s, char* dna, char* rna,double* weight, int numElements);
__device__ void compare2 (double *s, char dna, char rna, double* weight, int i);


__global__  void getTheScore(int *scores, char* dna, char* rna, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	
    // compare dna & rna in place i by thread i 
    if (i < numElements){
		compare(scores, dna[i], rna[i]);
	}
	
        
}

__global__  void arrayOfScores(double *s, char* dna, char* rna, double* weight, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	
    // compare dna & rna in place i by thread i and save each reslt in array
    if (i < numElements){
		compare2(s, dna[i], rna[i], weight, i);

	}
	
        
}
// create array in size rna 
__device__ void compare2 (double *s, char dna, char rna, double* weight, int i){

		if (dna == rna)
			s[i] = weight[0]; //num of stars
		
		else if (checkColon(dna, rna) == 1)
			s[i] = -weight[1]; // num of colons

		else if (checkPoint(dna, rna) == 1)
			s[i] = -weight[2]; // num of points

		else
			s[i] = -weight[3]; // num of spaces


	

}
// create array in size 4 that we can get from it the total score quickly
__device__ void compare (int *scores, char dna, char rna){
		for(int k=0; k<4; k++){
			scores[k] = 0;
		}

		if (dna == rna)
			atomicAdd(&scores[0],1); //num of stars
		
		else if (checkColon(dna, rna) == 1)
			atomicAdd(&scores[1],1); // num of colons

		else if (checkPoint(dna, rna) == 1)
			atomicAdd(&scores[2],1); // num of points

		else
			atomicAdd(&scores[3],1); // num of spaces


	

}
__device__ int checkColon(char a, char b) {
	int flag;
	int k;
	const char *conserativeGroup[9] = { "NDEQ", "MILV", "FYM", "NEQK", "QHRK",
			"HY", "STA", "NHQK", "MILF" };
	for (int j = 0; j < 9; j++) {
		flag = 0;
		k = 0;
		while (conserativeGroup[j][k] != '\0') {
			if ((conserativeGroup[j][k] == a)
					|| (conserativeGroup[j][k] == b)) {
				flag++;
			}
			k++;

		}
		if (flag == 2)
			return 1;


	}
	return 0;
}

__device__ int checkPoint(char a, char b) {
	int flag;
	int k;
	const char *semiConserativeGroup[11] = { "SAG", "SGND", "NEQHRK", "ATV",
			"STPA", "NDEQHK", "HFY", "CSA", "STNK", "SNDEQK", "FVLIM" };
	for (int j = 0; j < 11; j++) {
		flag = 0;
		k = 0;
		while (semiConserativeGroup[j][k] != '\0') {
			if ((semiConserativeGroup[j][k] == a)
					|| (semiConserativeGroup[j][k] == b)) {
				flag++;
			}
			k++;

		}
		if (flag == 2)
			return 1;
	}

	return 0;
}

int computeOnGPU(int *scores, char* dna, char* rna, int n) {
	
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

  
    // Allocate memory on GPU to copy the data from the host --> dna
    char *d_A;
    err = cudaMalloc((void **)&d_A, strlen(dna)*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   // Allocate memory on GPU to copy the data from the host--> rna
    char *d_B;
    err = cudaMalloc((void **)&d_B, strlen(rna)*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   // Allocate memory on GPU to copy the data from the host--> scores
    int *d_C;
    err = cudaMalloc((void **)&d_C, 4*sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to the GPU memory --> dna
    err = cudaMemcpy(d_A, dna, strlen(dna)*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	 


  // Copy data from host to the GPU memory--> rna
    err = cudaMemcpy(d_B, rna, strlen(rna)*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	

    // Launch the Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(strlen(rna) + threadsPerBlock - 1) / threadsPerBlock;
    getTheScore<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, strlen(rna));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	

    // Copy the  result from GPU to the host memory--> scores
    err = cudaMemcpy(scores, d_C, 4*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	
    // Free allocated memory on GPU
    if (cudaFree(d_A) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_B) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_C) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}




int computeOnGPU2(double *s, char* dna, char* rna, double* weight, int n) {
	
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

  
    // Allocate memory on GPU to copy the data from the host --> dna
    char *d_A;
    err = cudaMalloc((void **)&d_A, strlen(dna)*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   // Allocate memory on GPU to copy the data from the host--> rna
    char *d_B;
    err = cudaMalloc((void **)&d_B, strlen(rna)*sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   // Allocate memory on GPU to copy the data from the host--> scores
    double *d_C;
    err = cudaMalloc((void **)&d_C, strlen(rna)*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

// Allocate memory on GPU to copy the data from the host--> weight
    double *d_D;
    err = cudaMalloc((void **)&d_D, 4*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy data from host to the GPU memory--> dna
    err = cudaMemcpy(d_A, dna, strlen(dna)*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }	 


  // Copy data from host to the GPU memory-->rna
    err = cudaMemcpy(d_B, rna, strlen(rna)*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

// Copy data from host to the GPU memory--> weight
    err = cudaMemcpy(d_D, weight, 4*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	

    // Launch the Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(strlen(rna) + threadsPerBlock - 1) / threadsPerBlock;
    arrayOfScores<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, d_D, strlen(rna));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	

    // Copy the  result from GPU to the host memory --> scores
    err = cudaMemcpy(s, d_C, strlen(rna)*sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	
    // Free allocated memory on GPU
    if (cudaFree(d_A) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_B) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_C) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_D) != cudaSuccess) {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

