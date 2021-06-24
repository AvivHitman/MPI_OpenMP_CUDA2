# MPI_OpenMP_CUDA
 
Parallel implementation of Sequence Alignment with MPI + CUDA + OMP

MPI
At first the Master read all parameters from the file.
Then the master sends to each process different RNA in dynamic mpi (because the number of the RNAs in the input file is not constant).
Then, each process (slave) is doing the function of calculating score, offset and k.
The master gets the result from slave and write it to the output file.

Complexity - O (num of rna/ num of slaves)


CUDA
The big advantage of Cuda is that it can handle massive amount of small tasks on parallel, In this case, it handle massive amount of letters which need to be compared.
In this exercise the max amount of input letters is 2,000, and Invidia GPU have more than 500,000 threads. Which means that Each Cuda thread can handle a single comparing loop through its dimensions.

Two usages:
1.	Get DNA and RNA, comparing correspondent letters and return array that represent the num of points, colons, stars and spaces. In that way the score of each RNA is calculating very fast.

2.	Get DNA and RNA, comparing correspondent letters and return array in size of RNA that represent the scores of each letter of the RNA that has compared to DNA. I used this array when I wanted to find the best k of each offset because that the difference between RNA with k to k+1 is only the scores of two letters. These two scores I can easily get from the arrays that CUDA gave me to DNA and RNA (without k) with offset i and i+1.

Complexity - O(K) - when k is the dimension.


OMP
I used omp in the main function that calculate all the parameters. This function is a big for loop that calls to CUDA three times and do a lot of calculations, so the function is very massive and when I do it with omp in parallel way the run time get shorter significantly.

Complexity - O([length(dna) - length(rna)] * length(rna) / num of threads)



