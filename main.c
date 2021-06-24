/*
 ============================================================================
 Name        : aviv.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello MPI World in C 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"
#include "definition.h"

int main(int argc, char* argv[]){
	int  my_rank; /* rank of process */
	int taskNumber = 0;
	int  numOfProcesses;       /* number of processes */
	int numOfRna;
	int errorNumber = -1;
	int numOfTasks;
	MPI_Status status ;   /* return status for receive */
	Result result;
	Result tempResult;

	MPI_Datatype ResultMPIType;
	MPI_Datatype type[7] = {MPI_INT, MPI_CHAR, MPI_CHAR, MPI_DOUBLE,MPI_INT,MPI_INT, MPI_DOUBLE};
	int blocklen[7] = {1, 3000, 2000, 1, 1, 1, 4};
	MPI_Aint disp[7];

	/* start up MPI */
	MPI_Init(&argc, &argv);
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);
	
    disp[0] = (char *) &result.seqNum - (char *) &result;  
    disp[1] = (char *) &result.dna -	 (char *) &result;
    disp[2] = (char *) &result.rna -	 (char *) &result;
    disp[3] = (char *) &result.score - (char *) &result;
    disp[4] = (char *) &result.offset - (char *) &result;
    disp[5] = (char *) &result.k - (char *) &result;
    disp[6] = (char *) &result.weight - (char *) &result;
    

    MPI_Type_create_struct(7, blocklen, disp, type, &ResultMPIType);
    MPI_Type_commit(&ResultMPIType);


//MASTER
    if(my_rank == 0){
    	FILE *input;
	FILE *output;
    	input = fopen("/home/linuxu/input.txt", "r");
    	if (!input)
    		printf("ERROR! can't read the file");

    	fscanf(input, "%lf %lf %lf %lf \n", &result.weight[0], &result.weight[1], &result.weight[2], &result.weight[3]);
    	fscanf(input, "%s\n", result.dna);
    	fscanf(input, "%d\n", &numOfRna);
	
	// send tasks to slaves
    	for (int slaveNumber = 1; slaveNumber < numOfProcesses; slaveNumber++) {
    		if (slaveNumber > numOfRna) {
				result.dna[0] = '\0';
    				MPI_Send(&result, 1, ResultMPIType, slaveNumber, 0 ,MPI_COMM_WORLD);
    		}
    		else{
    			fscanf(input, "%s\n", result.rna);
			result.seqNum = taskNumber + 1;	
    			MPI_Send(&result, 1, ResultMPIType, slaveNumber, 0 ,MPI_COMM_WORLD);
    			taskNumber++;
    		}
    	}
	
	// Open a file for write result
	output = fopen("/home/linuxu/output.txt", "w");
	if(!output)
		printf("ERROR! can't open the file");
	

	// more tasks (rna) than slaves	
    	while(taskNumber != numOfRna){
    		MPI_Recv(&tempResult, 1, ResultMPIType, MPI_ANY_SOURCE, MPI_ANY_TAG,MPI_COMM_WORLD, &status);
		fprintf(output, "seq number: %d, score: %lf, offset: %d, and k: %d \n", tempResult.seqNum, tempResult.score, tempResult.offset, tempResult.k);

		result.seqNum = taskNumber + 1;			
		fscanf(input, "%s\n", result.rna);
		MPI_Send(&result, 1, ResultMPIType, status.MPI_SOURCE, 0,MPI_COMM_WORLD);
    		
    		taskNumber++;
    	}


	// check if there is more processes than slaves
	int numOfTask;
	if(numOfProcesses > numOfRna + 1)
		numOfTask = numOfRna + 1;
	else
		numOfTask = numOfProcesses;

    	for (int slaveNumber = 1; slaveNumber < numOfTask ; slaveNumber++) {
    			MPI_Recv(&tempResult, 1, ResultMPIType, MPI_ANY_SOURCE, MPI_ANY_TAG,MPI_COMM_WORLD, &status);
			fprintf(output, "seq number: %d, score: %lf, offset: %d, and k: %d \n",tempResult.seqNum, tempResult.score, tempResult.offset, tempResult.k);
    			tempResult.dna[0] = '\0';
    			MPI_Send(&tempResult, 1, ResultMPIType, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
    			}



    	fclose(input);
	fclose(output);
    }
// SLAVE
    else{
    	Result resultSlave;
    	while(1){
    		MPI_Recv(&resultSlave, 1, ResultMPIType, 0, MPI_ANY_TAG,MPI_COMM_WORLD, &status);
    		if (resultSlave.dna[0] == '\0')	
    			break;
    		
    		else{
    			function(&resultSlave, resultSlave.dna, resultSlave.rna);
        		MPI_Send(&resultSlave, 1, ResultMPIType, 0, 0,MPI_COMM_WORLD);

    		}

    	}
    }
	/* shut down MPI */
	MPI_Finalize();


	return 0;
}

// This function calculate the best score + offset + k
void function(Result *result, char dna[3000], char rna[2000]) {
	char temp[2000] = "";
	double maxScore = -10000000;
	double maxScore2 = -10000000;
	double x, y, firstScore, otherScore;
	double s1[strlen(rna)], s2[strlen(rna)];
	int scores[4];

	// the length of rna and dna is equal -> no offset & k
	if(strlen(dna) == strlen(rna)){
		result->offset = 0;
		result->k = 0;
		computeOnGPU(scores, dna, rna, strlen(rna));
		result->score = getScore(scores, result->weight); 

	}
	else{
	int i, k;
	#pragma omp parallel for ordered private(i, k, firstScore, temp, x, y, otherScore, s1,s2) shared(maxScore, maxScore2) 
	for (i = 0; i < strlen(dna) - strlen(rna); i++) {
		 k = 1;
		
		//create mutant with k = 1
		strncat(temp, rna, k);
		temp[k] = '-';
		temp[k + 1] = '\0';
		strcat(temp, rna + k);
		
		
		if(computeOnGPU(scores, dna + i, temp, strlen(temp)) != 0)
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		firstScore = getScore(scores, result->weight); // get score of offset = i, k = 1

		if(computeOnGPU2(s1, dna + i, rna, result->weight, strlen(rna)) != 0) // get array of scores for offset i
			MPI_Abort(MPI_COMM_WORLD, __LINE__);

		if(computeOnGPU2(s2, dna + i + 1, rna, result->weight, strlen(rna))!= 0)// get array of scores for offset i+1
			MPI_Abort(MPI_COMM_WORLD, __LINE__);

		otherScore = firstScore;
		
		//for each offset i find the best k 	
		// the difference between k to k+1 is the scores of two chars x&y
		for (k = 1; k < strlen(rna); k++){				
			x = s1[k];
			y = s2[k]; 
			otherScore = otherScore + x - y ; // get the score of k+1
			if(otherScore > maxScore2){
				maxScore2 = otherScore;
				result->k = k + 1;
			}			
			
		}
		result->score = maxScore2;
		if (result->score > maxScore) {
			maxScore = result->score;
			result->offset = i;			

		}
		
	}
	result->score = maxScore;
	}
}

double getScore(int scores[4], double weight[4]){
	return weight[0] * scores[0] - weight[1] * scores[1] - weight[2] * scores[2]
			- weight[3] * scores[3];
}



