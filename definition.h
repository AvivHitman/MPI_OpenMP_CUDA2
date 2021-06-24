/*
 * definition.h
 *
 *  Created on: Aug 14, 2020
 *      Author: linuxu
 */

#ifndef DEFINITION_H_
#define DEFINITION_H_

typedef struct{
	int seqNum;
	char dna[3000];
	char rna[2000];
	double score;
	int offset;
	int k;
	double weight[4];
}Result;

void function(Result* result, char dna[3000], char rna[2000]);
double getScore(int scores[4], double weight[4]);
int computeOnGPU(int* scores, char* dna, char* rna, int n);
int computeOnGPU2(double *s, char* dna, char* rna, double* weight, int n);



#endif /* DEFINITION_H_ */
