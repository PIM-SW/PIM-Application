// SVM_RFE_TEST.cpp : Defines the entry point for the console application.
//

#include <vector>
#include <iostream>
#include <fstream>
#include "time.h"

#include "svm_rfe.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#ifdef _VTUNE
#include "VtuneApi.h"
#endif

string dataset;
int GENES;
int CASES;

int nIteration;
int nGeneLength;

#ifdef _MKL
float **ppMatrix;
#endif

//const int GENES=15154;
//const int CASES=253;
 
//const int GENES=2000;
//const int CASES=62;

#define ALIGN_BYTES		128

char *alignMalloc( int len  )
{
	char *oriP = (char *)calloc(len + ALIGN_BYTES, sizeof (char));
	char *alignP = oriP + ALIGN_BYTES - ((long)oriP & (ALIGN_BYTES-1)); 
	((void **)alignP)[-1] = oriP;
	return alignP;
}

void alignFree(void * p)
{
	free(((void **)p)[-1]);
}

int cmdline_parse(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	int i, j;	
	clock_t start0, end0;
	float sum0;	

#ifdef _VTUNE
	VTPause();
#endif
	
	if(cmdline_parse(argc, argv)<0)
		exit(1);

	HISTORY eliminateHistory;
	
	ofstream fprank("./data/ranking.txt");
	ifstream fpname("./data/names.txt");
	
	string* GeneID;
	GeneID = new string[GENES];
	char GeneIDT[100] ;	
	
	nGeneLength = GENES;

	for(i=0; i<GENES; i++)
	{
		fpname >> GeneIDT;
		GeneID[i] = GeneIDT;
	}
	fpname.close();
	Svm_RFE* pSVM_RFE = new Svm_RFE();
	pSVM_RFE->set_step(0.01);    // set the elimination step. 
	pSVM_RFE->set_features(20);   // set the minimum gene(feature) subset.
	
#ifdef _VTUNE
	VTResume();
#endif

#ifdef _MKL
	ppMatrix = (float **) malloc (sizeof(float) * CASES);

	for (i = 0; i < CASES; i++)
		ppMatrix[i] = (float *) alignMalloc (sizeof(float) * GENES);
#endif

	pSVM_RFE->compute_start(dataset, CASES, GENES, eliminateHistory);//start to compute.
	
#ifdef _VTUNE
	VTPause();
#endif

#ifdef _MKL
	for (i = 0; i < CASES; i++)
		alignFree (ppMatrix[i]);
	free (ppMatrix);
#endif
	for(j=0;j<eliminateHistory.weightEl.size();j++)
	{		
		fprank << eliminateHistory.weightEl[j].ID << " \t" << eliminateHistory.weightEl[j].weight << "\t"<<GeneID[eliminateHistory.weightEl[j].ID].c_str() << "\n"; //output
	}
	fprank.close();

	return 0;
}



int cmdline_parse(int c, char* v[])
{
	switch(c)
	{
	case 5:
		dataset = v[1];
		CASES = atoi(v[2]);
		GENES = atoi(v[3]);
		nIteration = atoi(v[4]);
		break;
	case 4:
		dataset = v[1];
		CASES = atoi(v[2]);
		GENES = atoi(v[3]);
		nIteration = 0;
		break;
	default:
		cout << "Usage format:" << endl;
		cout << "svm_rfe_(versions) dataset_name CASES GENES nIteration" << endl;

		return -1;
	}
	return 1;
}
