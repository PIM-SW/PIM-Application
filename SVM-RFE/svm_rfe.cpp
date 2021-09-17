// Svm_RFE.cpp: implementation of the Svm_RFE class.
//
//////////////////////////////////////////////////////////////////////

#include <vector>
#include <fstream>
#include <math.h>
#include <string>
#include <iostream>

#include "svm_rfe.h"

#ifndef WIN32
#include "wtime.h"
#else
#include <time.h>
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

extern int nIteration;
extern int nGeneLength;

#ifdef _MKL
extern float **ppMatrix;
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Svm_RFE::Svm_RFE()
{
	MATRIX_M = 62;
	MATRIX_N = 2000;
	deleted = 0;	
	features = 30;
	step = 0.1f;

}

Svm_RFE::~Svm_RFE()
{
	svm_destroy_prob(prob);
}

bool Svm_RFE::input_file_to_vector(string FileName, vector<vector<float> >& array)
{	
	float tmp;
	int iTmp,i,j;
	bool bHead = true;
	ifstream fp(FileName.c_str());
	if(!fp)
    {
        error_message("can't open colon file!\n");
		return false;
    }

	data.resize(MATRIX_M);
	for(i=0 ;i<MATRIX_M; i++)
	{
		data[i].resize(MATRIX_N);
	}
	for(i=0; i<MATRIX_M; i++)
	{
		for(j=0; j<MATRIX_N+1; j++)
		{
			fp >> tmp;
			if(bHead)
			{
				iTmp = (int)tmp;
				index.push_back(iTmp);
				bHead = false;
			}
			else
			{
				data[i][j-1] = tmp;
			}
			if(fp.peek() == '\n')
			{
				bHead = true;
			}
		}
	}
	fp.close();
	return true;	
}

void Svm_RFE::normalize(vector<vector<float> >& array)
{
	int M,N;
	int i,j;
	float tmp;

	M = array.size();
	if(M == 0)
	{
		error_message("Null Matrix.\n");
		return;
	}
	N = array[0].size();
	if(N == 0)
	{
		error_message("Null Matrix.\n");
		return;
	}

    for (i = 0 ; i < M; i++)
    {
        tmp = 0.0f;
        for (j = 0; j < N; j++)
        {
             tmp = tmp + array[i][j]*array[i][j];
        }
        for (j = 0; j < N; j++)
        {
            array[i][j] = array[i][j]/sqrt(tmp);		
        }		
     }
		
}

bool Svm_RFE::dump_matrix(vector<vector<float> >& array,string FileName)
{
	ofstream fp(FileName.c_str());
	if(!fp)
    {
        error_message("can't open dump files\n");
		return false;
    }

	int M,N;
	int i,j;
	M = array.size();
	float tmp;
	if(M == 0)
	{
		error_message("Null Matrix.\n");
		return false;
	}
	N = array[0].size();
	if(N == 0)
	{
		error_message("Null Matrix.\n");
		return false;
	}
	
    for (i = 0 ; i < M; i++)
    {     
        for (j = 0; j < N; j++)
        {
			tmp = array[i][j];
			fp << i << ":" << j << " " << array[i][j]<<"  ";
        }
		fp << '\n';
	}
	fp.close();
	
	return true;
}

bool Svm_RFE::compute_start(string FileName, int M, int N,HISTORY & eliminateHistory)
{
	bool bRet;
	init_svm_param();

	MATRIX_M = M;
	MATRIX_N = N;
	clock_t start, end;
#ifndef WIN32
	timer_start(20);
	if(!input_file_to_prob(FileName,prob))
		return false;
	timer_stop(20);
	printf ("Reading data set time: %.2fs\n", timer_read(20));
	timer_start(21);
	bRet = svm_rfe_engine(prob, param, eliminateHistory);
	timer_stop(21);
	printf ("Data prepare + SVM Training time: %.2fs\n", timer_read(21));
#else
	start = clock();
	if(!input_file_to_prob(FileName,prob))
		return false;
	end = clock();
	printf ("Reading data set time: %.2fs\n", (float)(end-start)/CLOCKS_PER_SEC);
	bRet = svm_rfe_engine(prob, param, eliminateHistory);
	start = clock();
	printf ("Data prepare + SVM Training time: %.2fs\n", (float)(start-end)/CLOCKS_PER_SEC);
#endif	
	return bRet;
}

bool Svm_RFE::get_weight(struct svm_model* pModel)
{
	if(!model)
	{
		error_message("Null model.\n");
		return false;
	}
	vector<vector<vector<float> > > w_list;
	vector<int> nr_sv;
	
	int i,j;
	int datanum = pModel->l;
	int nr_class = pModel->nr_class;
	int dim = MATRIX_N - deleted;
	
	//reserve memory
	w_list.resize(nr_class);
	for(i = 0; i < nr_class; ++i)
	{
		w_list[i].resize(nr_class);
		for(j = 0; j < nr_class-1; ++j)
		{
			w_list[i][j].resize(dim);
		}
	}
	nr_sv.resize(nr_class);

	weights.resize(dim);
	
	//Initialization
	for(i=0;i<nr_class;i++)
		nr_sv[i] = pModel->nSV[i];


	//Compute start
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for(i = 0; i < dim; ++i)
	{
		for(int j = 0; j < nr_class - 1; ++j)
		{
			int index = 0;
			int end = 0;                 
			for(int k = 0; k < nr_class; ++k)
			{
				float acc = 0.0;
				index += (k == 0) ? 0 : nr_sv[k - 1];
				end = index + nr_sv[k]; 
				for(int m = index; m < end; ++m)
					acc += pModel->sv_coef[j][m] * (pModel->SV[m]+i)->value; //zhh 
				
				w_list[k][j][i] = acc;
			}
		}
	}


    for( i = 0; i < nr_class - 1; ++i)
	{
        for(int j = i + 1, k = i; j < nr_class; ++j, ++k)
		{
            for(int m = 0; m < dim; ++m)
			{
                 weights[m].weight = w_list[i][k][m] + w_list[j][i][m];				 
            }                       
        }
    }
	return true;	
}

bool Svm_RFE::mark_min_weight(float minPercent)
{
	if( (minPercent>1.0f) || (minPercent<0.0f))
	{
		error_message("Bad minPercent. \n");
		return false;
	}
	if(minPercent == 0.0f)
	{
		return true;
	}
	if(weights.size() == 0)
	{
		error_message("Null weights\n");
		return false;
	}

	int i;
	MINWEIGHT MinWeight;
	int MinCounter;
	MinCounter= int(weights.size()*minPercent);
	if(MinCounter == 0)
	{
		MinCounter = 1;
	}
	deleted += MinCounter;
	
	MinWeight.weight = weights[0].weight;
	MinWeight.ID = 0;

	for(i=0; i<weights.size(); i++)
	{
		weights[i].marked = false;
	}
	
	while(MinCounter > 0)
	{
		for(i=0; i<weights.size(); i++)
		{
			if( (weights[i].marked == false) && (weights[i].weight < MinWeight.weight) )
			{
				MinWeight.weight = weights[i].weight;
				MinWeight.ID = i;
			}
		}

		weights[MinWeight.ID].marked = true;

		i = 0;
		while(weights[i].marked) i++;
		MinWeight.weight = weights[i].weight;
		MinWeight.ID = i;
		
		MinCounter--;
	}
	return true;
}

bool Svm_RFE::eliminate(vector<vector<float> >& array, vector<WEIGHT>& weights)
{
	if(array.size() == 0)
	{
		error_message("Null data. \n");
		return false;
	}	

	if(weights.size() == 0)
	{
		error_message("Null weights. \n") ;
		return false;
	}

	if(weights.size() != array.size())
	{
		error_message("Data and weights sizes don't match!\n");
		return false;
	}

	int i,j;
	WEIGHTINDEX temp;
	weightEl.clear();

	for(i = weights.size()-1; i>=0; i--)
	{
		if(weights[i].marked == true)
		{
			for(j=0; j<MATRIX_M; j++)
			{				
				data[j].erase(data[j].begin() + i);;
			}
			temp.ID = RemainedCol[i];
			temp.weight = weights[i].weight;
			weightEl.push_back(temp);
			RemainedCol.erase(RemainedCol.begin() + i);
		}
	}
	return true;

}

void Svm_RFE::error_message(string error_message)
{
	if(error_message.length()!=0)
	{
		cout << "Error Message:\n";
		cout << error_message.c_str();
	}
}

void Svm_RFE::message_out(string MessageOut)
{
	if(MessageOut.length()!=0)
	{
		cout << MessageOut.c_str();
	}
}

bool Svm_RFE::init_svm_param()
{
	param.svm_type = NU_SVC;
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = 0.1;
	param.cache_size = 100;
	param.C = 100000;
	param.eps = 5e-3;
	param.p = 0.1;
	// param.shrinking = 1;
	param.shrinking = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	return true;
}


void Svm_RFE::dump_prob_x(struct svm_problem& prob, string FileName)
{
	FILE *fpdump = fopen(FileName.c_str(),"w");
	int i,j;
	for(i=0; i< prob.l; i++)
	{		
		for(j=0; j<MATRIX_N - deleted; j++)
		{			
			//fprintf(fpdump,"prob.x[%d][%d]index=%d value=%f ",i,j,prob.x[i][j].index,prob.x[i][j].value);			
			fprintf(fpdump,"%f  ",prob.x[i][j].value);			
		}
		fprintf(fpdump,"\n");
	}
	fclose(fpdump);
}

void Svm_RFE::do_cross_validation()
{
	int i;
	int total_correct = 0;
	float total_error = 0;
	float sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

	// random shuffle
	for(i=0;i<prob.l;i++)
	{
		int j = i+rand()%(prob.l-i);
		struct svm_node *tx;
		float ty;
			
		tx = prob.x[i];
		prob.x[i] = prob.x[j];
		prob.x[j] = tx;

		ty = prob.y[i];
		prob.y[i] = prob.y[j];
		prob.y[j] = ty;
	}

	for(i=0;i<nr_fold;i++)
	{
		int begin = i*prob.l/nr_fold;
		int end = (i+1)*prob.l/nr_fold;
		int j,k;
		struct svm_problem subprob;

		subprob.l = prob.l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(float,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob.x[j];
			subprob.y[k] = prob.y[j];
			++k;
		}
		for(j=end;j<prob.l;j++)
		{
			subprob.x[k] = prob.x[j];
			subprob.y[k] = prob.y[j];
			++k;
		}

		if(param.svm_type == EPSILON_SVR ||
		   param.svm_type == NU_SVR)
		{
			struct svm_model *submodel = svm_train(&subprob,&param);
			float error = 0;
			for(j=begin;j<end;j++)
			{
				float v = svm_predict(submodel,prob.x[j]);
				float y = prob.y[j];
				error += (v-y)*(v-y);
				sumv += v;
				sumy += y;
				sumvv += v*v;
				sumyy += y*y;
				sumvy += v*y;
			}
			svm_destroy_model(submodel);
			printf("Mean squared error = %g\n", error/(end-begin));
			total_error += error;			
		}
		else
		{
			struct svm_model *submodel = svm_train(&subprob,&param);
			int correct = 0;
			for(j=begin;j<end;j++)
			{
				float v = svm_predict(submodel,prob.x[j]);
				if(v == prob.y[j])
					++correct;
			}
			svm_destroy_model(submodel);
			printf("Accuracy = %g%% (%d/%d)\n", 100.0*correct/(end-begin),correct,(end-begin));
			total_correct += correct;
		}

		free(subprob.x);
		free(subprob.y);
	}		
	if(param.svm_type == EPSILON_SVR || param.svm_type == NU_SVR)
	{
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
}

bool Svm_RFE::input_file_to_prob(string FileName, struct svm_problem& prob)
{
	float tmp;
	int iTmp,i,j;
	bool bHead = true;
	ifstream fp(FileName.c_str());
	if(!fp)
    {
        error_message("can't open colon file!\n");
		return false;
    }	
	prob.l = MATRIX_M;
	prob.y = Malloc(float,prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);

	if( !prob.x || !prob.y )
	{		
		error_message("Out of memory!\n");
		return false;
	}
	
	for(i=0; i<MATRIX_M; i++)
	{
		prob.x[i] = Malloc(struct svm_node, MATRIX_N+1);
		for(j=0; j<MATRIX_N+1; j++)
		{		
			fp >> tmp;
			if(bHead)
			{
				iTmp = (int)tmp;
				prob.y[i] = iTmp;
				bHead = false;
			}
			else
			{
				prob.x[i][j - 1].index = j;
				prob.x[i][j - 1].value = tmp;				
			}
		}
		bHead = true;
		prob.x[i][j - 1].index = -1;
	}
	return true;
}	

void Svm_RFE::normalize(struct svm_problem& prob)
{
	int i,j;
	float tmp, tmp1, mean, std;	
	if(prob.x == NULL)
	{
		error_message("Null Prob\n");
		return;
	}

/*	for (i = 0 ; i < MATRIX_M; i++)
    {
        tmp = 0.0f;
        for (j = 0; j < MATRIX_N; j++)
        {
             tmp = tmp + prob.x[i][j].value*prob.x[i][j].value;
        }
        for (j = 0; j < MATRIX_N; j++)
        {
            prob.x[i][j].value = prob.x[i][j].value/sqrt(tmp);		
        }		
     }	
*/
	//standardlize
	for (i = 0 ; i < MATRIX_M; i++)
    {
        tmp = 0.0f;
		tmp1 = 0.0f;
        for (j = 0; j < MATRIX_N; j++)
        {
             tmp = tmp + prob.x[i][j].value;
        }
		mean = tmp/MATRIX_N;
        for (j = 0; j < MATRIX_N; j++)
        {
            tmp1 = tmp1+(prob.x[i][j].value-mean)*(prob.x[i][j].value-mean);		
        }
		std = sqrt(tmp1/(MATRIX_N-1));
        for (j = 0; j < MATRIX_N; j++)
        {
            prob.x[i][j].value = (prob.x[i][j].value-mean)/std;		
        }		
    }
	
	for (j = 0 ; j < MATRIX_N; j++)
    {
        tmp = 0.0f;
		tmp1 = 0.0f;
        for (i = 0; i < MATRIX_M; i++)
        {
             tmp = tmp + prob.x[i][j].value;
        }
		mean = tmp/MATRIX_M;
        for (i = 0; i < MATRIX_M; i++)
        {
            tmp1 = tmp1+(prob.x[i][j].value-mean)*(prob.x[i][j].value-mean);		
        }
		std = sqrt(tmp1/(MATRIX_M-1));
        for (i = 0; i < MATRIX_M; i++)
        {
            prob.x[i][j].value = (prob.x[i][j].value-mean)/std;		
        }		
    }	

	for (j = 0 ; j < MATRIX_N; j++)
    {
        for (i = 0; i < MATRIX_M; i++)
        {
             prob.x[i][j].value=1*atan(prob.x[i][j].value/1);
        }
	}
}


bool Svm_RFE::eliminate(vector<WEIGHT>& weights,struct svm_problem& prob)
{
	if(weights.size() == 0)
	{
		error_message("Null weights. \n") ;
		return false;
	}
	WEIGHTINDEX temp;
	weightEl.clear();
	int i;
	int m,n;
	int count = 0;
	
	for(i = weights.size()-1; i>=0; i--)
	{
		if(weights[i].marked == true)
		{
			printf("Deleted(%d)\n", prob.x[0][i].index);
			for(m=0; m<MATRIX_M; m++)
			{
#ifndef _OPT
				memcpy (&(prob.x[m][i]), &(prob.x[m][i+1]), sizeof(struct svm_node)*(weights.size()-i));
#else
				memcpy (&(prob.x[m][i]), &(prob.x[m][nGeneLength-1]), sizeof(struct svm_node));
#endif
			}
			temp.ID = RemainedCol[i];
			temp.weight = weights[i].weight;
			weightEl.push_back(temp);
			RemainedCol.erase(RemainedCol.begin() + i);
		}
	}
	return true;
}

bool Svm_RFE::abs_weight(vector<WEIGHT>& weights)
{
	if(weights.size() == 0)
	{
		error_message("Null weights.\n");
		return false;
	}
	for(int i=0; i<weights.size(); i++)
		weights[i].weight = fabs(weights[i].weight);
	return true;
}

bool Svm_RFE::vector_to_prob(vector<vector<float> >& array,struct svm_problem& prob)
{
	int i,j;		
	
	if(prob.y)
		free(prob.y);
	if(prob.x)
		free(prob.x);	

	prob.l = array.size();	
	prob.y = Malloc(float,prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);

	if( !prob.x || !prob.y )
	{		
		error_message("Out of memory!\n");
		return false;
	}

	if(array.size() != MATRIX_M)
	{
		error_message("Matrix size doesn't match!\n");
		return false;
	}

	for(i=0; i<MATRIX_M; i++)
	{
		prob.y[i] = array[i][0];		
		prob.x[i] = Malloc(struct svm_node, MATRIX_N+1);
		if(!prob.x[i])
		{
			error_message("Out of memory!\n");
			return false;
		}
		for(j=0; j<MATRIX_N; j++)
		{
			prob.x[i][j].index = j + 1;
			prob.x[i][j].value = array[i+1][j];
		}
		prob.x[i][j].index = -1;
	}
	return true;
}

bool Svm_RFE::prob_to_vector(vector<vector<float> >& array,struct svm_problem& prob)
{
	array.resize(prob.l);
	message_out("Not done!\n");
	return true;
}

bool Svm_RFE::get_new_train(struct svm_problem& prob, int row , struct svm_problem& NewProb, struct svm_node* slice)
{
	int i,j;			

	if(!prob.y || !prob.x)
	{
		message_out("Null prob.\n");
		return false;
	}

	NewProb.l = prob.l - 1;

	
	for(i=0; i<prob.l ; i++)
	{
		if(i < row)
		{
			NewProb.y [i] = prob.y[i];
		
			for(j=0; j<MATRIX_N; j++)
			{
				NewProb.x[i][j].index = prob.x[i][j].index;
				NewProb.x[i][j].value = prob.x[i][j].value;
			}			
		}

		if(i > row)
		{
			NewProb.y[i - 1] = prob.y[i];			
	
			for(j=0; j<MATRIX_N; j++)
			{
				NewProb.x[i-1][j].index = prob.x[i][j].index;
				NewProb.x[i-1][j].value = prob.x[i][j].value;		
			}
		}

		if(i == row)
		{		
			for(j=0; j<MATRIX_N; j++)
			{
				slice[j].index = prob.x[i][j].index;
				slice[j].value = prob.x[i][j].value;
			}			
		}
	}

	
	for(i=0; i<prob.l-1 ; i++)
	{	
		for(j=0; j<MATRIX_N-deleted; j++)
		{
			NewProb.x[i][j].index = j+1;
		}
		NewProb.x[i][j].index = -1;
		NewProb.x[i][j].value = -1.0f;
	}

	for(i=0; i<MATRIX_N-deleted; i++)
	{
		slice[i].index = i + 1;
	}
	slice[i].index = -1;
	slice[i].value = -1.0f;
	return true;

}

int Svm_RFE::partition_accuracy(vector<ACCURACY>& accuracies, int l, int r, float Key)
{
	ACCURACY Temp;
	do
	{
		while((accuracies[++l].accuracy > Key) && l<r);
		while( r && (accuracies[--r].accuracy < Key) );
		Temp = accuracies[l];
		accuracies[l] = accuracies[r];
		accuracies[r] = Temp;
	}while(l<r);
	Temp = accuracies[l];
	accuracies[l] = accuracies[r];
	accuracies[r] = Temp;
	return l;
}

void Svm_RFE::quick_sort_accuracy(vector<ACCURACY>& accuracies, int i, int j)
{
	ACCURACY Key = accuracies[j];
	int k = partition_accuracy(accuracies,i-1,j, Key.accuracy);
	accuracies[j] = accuracies[k];
	accuracies[k] = Key;
//	dump_accuracy(accuracies, "tempaccura.txt");

	if((k-i)>1) quick_sort_accuracy(accuracies, i, k-1);
	if((j-k)>1) quick_sort_accuracy(accuracies, k+1, j);
}

bool Svm_RFE::dump_matrix(vector<int>& array,string FileName)
{
	ofstream fp(FileName.c_str());
	if(!fp)
    {
        error_message("can't open dump files\n");
		return false;
    }

	int M;
	int i;
	M = array.size();	
	if(M == 0)
	{
		error_message("Null Matrix.\n");
		return false;
	}	
	
    for (i = 0 ; i < M; i++)
    {        
		fp << i << ":" <<" " << array[i]<<"\n";        
	}	
	fp.close();
	return true;
}

bool Svm_RFE::dump_accuracy(vector<ACCURACY>& accuracies, string FileName)
{
	ofstream fp(FileName.c_str());
	if(!fp)
    {
        error_message("can't open dump files\n");
		return false;
    }

	int M;
	int i;
	M = accuracies.size();	
	if(M == 0)
	{
		error_message("Null Matrix.\n");
		return false;
	}	
	
    for (i = 0 ; i < M; i++)
    {        
		fp << i << ":" << " " << accuracies[i].accuracy<<" "<< accuracies[i].iterator << " " << accuracies[i].RemainSize << "\n";        
	}	
	fp.close();
	return true;
}

bool Svm_RFE::dump_weights(vector<WEIGHT>& weights,string FileName)
{
	ofstream fp(FileName.c_str());
	if(!fp)
    {
        error_message("can't open dump files\n");
		return false;
    }

	int M;
	int i;
	M = weights.size();	
	if(M == 0)
	{
		error_message("Null Matrix.\n");
		return false;
	}	
	
    for (i = 0 ; i < M; i++)
    {        
		fp << i << ":" << " " << weights[i].weight <<" "<< weights[i].marked << "\n";        
	}	
	fp.close();
	return true;
}

bool Svm_RFE::dump_label(int* label, string FileName)
{
	ofstream fp(FileName.c_str());
	if(!fp)
    {
        error_message("can't open dump files\n");
		return false;
    }
	if(!label)
	{
	    error_message("Null label.\n");
		return false;    
	}

	int M;
	int i;
	M = prob.l;	
	
    for (i = 0 ; i < M; i++)
    {        
		fp << i << ":" << " " << label[i] << "\n";        
	}	
	fp.close();
	return true;
}

void Svm_RFE::dump_prob_y(struct svm_problem& prob, string FileName)
{
	FILE *fpdump = fopen(FileName.c_str(),"w");
	int i;
	for(i=0; i< prob.l; i++)
	{
		fprintf(fpdump,"y[%d]=%f\n",i,prob.y[i]);		
	}
	fclose(fpdump);
}

void Svm_RFE::dump_prob(struct svm_problem& prob, string FileName)
{
	FILE *fpdump = fopen(FileName.c_str(),"w");
	int i,j;
	for(i=0; i< prob.l; i++)
	{		
		fprintf(fpdump,"%d ",int(prob.y[i]));	
		for(j=0; j<MATRIX_N - deleted ; j++)
		{			
			fprintf(fpdump,"%d:%f ",prob.x[i][j].index,prob.x[i][j].value);			
			//fprintf(fpdump,"%f  ",prob.x[i][j].value);			
		}
		fprintf(fpdump,"\n");
	}
	fclose(fpdump);
}

void Svm_RFE::dump_slice(struct svm_node* slice, string FileName)
{
	FILE *fpdump = fopen(FileName.c_str(),"w");
	int j;	
	for(j=0; j<MATRIX_N - deleted + 1; j++)
	{			
	//	fprintf(fpdump,"%f ",slice[j].value);			
		fprintf(fpdump,"%d ",slice[j].index);			
		//fprintf(fpdump,"%f  ",prob.x[i][j].value);			
	}
	
	fclose(fpdump);
}

void Svm_RFE::dump_AlphaY(struct svm_model *model, string FileName)
{
	FILE *fpdump = fopen(FileName.c_str(),"w");
	int i,j;
	for(i=0; i< model->l ; i++)
	{		
		for(j=0; j<model->nr_class -1 ; j++)
		{			
			fprintf(fpdump,"%f\n",model->sv_coef[j][i]);			
			//fprintf(fpdump,"%f  ",prob.x[i][j].value);			
		}		
	}
	fclose(fpdump);
}

void Svm_RFE::dump_accuracies(vector<ACCURACY>& accuracies, string FileName)
{
	ofstream fp(FileName.c_str());
	int j;	
	for(j=0; j<accuracies.size(); j++)
	{		
		fp << accuracies[j].iterator << " "  << accuracies[j].accuracy << " " << accuracies[j].RemainSize << "\n";
	}
	fp.close();
}

void Svm_RFE::dump_history(vector<HISTORY> History, string FileName)
{
	ofstream fp(FileName.c_str());
	int i,j;	
	for(i=0; i<History.size(); i++)
	{		
		fp << History[i].iterator<< ":" <<History[i].RemainSize << "\n" ;
		for(j=0; j<History[i].RemaineddCol.size(); j++)
		{
			fp << History[i].RemaineddCol[j] << " ";
		}
		fp << "\n\n";
		for (j=0;j<History[i].weightEl.size();j++)
		{
			fp << History[i].weightEl[j].ID << "\t" << History[i].weightEl[j].weight <<"\n";
		}
		fp << "\n\n\n\n";
	}
	fp.close();
}

//bool Svm_RFE::compute_start(vector<vector<float> > GeneData, vector<RANKING>& weightfs,vector<RANKING>& weightrm)
bool Svm_RFE::compute_start(vector<vector<float> > GeneData, HISTORY & eliminateHistory)
{	
	MATRIX_M = GeneData.size();
	if(MATRIX_M == 0)
	{
		error_message("NULL GeneData.\n");
		return false;
	}
	MATRIX_N = GeneData[0].size();
	init_svm_param();

	vector_to_prob(GeneData,prob);
	
	return svm_rfe_engine(prob, param,  eliminateHistory);	
}

//bool Svm_RFE::compute_start(vector<vector<float> > GeneData, struct svm_parameter paramvector, vector<RANKING>& weightfs,vector<RANKING>& weightrm)
//bool Svm_RFE::compute_start(string FileName, int M, int N,vector<WEIGHT>& weights)
bool Svm_RFE::compute_start(vector<vector<float> > GeneData, struct svm_parameter paramvector, HISTORY & eliminateHistory)
{	
	MATRIX_M = GeneData.size();
	if(MATRIX_M == 0)
	{
		error_message("NULL GeneData.\n");
		return false;
	}
	MATRIX_N = GeneData[0].size();
	vector_to_prob(GeneData,prob);
	init_svm_param(param);

	return svm_rfe_engine(prob, param,  eliminateHistory);	
}

bool Svm_RFE::init_svm_param(struct svm_parameter paramT)
{
	param = paramT;
	return true;
}

//bool Svm_RFE::compute_start(vector<vector<float> > GeneData, char paramT, float ParamValue,vector<RANKING>& weightfs,vector<RANKING>& weightrm)
//bool Svm_RFE::compute_start(string FileName, int M, int N,vector<WEIGHT>& weights)
bool Svm_RFE::compute_start(vector<vector<float> > GeneData, char paramT, float ParamValue,HISTORY & eliminateHistory)
{	
	MATRIX_M = GeneData.size();
	if(MATRIX_M == 0)
	{
		error_message("NULL GeneData.\n");
		return false;
	}
	MATRIX_N = GeneData[0].size();
	vector_to_prob(GeneData,prob);
	init_svm_param(paramT, ParamValue);
	
	return svm_rfe_engine(prob, param,  eliminateHistory);	
}

bool Svm_RFE::init_svm_param(char paramT, float ParamValue)
{
	switch(paramT)
	{
		case 's':
			param.svm_type = int(ParamValue);
			break;
		case 't':
			param.kernel_type = int(ParamValue);
			break;
		case 'd':
			param.degree = int(ParamValue);
			break;
		case 'g':
			param.gamma = ParamValue;
			break;
		case 'r':
			param.coef0 = ParamValue;
			break;
		case 'n':
			param.nu = ParamValue;
			break;
		case 'm':
			param.cache_size = ParamValue;
			break;
		case 'c':
			param.C = ParamValue;
			break;
		case 'e':
			param.eps = ParamValue;
			break;
		case 'p':
			param.p = ParamValue;
			break;
		case 'h':
			param.shrinking = int(ParamValue);
			break;
		default:	
			;
	}
	return true;
}

void Svm_RFE::set_features(int featuresT)
{
	features = featuresT;
}

void Svm_RFE::set_step(float stepT)
{
	step = stepT;
}

bool Svm_RFE::mark_min_weight(int step)
{
	if( step<0 )
	{
		error_message("Bad minPercent. \n");
		return false;
	}
	if( step == 0)
	{
		step = 1;
	}
	if(weights.size() == 0)
	{
		error_message("Null weights\n");
		return false;
	}

	int i;
	MINWEIGHT MinWeight;
	int MinCounter;
	MinCounter= step;
	deleted += MinCounter;
	
	MinWeight.weight = weights[0].weight;
	MinWeight.ID = 0;

	for(i=0; i<weights.size(); i++)
	{
		weights[i].marked = false;
	}
	
	while(MinCounter > 0)
	{
		for(i=0; i<weights.size(); i++)
		{
			if( (weights[i].marked == false) && (weights[i].weight < MinWeight.weight) )
			{
				MinWeight.weight = weights[i].weight;
				MinWeight.ID = i;
			}
		}

		weights[MinWeight.ID].marked = true;

		i = 0;
		while(weights[i].marked) i++;
		MinWeight.weight = weights[i].weight;
		MinWeight.ID = i;
		
		MinCounter--;
	}
	return true;
}


bool Svm_RFE::save_weight(vector<RANKING>& ranking,int MaxIndex,vector<HISTORY>& History,vector<RANKING>& weightfs,vector<RANKING>& weightrm)
{
	if(ranking.size() == 0)
	{
		error_message("Null ranking");
		return false;
	}
	int i;
	weightfs.resize(History[MaxIndex].RemaineddCol.size());
	for(i=0; i<History[MaxIndex].RemaineddCol.size(); i++)
	{
		weightfs[i].index = History[MaxIndex].RemaineddCol[i];
		weightfs[i].weight = ranking[weightfs[i].index].weight;		
	}
	for(i=weightfs.size()-1; i>=0; i--)
	{
		ranking.erase(ranking.begin() + weightfs[i].index);
	}
	
	for(i=0; i<ranking.size(); i++)
	{
		weightrm.push_back(ranking[i]);
	}
	return true;
}

void  Svm_RFE::save_ranking(vector<WEIGHT>& weights,vector<RANKING>& ranking)
{
	ranking.resize(weights.size());

	for(int i=0; i<weights.size(); i++)
	{
		ranking[i].weight = weights[i].weight;
		ranking[i].index = i;
	}
}

bool Svm_RFE::svm_rfe_engine(struct svm_problem prob,struct svm_parameter param,HISTORY & eliminateHistory)
{
	const char *error_msg;
	int cross_validation = 0;
	float v=0;
	
	HISTORY present;
	int iterator = 0;
	int i;

	error_msg = svm_check_parameter(&prob,&param);  // check the parameters
	printf("okay after check\n");
	if(error_msg)
	{
		error_message(error_msg);
		return false;
	}		

	RemainedCol.resize(MATRIX_N);
	for(i=0; i<MATRIX_N; i++)
	{
		RemainedCol[i] = i;
	}
	
	normalize(prob); 
   
	 // normalize the data.  if  we use the normalized data we should not do the procedure 
	 // if we do not have normalzie the data we should normalzie  the data.
	 // something maybe happend if we do not do normalize.	
	
	float  start, end, total;

   // int nIteration;	// Test use, eric, 5/14/2004
#ifndef WIN32
	timer_start(0);
#else
	clock_t start_t, end_t;
	start_t = clock();
#endif

	model = svm_train(&prob,&param);	 // do svmtrain -> get model of svmtrain 
	get_weight(model);                   // get weights of each (genes)features
	abs_weight(weights);                 // abs(weights)


#ifndef WIN32
	timer_stop(0);
	total = timer_read(0);
#else
	end_t = clock();
	total = (float)(end_t - start_t)/CLOCKS_PER_SEC;
#endif
	

    if(nIteration == 0)
	{
		while(weights.size() > features)// if the remain genes(features) is less than the feature remained we go out loops 
		{
		
			int temp= weights.size();		
#ifndef WIN32
			printf ("Remains(%d), time(%.2fs), ", temp, timer_read(0));
#else		
			printf ("Remains(%d) time(%2.1fs), ", temp, (float)(end_t - start_t)/CLOCKS_PER_SEC);
#endif

#ifndef WIN32
			timer_clear(0);
			timer_start(0);
#else 
			start_t = 0;
			end_t = 0;
			start_t = clock();
#endif
			abs_weight(weights);
			if(0.0f<step && step<1.0f)
			{			
				mark_min_weight(step);            // mark genes(features) we need to eliminated
			}
			if(step >= 1.0f)
			{			
				mark_min_weight(int(step));
			}
			if(step < 0.0f)
			{
				error_message("Bad step.\n");
				return false;
			}
			eliminate(weights,prob);		 // eliminate the features
			svm_destroy_model(model);         // free model
			eliminateHistory.RemaineddCol.clear();
	
	

			for(i=0; i<RemainedCol.size(); i++)
			{
				eliminateHistory.RemaineddCol.push_back(RemainedCol[i]);
			}
			for(i=0; i<weightEl.size(); i++)
			{
				eliminateHistory.weightEl.push_back(weightEl[i]);
			}
			eliminateHistory.RemainSize = present.RemaineddCol.size();

			eliminateHistory.iterator = iterator;

   			model = svm_train(&prob, &param);  // get new train on new Microarray dataset.
			nGeneLength--;

			get_weight(model);
#ifndef WIN32
			timer_stop(0);
			total += timer_read(0);
#else 
			end_t = clock();
			total += (float)(end_t-start_t)/CLOCKS_PER_SEC;
#endif
		}
	}	
	else
	{
		while (nIteration --)	// Test use, eric, 5/14/2004
		{	
			int temp= weights.size();		
#ifndef WIN32
			printf ("Remains(%d), time(%.2fs), ", temp, timer_read(0));
#else		
			printf ("Remains(%d) time(%2.1fs), ", temp, (float)(end_t - start_t)/CLOCKS_PER_SEC);
#endif

#ifndef WIN32
			timer_clear(0);
			timer_start(0);
#else 
			start_t = 0;
			end_t = 0;
			start_t = clock();
#endif
			abs_weight(weights);
			 if(0.0f<step && step<1.0f)
			{			
				mark_min_weight(step);            // mark genes(features) we need to eliminated
			}
			if(step >= 1.0f)
			{			
				mark_min_weight(int(step));
			}
			if(step < 0.0f)
			{
				error_message("Bad step.\n");
				return false;
			}
			eliminate(weights,prob);		 // eliminate the features
			svm_destroy_model(model);         // free model
			eliminateHistory.RemaineddCol.clear();

			for(i=0; i<RemainedCol.size(); i++)
			{
				eliminateHistory.RemaineddCol.push_back(RemainedCol[i]);
			}
			for(i=0; i<weightEl.size(); i++)
			{
				eliminateHistory.weightEl.push_back(weightEl[i]);
			}
			eliminateHistory.RemainSize = present.RemaineddCol.size();

			eliminateHistory.iterator = iterator;

   			model = svm_train(&prob, &param);  // get new train on new Microarray dataset.
			nGeneLength--;

			get_weight(model);
#ifndef WIN32
			timer_stop(0);
			total += timer_read(0);
#else 
			end_t = clock();
			total += (float)(end_t-start_t)/CLOCKS_PER_SEC;
#endif
		}
	}	
	
	cout << "SVM Training time: " << total << "s" << endl;

	return true;
}

bool Svm_RFE::dump_matrix(vector<float>& array,string FileName)
{
	ofstream fp(FileName.c_str());
	if(!fp)
    {
        error_message("can't open dump files\n");
		return false;
    }

	int M;
	int i;
	M = array.size();	
	if(M == 0)
	{
		error_message("Null Matrix.\n");
		return false;
	}	
	
    for (i = 0 ; i < M; i++)
    {        
		fp << i << ":" <<" " << array[i]<<"\n";        
	}	
	fp.close();
	return true;

}

void Svm_RFE::quick_sort_cov(vector<RANKING>& cov, int i, int j)
{
    RANKING Key = cov[j];
	int k = partition_cov(cov,i-1,j, Key.cov);
	cov[j] = cov[k];
	cov[k] = Key;
//	dump_accuracy(accuracies, "tempaccura.txt");

	if((k-i)>1) quick_sort_cov(cov, i, k-1);
	if((j-k)>1) quick_sort_cov(cov, k+1, j);
}

int Svm_RFE::partition_cov(vector<RANKING>& cov, int l, int r, float Key)
{
	RANKING Temp;
	do
	{
		while((cov[++l].cov > Key) && l<r);
		while( r && (cov[--r].cov < Key) );
		Temp = cov[l];
		cov[l] = cov[r];
		cov[r] = Temp;
	}while(l<r);
	Temp = cov[l];
	cov[l] = cov[r];
	cov[r] = Temp;
	return l;
}

void Svm_RFE::get_cov(struct svm_problem& prob,vector<RANKING>& weights)
{
	float x,y;
	float yy,xx,xy;

	x = 0.0f;
	y = 0.0f;
	yy = 0.0f;
	xy = 0.0f;
	xx = 0.0f;

	int i,j;

	for(i=0; i<MATRIX_M; i++)
	{
		y += prob.y[i];
	}
	y = y/MATRIX_M;	

	for(i=0; i<MATRIX_M; i++)
	{
		yy += (prob.y[i]-y)*(prob.y[i]-y);
	}

	for(i=0; i<weights.size(); i++)
	{
		for(j=0; j<MATRIX_M; j++)
		{
			x += prob.x[j][weights[i].index].value;
		}
		x = x/MATRIX_M;
		for(j=0; j<MATRIX_M; j++)
		{
			xy += (prob.x[j][weights[i].index].value-x)*(prob.y[j]-y);
			xx += (prob.x[j][weights[i].index].value-x)*(prob.x[j][weights[i].index].value-x);
		}
		weights[i].cov = xy/(xx*yy);
	}
}

void Svm_RFE::quick_sort_weight(vector<WEIGHT>& weights, int i, int j)
{
    WEIGHT Key = weights[j];
	int k = partition_weight(weights,i-1,j, Key.weight);
	weights[j] = weights[k];
	weights[k] = Key;
	if((k-i)>1) quick_sort_weight(weights, i, k-1);
	if((j-k)>1) quick_sort_weight(weights, k+1, j);
}

int Svm_RFE::partition_weight(vector<WEIGHT>& weights, int l, int r, float Key)
{
	WEIGHT Temp;
	do
	{
		while((weights[++l].weight > Key) && l<r);
		while( r && (weights[--r].weight < Key) );
		Temp = weights[l];
		weights[l] = weights[r];
		weights[r] = Temp;
	}while(l<r);
	Temp = weights[l];
	weights[l] = weights[r];
	weights[r] = Temp;
	return l;
}
