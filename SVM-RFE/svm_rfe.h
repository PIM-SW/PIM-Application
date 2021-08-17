// Svm_RFE.h: interface for the Svm_RFE class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _SVM_RFE
#define _SVM_RFE

#include <stdlib.h>
#include <string.h>
#include "svm.h"


using namespace std;

typedef struct {
	float weight;
	bool marked;
} WEIGHT;

typedef struct {
	float weight;
	int ID;
} MINWEIGHT; 

typedef struct {
	float weight;
	int ID;
} WEIGHTINDEX;

typedef struct {
	float accuracy;
	int RemainSize;
	int iterator;
}  ACCURACY;

typedef struct{
	int iterator;
	int RemainSize;
	vector<WEIGHTINDEX> weightEl;
	vector<int> RemaineddCol;
}  HISTORY;

typedef struct{
	float weight;
	int index;
	float cov;
} RANKING;

class Svm_RFE  
{
public:	
	void set_step(float stepT);       //set eliminate step  default= 0.01   (eliminate 1% features); 
	void set_features(int featuresT);  // set remain features default = 20   (selected features output)

	bool compute_start(vector<vector<float> > GeneData /*Microarray dataset Matrix with each row is one sample*/, HISTORY &eliminateHistory /*the weights of those do not use!~*/);  //begin SVM-RFE procedure with default SVM Parameters
	bool compute_start(string FileName, int M/*number of user define the Microarray samples */, int N /*Number of user define the Microarray features*/, HISTORY &eliminateHistory);	 //
	bool compute_start(vector<vector<float> > GeneData, char paramT/*svm_Parameters Char*/, float ParamValue/*SVM_Parameters value*/, HISTORY &eliminateHistory);  //begin SVM-RFE procedure with  user's FUll SVM Parameters
	bool compute_start(vector<vector<float> > GeneData, struct svm_parameter paramvector/*user define Parameters*/, HISTORY &eliminateHistory);       //begin SVM-RFE procedure with user can give some  SVM Parameters  this parameters need reference SVMlib toolkit
	                                     /*SVM Parameter pls to reference SVM_LIB  toolkit*/

	Svm_RFE();                           
	virtual ~Svm_RFE();

protected:
	vector<vector<float> > data; //input Microarray dataset value; the structure of Microarray is "targerts index1: value1 index2:value2����;"
	vector<int> index;            // input Microarray dataset index
	vector<WEIGHT> weights;	      //ranking each features with SVM_weights f(x)=w*x+b;
	vector<RANKING> ranking;	  //copy each ranking features 
	vector<ACCURACY> accuracies;  // accuracies obtained by each out loop <- leave one out cross validation;
	vector<int> RemainedCol;      //the remain features after eliminate step;
	vector<WEIGHTINDEX> weightEl;
	HISTORY History;      // the remain features' history : we can trace the history to find the optimization feature subset;

	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob;		// set by read_problem
	struct svm_model *model;        //output of svmtrain -> svm_lib 
	int nr_fold;                     // n fold cross validation of svm_tain. the Parameters we do not use -> SVM_lib
	int deleted;                     //  eliminited features 
	int* label;                       
	float step;                      // eliminition step it can be an int or an 0<float <1
	int features;                    //use define the minimun feautures can output defaut:1
	int MATRIX_M;                    // Microarray Samples
    int MATRIX_N;                    // Microarray genes(features);


private:	
	bool svm_rfe_engine(struct svm_problem prob,struct svm_parameter param,HISTORY & eliminateHistory);   //do kernel_eng of feature elimination 
	
	bool mark_min_weight(float minPercent);         // when we get weights of each genes(features) we mark the minimum feaures 
	bool mark_min_weight(int step);                 // when we get weights of each genes(features) we mark the minimum feaures 

	bool save_weight(vector<RANKING>& ranking,int MaxIndex,vector<HISTORY>& History,vector<RANKING>& weightfs,vector<RANKING>& weightrm); // backup the first svmtrain weights of each genes(features);
	void save_ranking(vector<WEIGHT>& weights,vector<RANKING>& ranking);   //when we get genes(features) subset ,we ranking the genes(features); we get ranking list now : this is we out put;
	
	bool get_weight(struct svm_model* pModel);    // get the genes(features) weights  by svmtrain procedure;
    
	/*dump the data procedure ,when we debug this code.*/
	bool dump_matrix(vector<vector<float> >& array,string FileName);    
	bool dump_matrix(vector<float>& array,string FileName);
	bool dump_matrix(vector<int>& array,string FileName);
	void dump_AlphaY(struct svm_model *model, string FileName);	
	void dump_prob(struct svm_problem& prob, string FileName);
	void dump_prob_x(struct svm_problem& prob, string FileName);
	void dump_prob_y(struct svm_problem& prob, string FileName);
	bool dump_accuracy(vector<ACCURACY>& accuracies, string FileName);
	bool dump_weights(vector<WEIGHT>& weights,string FileName);
	bool dump_label(int* label, string FileName);
	void dump_slice(struct svm_node* slice, string FileName);
	void dump_accuracies(vector<ACCURACY>& accuracies, string FileName);
	void dump_history(vector<HISTORY> History, string FileName);

/*********************************************************/
	
	bool input_file_to_vector(string FileName, vector<vector<float> >& array); // read input file -> vector.
	bool input_file_to_prob(string FileName,struct svm_problem& prob);           //read input file -> prob.


	bool vector_to_prob(vector<vector<float> >& array, struct svm_problem& prob);    //transform the vector into prob.
	bool prob_to_vector(vector<vector<float> >& array,struct svm_problem& prob);     //transform the prob. into vector.
	bool init_svm_param();                                   // initialize the svm_train Parameters : the is a most acult procedure. it can change the accurate of our procedure.
	bool init_svm_param(struct svm_parameter paramT);        // initialize the svm_train Parameters by user define;
	bool init_svm_param(char paramT, float ParamValue);     // initialize the svm_ train Parameters by user define; it can see svm_train.cpp
	
    
	bool eliminate(vector<WEIGHT>& weights,struct svm_problem& prob);           // eliminiate the features with minimum abs(weight)
	bool eliminate(vector<vector<float> >& array, vector<WEIGHT>& weights);
	bool abs_weight(vector<WEIGHT>& weights);                    // obtain the abs(weights )
	
	void do_cross_validation();                                  // do cross_ validation of svmtrain-> SVM_lib.

	bool get_new_train(struct svm_problem& prob, int row, struct svm_problem& NewProb,struct svm_node* slice);   // seperate the dataset int two subset when we do leave one out cross validation .

	int partition_accuracy(vector<ACCURACY>& accuracies, int l, int r, float Key); // svm_lib 
	void quick_sort_accuracy(vector<ACCURACY>& accuracies, int i, int j);		// sort accuracy and we can find the maximum one : we need it.
	int partition_cov(vector<RANKING>& cov, int l, int r, float Key);
	void quick_sort_cov(vector<RANKING>& cov, int i, int j);
	int partition_weight(vector<WEIGHT>& weights, int l, int r, float Key);
	void quick_sort_weight(vector<WEIGHT>& weights, int i, int j);
	void get_cov(struct svm_problem& prob,vector<RANKING>& weights);

private:	
	virtual void normalize(vector<vector<float> >& array);    // normalize the data
	virtual void normalize(struct svm_problem& prob);
	virtual void message_out(string MessageOut);               
	virtual void error_message(string ErrorMessage);
};

#endif
