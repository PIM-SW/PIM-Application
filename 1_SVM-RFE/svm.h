#ifndef _LIBSVM_H
#define _LIBSVM_H

#ifdef __cplusplus
extern "C" {
#endif

#define _OPENMP
// #define _MKL

struct svm_node
{
	int index;
	float value;
};

struct svm_problem
{
	int l;
	float *y;
	struct svm_node **x;
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID };	/* kernel_type */

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	float degree;	/* for poly */
	float gamma;	/* for poly/rbf/sigmoid */
	float coef0;	/* for poly/sigmoid */

	/* these are for training only */
	float cache_size; /* in MB */
	float eps;	/* stopping criteria */
	float C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	float* weight;		/* for C_SVC */
	float nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	float p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
};

struct svm_model
{
	svm_parameter param;	// parameter
	int nr_class;		// number of classes, = 2 in regression/one class svm
	int l;			// total #SV
	svm_node **SV;		// SVs (SV[l])
	float **sv_coef;	// coefficients for SVs in decision functions (sv_coef[n-1][l])
	float *rho;		// constants in decision functions (rho[n*(n-1)/2])

	// for classification only

	int *label;		// label of each class (label[n])
	int *nSV;		// number of SVs for each class (nSV[n])
				// nSV[0] + nSV[1] + ... + nSV[n-1] = l
	// XXX
	int free_sv;		// 1 if svm_model is created by svm_load_model
				// 0 if svm_model is created by svm_train
};

//struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param, int *pIndex);
struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
struct svm_model *svm_train_advanced(const svm_problem *prob, const svm_parameter *param, int* _label);
int svm_save_model(const char *model_file_name, const struct svm_model *model);

struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);

void svm_predict_values(const struct svm_model *model, const struct svm_node *x, float* dec_values);
float svm_predict(const struct svm_model *model, const struct svm_node *x);

void svm_destroy_model(struct svm_model *model);
void svm_destroy_prob(struct svm_problem prob);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);

int svm_save_model(const char *model_file_name, const svm_model *model);

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
