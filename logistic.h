#ifndef LOGISTIC_H
#define LOGISTIC_H

#define MAX_LINE 20000

typedef struct {
    double** X;
    double* y;
    int rows;
    int cols;
} Dataset;

typedef struct {
    double* weights;
    int n_features;
} LogisticRegression;

double sigmoid(double z);
Dataset load_csv(const char* filename);
void normalize_dataset(Dataset* data);
void shuffle_dataset(Dataset* data);

void train(LogisticRegression* model, Dataset* data, double lr, int epochs);
int predict(LogisticRegression* model, double* x);

// Métricas retornando valores
void compute_metrics(
    Dataset* data, LogisticRegression* model,
    double* err, double* acc, double* pre, double* rec, double* f1
);

#endif
