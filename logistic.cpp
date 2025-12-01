#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "logistic.h"

double sigmoid(double z) {
    if (z > 20) return 1.0;
    if (z < -20) return 0.0;
    return 1.0 / (1.0 + exp(-z));
}

// ===================== SHUFFLE ===================== //
void shuffle_dataset(Dataset* d) {
    for (int i = d->rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        double* tempX = d->X[i];
        d->X[i] = d->X[j];
        d->X[j] = tempX;

        double tempY = d->y[i];
        d->y[i] = d->y[j];
        d->y[j] = tempY;
    }
}

// ================= NORMALIZAÇÃO ==================== //
void normalize_dataset(Dataset* d) {
    for (int j = 0; j < d->cols; j++) {
        double mean = 0, std = 0;

        for (int i = 0; i < d->rows; i++)
            mean += d->X[i][j];
        mean /= d->rows;

        for (int i = 0; i < d->rows; i++)
            std += (d->X[i][j] - mean) * (d->X[i][j] - mean);
        std = sqrt(std / d->rows);

        if (std == 0) std = 1;

        for (int i = 0; i < d->rows; i++)
            d->X[i][j] = (d->X[i][j] - mean) / std;
    }
}

// ===================== PREVISÃO ===================== //
int predict(LogisticRegression* model, double* x) {
    double z = model->weights[0];

    for (int j = 0; j < model->n_features; j++)
        z += model->weights[j + 1] * x[j];

    return sigmoid(z) >= 0.5 ? 1 : 0;
}

// ===================== TREINAMENTO ===================== //
void train(LogisticRegression* model, Dataset* data, double lr, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {

        shuffle_dataset(data);

        for (int i = 0; i < data->rows; i++) {

            double z = model->weights[0];

            for (int j = 0; j < data->cols; j++)
                z += model->weights[j + 1] * data->X[i][j];

            double pred = sigmoid(z);
            double error = pred - data->y[i];

            model->weights[0] -= lr * error;

            for (int j = 0; j < data->cols; j++)
                model->weights[j + 1] -= lr * error * data->X[i][j];
        }

        if (epoch % 50 == 0)
            printf("Epoch %d concluído.\n", epoch);
    }
}

// ===================== CSV LOADER ===================== //
Dataset load_csv(const char* filename) {
    Dataset d;
    d.rows = 0;
    d.cols = 0;

    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Erro ao abrir %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE];

    fgets(line, sizeof(line), fp);

    int total_cols = 1;
    for (int i = 0; line[i]; i++)
        if (line[i] == ',') total_cols++;

    d.cols = total_cols - 1;

    int cap = 1024;
    d.X = (double**)malloc(sizeof(double*) * cap);
    d.y = (double*)malloc(sizeof(double) * cap);

    while (fgets(line, sizeof(line), fp)) {

        if (d.rows == cap) {
            cap *= 2;
            d.X = (double**)realloc(d.X, sizeof(double*) * cap);
            d.y = (double*)realloc(d.y, sizeof(double) * cap);
        }

        d.X[d.rows] = (double*)malloc(sizeof(double) * d.cols);

        int col = 0;
        char* cursor = line;
        char* end;

        for (int i = 0; i < total_cols; i++) {

            end = strchr(cursor, ',');
            if (!end) end = strchr(cursor, '\n');
            if (!end) end = cursor + strlen(cursor);

            char temp[512];
            int len = end - cursor;
            if (len > 511) len = 511;

            strncpy(temp, cursor, len);
            temp[len] = 0;

            double value = atof(temp);

            if (i < d.cols)
                d.X[d.rows][i] = value;
            else
                d.y[d.rows] = value;

            cursor = (*end == ',') ? end + 1 : end;
        }

        d.rows++;
    }

    fclose(fp);
    return d;
}

// ===================== MÉTRICAS COM RETORNO ===================== //
void compute_metrics(
    Dataset* data, LogisticRegression* model,
    double* err, double* acc, double* pre, double* rec, double* f1
) {
    int TP = 0, TN = 0, FP = 0, FN = 0;

    for (int i = 0; i < data->rows; i++) {
        int y_pred = predict(model, data->X[i]);
        int y_true = (int)data->y[i];

        if (y_pred == 1 && y_true == 1) TP++;
        else if (y_pred == 0 && y_true == 0) TN++;
        else if (y_pred == 1 && y_true == 0) FP++;
        else if (y_pred == 0 && y_true == 1) FN++;
    }

    *acc = (TP + TN) / (double)(TP + TN + FP + FN);
    *pre = (TP + FP) ? (TP / (double)(TP + FP)) : 0;
    *rec = (TP + FN) ? (TP / (double)(TP + FN)) : 0;
    *f1 = (*pre + *rec) ? (2 * (*pre) * (*rec)) / (*pre + *rec) : 0;
    *err = 1 - *acc;
}
