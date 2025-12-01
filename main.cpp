//Isaque gabriel hellvig rodrigues R.A: a2753103

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "logistic.h"

void save_metrics_json(
    const char* filename,
    double err_train, double acc_train, double pre_train, double rec_train, double f1_train,
    double err_test, double acc_test, double pre_test, double rec_test, double f1_test
) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Erro ao criar JSON.\n");
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"train\": {\n");
    fprintf(f, "    \"error\": %.6f,\n", err_train);
    fprintf(f, "    \"accuracy\": %.6f,\n", acc_train);
    fprintf(f, "    \"precision\": %.6f,\n", pre_train);
    fprintf(f, "    \"recall\": %.6f,\n", rec_train);
    fprintf(f, "    \"f1score\": %.6f\n", f1_train);
    fprintf(f, "  },\n");
    fprintf(f, "  \"test\": {\n");
    fprintf(f, "    \"error\": %.6f,\n", err_test);
    fprintf(f, "    \"accuracy\": %.6f,\n", acc_test);
    fprintf(f, "    \"precision\": %.6f,\n", pre_test);
    fprintf(f, "    \"recall\": %.6f,\n", rec_test);
    fprintf(f, "    \"f1score\": %.6f\n", f1_test);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
    printf("\nArquivo metrics.json gerado!\n");
}

int main() {
    srand(time(NULL));

    printf("Carregando dataset...\n");
    Dataset data = load_csv("pd_speech_features.csv");

    printf("Dataset carregado!\nLinhas: %d\nFeatures: %d\n", data.rows, data.cols);

    normalize_dataset(&data);

    int train_size = (int)(data.rows * 0.8);
    int test_size = data.rows - train_size;

    Dataset trainData, testData;

    trainData.rows = train_size;
    trainData.cols = data.cols;
    trainData.X = data.X;
    trainData.y = data.y;

    testData.rows = test_size;
    testData.cols = data.cols;
    testData.X = &data.X[train_size];
    testData.y = &data.y[train_size];

    LogisticRegression model;
    model.n_features = data.cols;
    model.weights = (double*)malloc(sizeof(double) * (model.n_features + 1));

    for (int i = 0; i <= model.n_features; i++)
        model.weights[i] = ((rand() / (double)RAND_MAX) - 0.5) * 0.01;

    train(&model, &trainData, 0.001, 600);

    double err_tr, acc_tr, pre_tr, rec_tr, f1_tr;
    double err_te, acc_te, pre_te, rec_te, f1_te;

    compute_metrics(&trainData, &model, &err_tr, &acc_tr, &pre_tr, &rec_tr, &f1_tr);
    compute_metrics(&testData, &model, &err_te, &acc_te, &pre_te, &rec_te, &f1_te);

    save_metrics_json(
        "metrics.json",
        err_tr, acc_tr, pre_tr, rec_tr, f1_tr,
        err_te, acc_te, pre_te, rec_te, f1_te
    );

    return 0;
}

