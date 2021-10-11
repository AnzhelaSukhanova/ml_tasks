import sys
import math
from tabulate import tabulate
from random import randrange
from copy import copy
from collections import defaultdict
from statistics import mean, variance

import pandas as pd

feature_num = 0
fold_num = 5
epoch_num = 80


def normalize(train_data):
    min_row = train_data.min()
    max_row = train_data.max()

    norm_data = pd.DataFrame(columns=range(feature_num))
    for index, row in train_data.iterrows():
        norm_data = norm_data.append((row - min_row)/(max_row - min_row),
                                     ignore_index=True)
    norm_data.iloc[:, -1] = train_data.iloc[:, -1]
    return norm_data


def predict(row, weights):
    pred_res = weights[0]
    for i in range(1, len(row) + 1):
        if math.isnan(row[i - 1]):
            row.iat[i - 1] = 0
        pred_res += weights[i] * row[i - 1]
    return pred_res


def sgd(folds, ground_truth, test_ind):
    weights = [0] * (feature_num + 1)
    rows_num = folds[0].shape[0]

    for i in range(fold_num):
        if i != test_ind:
            train_data = folds[i]
            gt = ground_truth[i]
            for j in range(1, epoch_num):
                error_sum = 0
                step = 1 / j
                grad = defaultdict(int)

                for k, row in train_data.iterrows():
                    pred = predict(row, weights)
                    error = pred - gt[k]
                    error_sum += error ** 2
                    grad[0] += error
                    for l in range(1, feature_num):
                        grad[l] += error * row[l - 1]

                for k in range(feature_num):
                    weights[k] = weights[k] - step * (2 / rows_num) * grad[k]

                print('epoch=%d, step=%.3f, mse=%.3f' % (j, step, error_sum / rows_num))

    return weights


def get_folds(train_data):
    folds = []
    ground_truth = []
    data_copy = copy(train_data)
    fold_size = int(len(train_data) / fold_num)

    for _ in range(fold_num):
        fold = pd.DataFrame(index=range(fold_size), columns=range(feature_num + 1))
        for i in range(fold_size):
            index = randrange(len(data_copy))
            row = data_copy.iloc[index]
            data_copy.drop(index)
            fold.iloc[i] = row

        ground_truth.append(fold.iloc[:, -1])
        folds.append(fold.iloc[:, :-1])
    return folds, ground_truth


def get_stats(fold, ground_truth, weights):
    error_sum = 0
    dev_sum = 0
    gt_mean = ground_truth.mean()
    predictions = []

    for k, row in fold.iterrows():
        pred = predict(row, weights)
        error = pred - ground_truth[k]
        error_sum += error ** 2
        dev_sum += (ground_truth[k] - gt_mean) ** 2
        predictions.append(pred)

    n = fold.shape[0]
    R2 = 1 - (error_sum / dev_sum)
    rmse = math.sqrt(error_sum / n)
    pred_mean = mean(predictions)
    D = variance(predictions)
    return R2, rmse, pred_mean, D


def main(argv):
    global feature_num

    variants_num = 2 if len(argv) < 2 else argv[1]

    for k in range(1, variants_num):
        print('\nVariant: ' + str(k))
        train_data = pd.read_csv('Dataset/Training/Features_Variant_' +
                                 str(k) + '.csv', header=None)

        feature_num = train_data.shape[1] - 1
        train_data = normalize(train_data)
        print('Normalization: done')
        folds, ground_truth = get_folds(train_data)
        print('Cross validation: done')

        stats = pd.DataFrame(index=range(5), columns=['F1', 'F2', 'F3', 'F4', 'F5', 'weights'])
        for i in range(fold_num):
            weights = sgd(folds, ground_truth, i)
            w_str = str(weights[0])
            for j in range(1, len(weights)):
                w_str += '\n' + str(weights[j])
            stats.iloc[i, 5] = w_str
            for j in range(fold_num):
                fold_stats = 'Train:\n' if j != i else 'Test:\n'
                R2, rmse, mean, D = get_stats(folds[j], ground_truth[j], weights)
                fold_stats += 'R^2: ' + str(R2) + '\n'
                fold_stats += 'RMSE: ' + str(rmse) + '\n'
                fold_stats += 'Mean: ' + str(mean) + '\n'
                fold_stats += 'Variance: ' + str(D)
                stats.iloc[i, j] = fold_stats
        print(tabulate(stats, headers='keys', tablefmt='grid'))


if __name__ == '__main__':
    main(sys.argv)
