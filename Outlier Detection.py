import pandas as pd
from sklearn import metrics
from scipy import interpolate
import numpy as np
import os
import csv
import collections
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.loda import LODA
from pyod.models.lscp import LSCP
from sklearn.datasets import make_classification
from pyod.models.combination import aom, moa, average, maximization
from pyod.utils.utility import precision_n_scores
import time
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    data = pd.read_csv('data/wine/meta_data/meta_wine.csv')
    ar = data['anomaly.rate']
    ar_level = data['anomaly.rate.level']

    set(ar_level)

    data_dic = collections.defaultdict(list)
    for item in data.iterrows():
        data_dic[item[1]['anomaly.rate.level']].append(item[1]['bench.id'])
    for item in data_dic:
        print(len(data_dic[item]))

    ar_mean = [np.mean(ar[:300]), np.mean(ar[300:480]), np.mean(ar[480:670]),
               np.mean(ar[670:850]), np.mean(ar[850:1030]), np.mean(ar[1030:1210])]


    data = pd.read_csv('data/wine/benchmarks/wine_benchmark_0001.csv')


    def evaluation(y, y_scores, method):
        '''
        评估函数，y为groundtruth，y_scores为预测值，返回PR曲线，ROC曲线和AUC
        '''
        if isinstance(y_scores, dict):
            colors = ['r', 'g', 'b', '#FF1493', '#483D8B']
            plt.figure(figsize=(7,7))
            i = 0
            for algo in y_scores:
                pre_scr = y_scores[algo]
                print(algo, pre_scr.shape)
                fpr_level = np.arange(0, 1, 0.01)
                fpr, tpr, threshold = metrics.roc_curve(y, pre_scr)
                interp = interpolate.interp1d(fpr, tpr)
                tpr_at_fpr = [interp(x) for x in fpr_level]
                roc_auc = metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[i], label='%s ROC(area = %0.2f)' % (algo, roc_auc))
                i += 1
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('Models Compare' + '-ROC')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.legend(loc="lower right")
        else:
            fpr_level = np.arange(0, 1, 0.01)
            fpr, tpr, threshold = metrics.roc_curve(y, y_scores)
            interp = interpolate.interp1d(fpr, tpr)
            tpr_at_fpr = [interp(x) for x in fpr_level]
            roc_auc = metrics.auc(fpr, tpr)
            precision, recall, _ = metrics.precision_recall_curve(y, y_scores)
            pr_auc = round(precision_n_scores(y, y_scores), ndigits=4)
    #         pr_auc_t = metrics.auc(recall, precision)
            plt.figure(figsize=(12,5))
            plt.subplot(1, 2, 1)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(method + '-ROC')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.plot(fpr, tpr, color='r', label='ROC curve (area = %0.2f)' % roc_auc)
            plt.legend(loc="lower right")
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, marker='.', label='precision @ rank n: %0.2f)' % pr_auc)
            plt.legend(loc="upper right")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(method + '-PR')
            plt.show()




    path = 'data/wine/benchmarks/'
    samples = []
    for item in data_dic:
        samples.append(data_dic[item][0])
    y = []
    x_train = []
    contam = 0
    for sample in samples:
        p = os.path.join(path, sample+'.csv')
        data = pd.read_csv(p)
        data = data.dropna()
        for i in data.iterrows():
            # 0为正常，1为异常点
            if i[1][5] == 'anomaly':
                y.append(1)
                contam += 1
            else:
                y.append(0)
            x_train.append(list(i[1][6:17]))
    x_train = np.array(x_train)
    y = np.array(y)
    contam /= len(y)



    algorithms = ['KNN', 'LOF', 'PCA', 'LODA']
    all_scores = {}

    clf_name = 'KNN'
    clf = KNN(n_neighbors=5, contamination=contam)
    x_train = standardizer(x_train)
    clf.fit(x_train)
    knn_y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    knn_y_scores = clf.decision_scores_  # raw outlier scores
    evaluation(y, knn_y_scores, clf_name)
    all_scores['KNN'] = knn_y_scores


    clf_name = 'LOF'
    clf = LOF(contamination=contam)
    x_train = standardizer(x_train)
    clf.fit(x_train)
    y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_scores = clf.decision_scores_  # raw outlier scores
    evaluation(y, y_scores, clf_name)
    all_scores['LOF'] = y_scores


    clf_name = 'PCA'
    clf = PCA(contamination=contam)
    x_train = standardizer(x_train)
    clf.fit(x_train)
    y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_scores = clf.decision_scores_  # raw outlier scores
    evaluation(y, y_scores, clf_name)
    all_scores['PCA'] = y_scores



    clf_name = 'LODA'
    clf = LODA(contamination=contam)
    x_train = standardizer(x_train)
    clf.fit(x_train)
    y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_scores = clf.decision_scores_  # raw outlier scores
    evaluation(y, y_scores, clf_name)
    all_scores['LODA'] = y_scores


    pca = PCA(n_components=2)
    kpca = KernelPCA(n_components=2, kernel="poly")
    x_train_pca = kpca.fit_transform(x_train)
    clf = KNN(n_neighbors=5, contamination=contam)
    x_train_pca = standardizer(x_train_pca)
    clf.fit(x_train_pca)
    y_pred_pca = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_scores = clf.decision_scores_  # raw outlier scores
    evaluation(y, y_scores, 'PCA+KNN')
    all_scores['PCA+KNN'] = y_scores

    evaluation(y, all_scores, algorithms)


    visualize(clf_name, x_train_pca, y, x_train_pca, y, y_pred_pca, y_pred_pca,
             show_figure=True, save_figure=False)




    visual_ano = [[], []]
    visual_nor = [[], []]
    for i in range(0, len(x_train_pca), 10):
        if y[i] == 1:
            visual_ano[0].append(x_train_pca[i][0])
            visual_ano[1].append(x_train_pca[i][1])
        else:
            visual_nor[0].append(x_train_pca[i][0])
            visual_nor[1].append(x_train_pca[i][1])
    plt.scatter(visual_ano[0], visual_ano[1], c='r')
    plt.scatter(visual_nor[0], visual_nor[1], c='b')
    plt.legend(['anomaly', 'normal'])
    plt.show()



    results = collections.defaultdict(list)
    with open('results.csv', 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['benchmark_id', 'Model', 'ROC', 'precision at rank n', 'execution time'])
        for item in data_dic:
            for bench_id in data_dic[item]:
                x_item = []
                y_item = []
                contam = 0
                p = os.path.join(path, bench_id+'.csv')
                for i in pd.read_csv(p).iterrows():
                    if i[1][5] == 'anomaly':
                        y_item.append(1)
                        contam += 1
                    else:
                        y_item.append(0)
                    x_item.append(list(i[1][6:17]))
                x_item = np.array(x_item)
                y_item = np.array(y_item)
                contam /= len(y_item)
                contam = min(0.5, contam)
                # 定义模型
                classifiers = {'KNN': KNN(contamination=contam),
                               'LOF': LOF(contamination=contam),
                               'PCA': PCA(contamination=contam),
                               'LODA': LODA(contamination=contam)
                              }
                for cls in classifiers:
                    clf = classifiers[cls]
                    t0 = time.time()
                    x_item = standardizer(x_item)
                    clf.fit(x_item)
                    y_scores = clf.decision_function(x_item)
                    t1 = time.time()
                    duration = round(t1 - t0, ndigits=4)

                    roc = round(roc_auc_score(y_item, y_scores), ndigits=4)
                    prn = round(precision_n_scores(y_item, y_scores), ndigits=4)
                    results[cls].append(roc)

                    print('benchmark id:{bench_id}, model:{clf_name}, ROC:{roc}, precision @ rank n:{prn}, '
                          'execution time: {duration}s'.format(
                        bench_id=bench_id, clf_name=cls, roc=roc, prn=prn, duration=duration))

                    csv_writer.writerow([bench_id, cls, roc, prn, duration])

    f.close()

    ar_mean.sort()
    plt.figure(figsize=(7,7))
    plt.xlabel('anomaly rate')
    plt.ylabel('ROC-AUC')
    colors = ['r', 'g', 'b', '#FF1493', '#483D8B']
    i = 0
    for cls in classifiers:
        mean_y = [np.mean(results[cls][300:480]), np.mean(results[cls][480:670]), np.mean(results[cls][670:850]),
                  np.mean(results[cls][850:1030]), np.mean(results[cls][1030:1210]), np.mean(results[cls][:300]), ]
        plt.plot(ar_mean, mean_y, color=colors[i], label=cls)
        i += 1
    plt.title('Models in Benchmarks Compare')
    plt.legend(loc="upper right")
    plt.show()


