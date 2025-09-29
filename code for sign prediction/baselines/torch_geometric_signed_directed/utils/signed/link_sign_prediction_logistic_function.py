from typing import Tuple, Union

import numpy as np
from sklearn import linear_model, metrics


def link_sign_prediction_logistic_function(embeddings: np.ndarray, train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray, class_weight: Union[dict, str] = None) -> Tuple[float, float, float, float, float]:
    train_X1 = []
    test_X1 = []

    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))

    # 检查训练数据是否只有一个类别
    unique_classes = np.unique(train_y)
    if len(unique_classes) == 1:
        # 如果只有一个类别，我们使用DummyClassifier预测常量值
        from sklearn.dummy import DummyClassifier
        print(f"警告：训练数据只包含一个类别: {unique_classes[0]}。使用DummyClassifier代替。")
        classifier = DummyClassifier(strategy="constant", constant=unique_classes[0])
        classifier.fit(train_X1, train_y)
        pred = classifier.predict(test_X1)
        # 创建一个假的概率分布，因为DummyClassifier不提供predict_proba
        # 所有预测都是同一个类别，概率为1.0
        pred_p = np.zeros((len(test_X1), 2))
        if unique_classes[0] == 0:
            pred_p[:, 0] = 1.0  # 所有预测为类别0的概率为1
        else:
            pred_p[:, 1] = 1.0  # 所有预测为类别1的概率为1
    else:
        # 正常情况，使用逻辑回归
        classifier = linear_model.LogisticRegression(
            solver='lbfgs', max_iter=1000, class_weight=class_weight)
        classifier.fit(train_X1, train_y)
        pred = classifier.predict(test_X1)
        pred_p = classifier.predict_proba(test_X1)

    # 计算指标 - 需要处理test_y也只有一个类别的情况
    test_unique_classes = np.unique(test_y)
    if len(test_unique_classes) == 1:
        # 如果测试数据只有一个类别，准确率就是预测正确率
        accuracy = np.mean(pred == test_y)
        # 其他指标可能会出现问题，因为它们需要多个类别
        # 设置默认值
        precision = 1.0 if np.all(pred == test_y) else 0.0
        recall = 1.0 if np.all(pred == test_y) else 0.0
        f1 = 1.0 if np.all(pred == test_y) else 0.0
        # AUC不适用于单一类别，设置为0.5
        auc_score = 0.5
    else:
        # 正常计算所有指标
        accuracy = metrics.accuracy_score(test_y, pred)
        precision = metrics.precision_score(test_y, pred)
        recall = metrics.recall_score(test_y, pred)
        f1 = metrics.f1_score(test_y, pred)
        auc_score = metrics.roc_auc_score(test_y, pred_p[:, 1])

    return accuracy, precision, recall, f1, auc_score
