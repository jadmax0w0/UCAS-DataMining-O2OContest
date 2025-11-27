import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
import sklearn.decomposition as skdecom
from typing import Optional, List
import sys; sys.path.append(".")
from base_model import BaseModel


def shuffle_along_axis(a, axis=0, return_perm = False):
    idx = np.random.permutation(a.shape[axis])
    shuffled = np.take(a, idx, axis=axis)
    return shuffled if not return_perm else (shuffled, idx)


def load_mnist(
        path: str,
        shuffle: bool = True,
        load_range: Optional[List[int]] = None,
        standardize: bool = True,
        pca_n: Optional[int] = None
):
    """
    Returns:
        (x_train, x_test, y_train, y_test): shaped `[N, D]`, `[N, D]`, `[N]`
    """
    info = loadmat(path)
    # dict_keys([
    # '__header__', '__version__', '__globals__', 
    # 'train0', 'test0', 'train1', 'test1', 'train2', 'test2', 'train3', 'test3', 
    # 'train4', 'test4', 'train5', 'test5', 'train6', 'test6', 'train7', 'test7', 
    # 'train8', 'test8', 'train9', 'test9'])
    x_train = [info[f"train{i}"] for i in (range(10) if load_range is None else load_range)]
    x_test  = [info[f"test{i}"] for i in (range(10) if load_range is None else load_range)]
    y_train = [(np.full(info[f"train{i}"].shape[0], i)) for i in (range(10) if load_range is None else load_range)]
    y_test  = [(np.full(info[f"test{i}"].shape[0], i)) for i in (range(10) if load_range is None else load_range)]

    x_train = np.concat(x_train, axis=0)
    x_test  = np.concat(x_test, axis=0)
    y_train = np.concat(y_train, axis=0)
    y_test  = np.concat(y_test, axis=0)

    if standardize:
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    if pca_n is not None:
        print(f"Applying PCA on MNIST dataset, n_components = {pca_n}")
        pca = skdecom.PCA(n_components=pca_n)
        x_train = pca.fit_transform(x_train)
        x_test = pca.fit_transform(x_test)

    if shuffle:
        x_train, idx = shuffle_along_axis(x_train, axis=0, return_perm=True)
        y_train = np.take(y_train, idx, axis=0)
    
    return x_train, x_test, y_train, y_test

def get_acc(y_pred: NDArray, y_test: NDArray):
    assert y_pred.shape == y_test.shape, f"{y_pred.shape=}, {y_test.shape=}"

    corrects = y_pred == y_test
    return np.sum(corrects) / y_pred.size

def get_prec(y_pred: NDArray, y_test: NDArray):
    classes = np.unique(np.concatenate((y_test, y_pred)))
    precision_sum = 0
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_test == c))
        fp = np.sum((y_pred == c) & (y_test != c))
        
        if (tp + fp) > 0:
            precision_sum += tp / (tp + fp)
        else:
            precision_sum += 0.0
            
    return precision_sum / len(classes)

def get_recall(y_pred: NDArray, y_test: NDArray):
    classes = np.unique(np.concatenate((y_test, y_pred)))
    recall_sum = 0
    
    for c in classes:
        tp = np.sum((y_pred == c) & (y_test == c))
        fn = np.sum((y_pred != c) & (y_test == c))
        
        if (tp + fn) > 0:
            recall_sum += tp / (tp + fn)
        else:
            recall_sum += 0.0
            
    return recall_sum / len(classes)

def get_f1(y_pred: NDArray, y_test: NDArray):
    classes = np.unique(np.concatenate((y_test, y_pred)))
    f1_sum = 0
    
    # Get F1 for each class first average then; instead of get Prec. and Rec. for all classes first then calc F1
    for c in classes:
        tp = np.sum((y_pred == c) & (y_test == c))
        fp = np.sum((y_pred == c) & (y_test != c))
        fn = np.sum((y_pred != c) & (y_test == c))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (p + r) > 0:
            f1_sum += 2 * (p * r) / (p + r)
        else:
            f1_sum += 0.0
            
    return f1_sum / len(classes)

def save_model(model: BaseModel, path: str):
    if path is None:
        return None
    
    import pickle
    import os
    from datetime import datetime

    fdir, _ = os.path.split(path)
    if fdir != "":
        os.makedirs(fdir, exist_ok=True)
    
    with open(path, mode="wb") as f:
        pickle.dump({"model": model, "train_time": datetime.now()}, f)

def load_model(path: str, return_only_model = True):
    if path is None:
        return None
    
    import pickle
    with open(path, mode="rb") as f:
        model = pickle.load(f)
    return model['model'] if return_only_model else model

def plot_loss(loss_file: str):
    import matplotlib.pyplot as plt
    from collections import defaultdict

    with open(loss_file, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    
    train_losses = defaultdict(list)
    for l in lines:
        epoch, loss = l.split(": ")
        epoch = int(epoch.removeprefix("Epoch "))
        train_losses[epoch].append(float(loss))
    
    train_times = -1
    incomplete_record = False
    for losses in train_losses.values():
        if train_times < 0:
            train_times = len(losses)
            continue
        if train_times != len(losses):
            incomplete_record = True
            train_times = min(train_times, len(losses))
    if incomplete_record:
        print("Incomplete loss record detected")
        for epoch in train_losses.keys:
            train_losses[epoch] = train_losses[epoch][:train_times]
    
    for train_id in range(train_times):
        # import pdb; pdb.set_trace()
        epoch_losses = [(epoch, losses[train_id]) for epoch, losses in train_losses.items()]
        epoch_losses = sorted(epoch_losses, key=lambda x: x[0])
        # epoch_losses = [(e+1, v) for e, v in epoch_losses]
        losses = [l for _, l in epoch_losses]
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

def plot_roc(y_test: NDArray, y_pred: NDArray, softmax_needed = False, class_names = None, figsize = (10, 8)):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    from scipy.special import softmax

    if softmax_needed:
        y_pred = softmax(y_pred, axis=-1)
    
    n_classes = y_pred.shape[1]
    
    # 2. 将 y_test 转换为二值化矩阵 (one-hot 形式)
    # 这一步是必须的，因为 roc_curve 只能处理二分类逻辑
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    
    # 3. 计算每一类的 ROC 曲线和 AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        # 取出第 i 列作为正例概率
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 4. 计算微平均 (Micro-average) ROC 曲线
    # 这代表了模型在全局层面的预测能力（将所有类别的预测拉平看作一个大二分类）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 5. 开始绘图
    plt.figure(figsize=figsize)
    
    # 绘制微平均曲线 (通常用粗虚线表示)
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # 绘制每一类的曲线
    # 生成颜色循环
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color in zip(range(n_classes), colors):
        label_name = class_names[i] if class_names else f'Class {i}'
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} (area = {1:0.2f})'.format(label_name, roc_auc[i]))

    # 绘制对角线 (随机猜测线)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # 图表修饰
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    # plt.title('Multi-class Receiver Operating Characteristic', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # plot_loss("model/nn_cnn_full2_loss.log")
    # plot_loss("model/svm_linear_full_10ep_loss.log")
    # y = load_model("./model/svm_linear_full3_y.np")
    y = load_model("./model/nn_cnn_full2_y.np")
    plot_roc(y_test=y['y_test'], y_pred=y['y_pred_raw'], softmax_needed=True)