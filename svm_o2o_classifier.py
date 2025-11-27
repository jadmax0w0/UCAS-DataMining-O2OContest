import numpy as np
from numpy.typing import NDArray
import sys; sys.path.append(".")
from svms.linear_svm import SimpleLinearSVM
from svms.multiclass_svm import MultiClassSVM
import utils
import o2o_utils as otils
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC

MAX_TEXT_FEAT_DIM = None
TRAIN_EPOCHS = 100


def run_linear_svm(x_train, x_test, y_train, y_test, model_path, save_path, loss_path, label_output):
    if model_path is None:
        # svm = MultiClassSVM(lr=0.01, svm_class=SimpleLinearSVM, lambda_value=0.001)
        # svm = SimpleLinearSVM(lr=0.01, lambda_value=0.001)
        svm = LinearSVC(
            class_weight='balanced',  # 关键！解决你的样本不平衡问题
            C=0.8,                    # 正则化系数，推荐 0.5 ~ 1.0 之间
            penalty='l2',             # 默认 L2 正则，对文本数据通常效果最好
            loss='squared_hinge',     # 默认损失函数，计算速度快
            dual=True,                # 文本特征维数高（TF-IDF）时，推荐开启对偶问题
            max_iter=3000,            # 增加最大迭代次数，防止报 "failed to converge" 警告
            random_state=42,          # 固定随机种子，保证结果可复现
            verbose=1                 # 设为 1 可以看训练进度
        )
        try:
            svm.train(x_train, y_train, epochs=TRAIN_EPOCHS, loss_path=loss_path)
        except AttributeError:
            svm.fit(x_train, y_train)
        utils.save_model(svm, save_path)
    else:
        svm = utils.load_model(model_path)
    
    if label_output is not None:
        try:
            y_pred = svm.predict(x_test, raw_output=False)
            y_pred_raw = svm.predict(x_test, raw_output=True)
            utils.save_model({"y_pred": y_pred, "y_pred_raw": y_pred_raw, "y_test": y_test}, label_output)
        except TypeError:
            label_output = None
    
    if label_output is None:
        try:
            y_pred = svm.predict(x_test, raw_output=False)
        except TypeError:
            y_pred = svm.predict(x_test)
    
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-o", "--model-output", type=str, default=None)
    parser.add_argument("-l", "--loss-file", type=str, default=None)
    parser.add_argument("-y", "--label-output", type=str, default=None)

    args = parser.parse_args()

    df = otils.load_o2o_csv("./train.csv")
    # df_train, df_val = otils.split_training_data(df, val_portion=0.2)

    # x_train, y_train, vec_train = otils.feat_extraction(df_train, max_feat_dim=MAX_TEXT_FEAT_DIM)
    # x_val, y_val, vec_val = otils.feat_extraction(df_val, max_feat_dim=MAX_TEXT_FEAT_DIM)
    x_train, x_val, y_train, y_val, vec = otils.feat_extraction(df, MAX_TEXT_FEAT_DIM, val_split_rate=0.2, vectorizer_type="count")

    print(f"Load O2O data complete: training set - {x_train.shape}, test set - {x_val.shape}")

    run_linear_svm(x_train, x_val, y_train, y_val, model_path=args.model, save_path=args.model_output, loss_path=args.loss_file, label_output=args.label_output)
