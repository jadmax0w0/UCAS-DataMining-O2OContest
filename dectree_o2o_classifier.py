import numpy as np
from numpy.typing import NDArray
import sys; sys.path.append(".")
from decision_trees.dectree import DecTree
from decision_trees.infogain_funcs import gini
import utils
import o2o_utils as otils
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MAX_TEXT_FEAT_DIM = None
MAX_DEPTH = 10
MIN_SAMPLE_PER_NODE = 2

def run_decision_tree(x_train, x_test, y_train, y_test, model_path, save_path):
    if model_path is None:
        dectree = DecTree(MAX_DEPTH, MIN_SAMPLE_PER_NODE, gini)
        # dectree = DecisionTreeClassifier(
        #     criterion='gini',          # CART 标准也是用 Gini 系数
        #     class_weight='balanced',   # 【关键】必须加！针对你的 O2O 数据不平衡
        #     max_depth=None,
        #     min_samples_leaf=MIN_SAMPLE_PER_NODE,
        #     ccp_alpha=0.0,             # 之后可以用这个参数做“后剪枝” (Cost Complexity Pruning)
        #     random_state=42
        # )
        # dectree = RandomForestClassifier(
        #     n_estimators=100,          # 种100棵树
        #     class_weight='balanced',   # 同样处理不平衡
        #     n_jobs=-1,                 # 开启所有 CPU 核心并行训练，速度飞快
        #     max_depth=None,              # 稍微限制一下深度，防止树太深
        #     random_state=42
        # )
        # ratio = float(sum(y_train == 0)) / sum(y_train == 1)
        # dectree = XGBClassifier(
        #     n_estimators=100,
        #     max_depth=6,              # XGBoost 不喜深树，一般 3-10 之间
        #     learning_rate=0.1,
        #     scale_pos_weight=ratio,   # 【最关键】专门应对样本不平衡
        #     use_label_encoder=False,
        #     eval_metric='logloss',
        #     n_jobs=-1,
        #     random_state=42
        # )
        try:
            dectree.train(x_train, y_train)
        except AttributeError:
            dectree.fit(x_train, y_train)
        utils.save_model(dectree, save_path)
    else:
        dectree = utils.load_model(model_path)
    
    y_pred = dectree.predict(x_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-o", "--model-output", type=str, default=None)

    args = parser.parse_args()

    df = otils.load_o2o_csv("./train.csv")
    # df_train, df_val = otils.split_training_data(df, val_portion=0.2)

    # x_train, y_train, vec_train = otils.feat_extraction(df_train, max_feat_dim=MAX_TEXT_FEAT_DIM)
    # x_val, y_val, vec_val = otils.feat_extraction(df_val, max_feat_dim=MAX_TEXT_FEAT_DIM)
    x_train, x_val, y_train, y_val, vec = otils.feat_extraction(df, MAX_TEXT_FEAT_DIM, val_split_rate=0.2, vectorizer_type="count")

    print(f"Load O2O data complete: training set - {x_train.shape}, test set - {x_val.shape}")

    run_decision_tree(x_train, x_val, y_train, y_val, model_path=args.model, save_path=args.model_output)
