import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report
import pandas as pd

import utils
import o2o_utils as otils

import sys; sys.path.append(".")
from decision_trees.dectree import DecTree
from decision_trees.infogain_funcs import gini
from booster.boost import VotingBoost

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_boost_models(ref_y_train: NDArray):
    ratio = float(sum(ref_y_train == 0)) / sum(ref_y_train == 1)

    boost_models = [
        LinearSVC(
            class_weight='balanced',  # 关键！解决你的样本不平衡问题
            C=0.8,                    # 正则化系数，推荐 0.5 ~ 1.0 之间
            penalty='l2',             # 默认 L2 正则，对文本数据通常效果最好
            loss='squared_hinge',     # 默认损失函数，计算速度快
            dual=True,                # 文本特征维数高（TF-IDF）时，推荐开启对偶问题
            max_iter=3000,            # 增加最大迭代次数，防止报 "failed to converge" 警告
            random_state=42,          # 固定随机种子，保证结果可复现
            verbose=1                 # 设为 1 可以看训练进度
        ),
        DecTree(max_depth=12, min_sample_count_per_node=2, entropy_func=gini),
        RandomForestClassifier(
            n_estimators=100,          # 种100棵树
            class_weight='balanced',   # 同样处理不平衡
            n_jobs=-1,                 # 开启所有 CPU 核心并行训练，速度飞快
            max_depth=None,              # 稍微限制一下深度，防止树太深
            random_state=42
        ),
        XGBClassifier(
            n_estimators=100,
            max_depth=6,              # XGBoost 不喜深树，一般 3-10 之间
            learning_rate=0.1,
            scale_pos_weight=ratio,   # 【最关键】专门应对样本不平衡
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42
        ),
    ]

    return boost_models


def run_boost(x_train, x_test, y_train, y_test, model_path, save_path):
    if model_path is None:
        models = build_boost_models(y_train)
        boost = VotingBoost(*models)
        boost.train(x_train, y_train)
        utils.save_model(boost, save_path)
    else:
        boost = utils.load_model(model_path, return_only_model=True)
    
    if x_test is not None:
        y_pred = boost.predict(x_test)
    if y_test is not None:
        print(classification_report(y_test, y_pred))
    
    try:
        return y_pred
    except UnboundLocalError:
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-o", "--model-output", type=str, default=None)
    parser.add_argument("-t", "--test", type=str, default=None, help="Save path for result file")
    parser.add_argument("--max-text-dim", type=int, default=1600)

    args = parser.parse_args()

    if not args.test:
        df = otils.load_o2o_csv("./train.csv")
        x_train, x_val, y_train, y_val, vec = otils.feat_extraction(df, max_feat_dim=int(args.max_text_dim), val_split_rate=0.2, vectorizer_type="count")
        print(f"Load O2O data complete: training set - {x_train.shape}{(', test set - ' + str(x_val.shape)) if x_val is not None else ''}")

        run_boost(x_train, x_val, y_train, y_val, model_path=args.model, save_path=args.model_output)

    else:
        df_train = otils.load_o2o_csv("./train.csv")
        df_test = otils.load_o2o_csv("./test_new.csv", sep=',')
        vectorizer = otils.get_train_test_vectorizer(df_train, df_test, max_feat_dim=None, vectorizer_type='count')

        x_train, _, y_train, _ = otils.feat_extraction(df_train, vectorizer)
        x_test, _, _, _ = otils.feat_extraction(df_test, vectorizer)

        y_pred = run_boost(x_train, x_test, y_train, None, args.model, args.model_output)

        df_ypred = pd.DataFrame(y_pred, columns=['label'])
        df_out = pd.concat([df_test['id'], df_ypred['label']], axis=1)

        df_out.to_csv(args.test, sep=',', index=False)

        exit(0)

        df = otils.load_o2o_csv("./test_new.csv", sep=',', other_path="./train.csv", other_sep='\t')
        df_test = df[np.isnan(df['label'])]
        df_test_indices = df_test.index

        x_test, _, _, _, vec = otils.feat_extraction(df, max_feat_dim=int(args.max_text_dim), vectorizer_type="count")
        uid_test = df['id'].values

        x_test = x_test[df_test_indices]
        uid_test = uid_test[df_test_indices]
        print(f"Load O2O test data complete: {x_test.shape}")

        y_pred = run_boost(None, x_test, None, None, model_path=args.model, save_path=None)

        import pdb; pdb.set_trace()
        pass

