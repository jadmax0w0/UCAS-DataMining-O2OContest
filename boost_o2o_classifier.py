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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


def build_boost_models(ref_y_train: NDArray, with_id: bool = False):
    ratio = float(sum(ref_y_train == 0)) / sum(ref_y_train == 1)

    boost_models = [
        LogisticRegression(C=1.2),
        LinearSVC(
            class_weight='balanced',  # 关键！解决你的样本不平衡问题
            C=0.8,                    # 正则化系数，推荐 0.5 ~ 1.0 之间
            penalty='l2',             # 默认 L2 正则，对文本数据通常效果最好
            loss='squared_hinge',     # 默认损失函数，计算速度快
            dual=True,                # 文本特征维数高（TF-IDF）时，推荐开启对偶问题
            max_iter=5000,            # 增加最大迭代次数，防止报 "failed to converge" 警告
            random_state=42,          # 固定随机种子，保证结果可复现
            verbose=1                 # 设为 1 可以看训练进度
        ),
        SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced'),
        DecTree(max_depth=12, min_sample_count_per_node=2, entropy_func=gini),
        DecisionTreeClassifier(
            criterion='gini',          # CART 标准也是用 Gini 系数
            class_weight='balanced',   # 【关键】必须加！针对你的 O2O 数据不平衡
            max_depth=None,
            min_samples_leaf=2,
            ccp_alpha=0.0,             # 之后可以用这个参数做“后剪枝” (Cost Complexity Pruning)
            random_state=42
        ),
        RandomForestClassifier(
            n_estimators=100,          # 种100棵树
            class_weight='balanced',   # 同样处理不平衡
            n_jobs=-1,                 # 开启所有 CPU 核心并行训练，速度飞快
            max_depth=None,              # 稍微限制一下深度，防止树太深
            random_state=42
        ),
        XGBClassifier(
            n_estimators=200,
            max_depth=6,              # XGBoost 不喜深树，一般 3-10 之间
            learning_rate=0.1,
            scale_pos_weight=ratio,   # 【最关键】专门应对样本不平衡
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42,
            reg_alpha=0.5,
            objective='binary:logistic',
            colsample_bytree=0.6,
        ),
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            max_features='sqrt',       # 核心：处理高维文本特征
            min_samples_leaf=10,       # 核心：忽略生僻词造成的过细分裂
            subsample=0.8,
            random_state=42
        ),
    ]

    if with_id:
        boost_models = [(f"{i}", model) for i, model in enumerate(boost_models)]

    return boost_models


def run_boost(x_train, x_test, y_train, y_test, model_path, save_path):
    if model_path is None:
        models = build_boost_models(y_train, with_id=False)
        boost = VotingBoost(*models)
        # boost = VotingClassifier(models, voting="hard")
        try:
            boost.train(x_train, y_train)
        except AttributeError:
            boost.fit(x_train, y_train)
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
    parser.add_argument("-b", "--test-break", action="store_true", help="Combined with -t, appends a breakpoint before writing out test result file")
    parser.add_argument("--max-text-dim", type=int, default=1600)

    args = parser.parse_args()

    if not args.test:
        df_train = otils.load_o2o_csv("./train.csv")
        df_test = otils.load_o2o_csv("./test_new.csv", sep=',')
        vectorizer = otils.get_train_test_vectorizer(df_train, df_test, max_feat_dim=None, vectorizer_type='count')
        x_train, x_val, y_train, y_val = otils.feat_extraction(df_train, vectorizer, val_split_rate=0.2)
        print(f"Load O2O data complete: training set - {x_train.shape}{(', val set - ' + str(x_val.shape)) if x_val is not None else ''}")

        run_boost(x_train, x_val, y_train, y_val, model_path=args.model, save_path=args.model_output)

    else:
        df_train = otils.load_o2o_csv("./train.csv")
        df_test = otils.load_o2o_csv("./test_new.csv", sep=',')
        vectorizer = otils.get_train_test_vectorizer(df_train, df_test, max_feat_dim=None, vectorizer_type='count')

        x_train, _, y_train, _ = otils.feat_extraction(df_train, vectorizer)
        x_test, _, _, _ = otils.feat_extraction(df_test, vectorizer)

        y_pred = run_boost(x_train, x_test, y_train, None, args.model, args.model_output)
        print(f"Prediction complete, {y_pred.sum().item()} positive samples discovered")

        if args.test_break:
            import pdb; pdb.set_trace()

        df_ypred = pd.DataFrame(y_pred, columns=['label'])
        df_out = pd.concat([df_test['id'], df_ypred['label']], axis=1)

        df_out.to_csv(args.test, sep=',', index=False)
