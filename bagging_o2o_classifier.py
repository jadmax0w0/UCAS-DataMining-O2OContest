import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report
import pandas as pd

import utils
import o2o_utils as otils

import sys; sys.path.append(".")
from bagger.bag import VotingBagger

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


def build_bagger_models(ref_y_train: NDArray, with_id: bool = False):
    ratio = float(sum(ref_y_train == 0)) / sum(ref_y_train == 1)

    bagged_models = [
        SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced'),
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=ratio,
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
            max_features='sqrt',
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        ),
    ]

    if with_id:
        bagged_models = [(f"{i}", model) for i, model in enumerate(bagged_models)]

    return bagged_models


def run_bagger(x_train, x_test, y_train, y_test, model_path, save_path):
    if model_path is None:
        models = build_bagger_models(y_train, with_id=False)
        bagger = VotingBagger(*models)
        # bagger = VotingClassifier(models, voting="hard")
        try:
            bagger.train(x_train, y_train)
        except AttributeError:
            bagger.fit(x_train, y_train)
        utils.save_model(bagger, save_path)
    else:
        bagger = utils.load_model(model_path, return_only_model=True)
    
    if x_test is not None:
        y_pred = bagger.predict(x_test)
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

        run_bagger(x_train, x_val, y_train, y_val, model_path=args.model, save_path=args.model_output)

    else:
        df_train = otils.load_o2o_csv("./train.csv")
        df_test = otils.load_o2o_csv("./test_new.csv", sep=',')
        vectorizer = otils.get_train_test_vectorizer(df_train, df_test, max_feat_dim=None, vectorizer_type='count')

        x_train, _, y_train, _ = otils.feat_extraction(df_train, vectorizer)
        x_test, _, _, _ = otils.feat_extraction(df_test, vectorizer)

        y_pred = run_bagger(x_train, x_test, y_train, None, args.model, args.model_output)
        print(f"Prediction complete, {y_pred.sum().item()} positive samples discovered")

        if args.test_break:
            import pdb; pdb.set_trace()

        df_ypred = pd.DataFrame(y_pred, columns=['label'])
        df_out = pd.concat([df_test['id'], df_ypred['label']], axis=1)

        df_out.to_csv(args.test, sep=',', index=False)
