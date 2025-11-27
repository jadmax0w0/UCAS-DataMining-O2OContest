import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List, Optional, Union, Literal

O2O_UID_KEY = 'id'
O2O_LABEL_KEY = 'label'
O2O_COMMENT_KEY = 'comment'

EXTRA_STOPWORDS = ['##', '.', 'a', 'ain', 'aren', 'c', 'couldn', 'd', 'didn', 'doesn', 'don', 'exp', 'hadn', 'hasn', 'haven', 'i', 'isn', 'lex', 'll', 'm', 'mon', 's', 'shouldn', 't', 've', 'wasn', 'weren', 'won', 'wouldn', '~', '±', '÷', 'β', 'δ', 'λ', 'ξ', 'ψ', 'в', '…', '′', '″', '℃', 'ⅲ', '∈', '∧', '∪', '≈', '─', '☆', '㈧', '为什', '什', '倒', '傥', '元', '先', '兼', '前', '吨', '唷', '啪', '啷', '喔', '外', '多年', '大面儿', '天', '始', '常', '後', '抗拒', '敞开', '数', '新', '方', '日', '昉', '末', '次', '毫无保留', '没', '漫', '然', '特', '特别', '理', '皆', '目前为止', '竟', '策略', '若果', '莫', '见', '设', '话', '说', '赶早', '赶晚', '达', '限', '非', '面', '麽', '０', '１', '２', '３', '５', 'ａ', 'ｂ', 'ｃ', 'ｄ', 'ｅ', 'ｆ', 'ｇ', 'ｈ', 'ｉ', 'ｊ', 'ｌ', 'ｎ', 'ｏ', 'ｒ', 'ｔ', 'ｘ', 'ｚ', '｛', '｜']

def load_o2o_csv(path: str, sep: str = "\t", other_path: Optional[str] = None, other_sep: Optional[str] = None):
    """
    Args:
        other_path: if specified, concat 'comment' field values in `other_path` file into `path` file, to make sure that the afterward-vectorized texts dimensions are aligned with `other_path`
    """
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    if other_path is not None:
        df_other = pd.read_csv(other_path, sep=other_sep, encoding='utf-8')
        df = pd.concat([df, df_other], ignore_index=True)
    return df

def split_training_data(df: DataFrame, val_portion: float = 0.2):
    """
    Returns:
        train_split, val_split
    """
    train_df = df.sample(frac=(1 - val_portion), random_state=42)
    val_df = df.drop(train_df.index)
    return train_df, val_df

def load_cn_stopwords(
        paths: Union[str, List[str]] = ["./stopwords/cn_stopwords.txt", "./stopwords/baidu_stopwords.txt", "./stopwords/hit_stopwords.txt", "./stopwords/scu_stopwords.txt"],
        sep: str = '\n',
        append_extra: bool = True,
):
    if isinstance(paths, str):
        paths = [paths]
    
    words = []
    for p in paths:
        with open(p, mode='r', encoding='utf-8') as f:
            cont = f.read()
        words.extend(cont.split(sep))
    
    if append_extra:
        words.extend(EXTRA_STOPWORDS)
    
    words = list(set(words))
    return words

def get_train_test_vectorizer(
        df_train: DataFrame,
        df_test: DataFrame,
        max_feat_dim: Optional[int] = None,
        cn_stopwords: Optional[List[str]] = None,
        vectorizer_type: Literal['tfidf', 'count'] = 'count',
):
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    def cn_tokenizer(text: str):
        return jieba.lcut(text)
    
    if cn_stopwords is None:
        print("No stopwords are specified, reading from ./stopwords")
        cn_stopwords = load_cn_stopwords()  # read stopwords using default path
    
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            tokenizer=cn_tokenizer,
            encoding="utf-8",
            stop_words=cn_stopwords,
            ngram_range=(1,2),
            max_features=max_feat_dim,
        )
    elif vectorizer_type == "count":
        vectorizer = CountVectorizer(min_df=10, ngram_range=(1, 1), tokenizer=cn_tokenizer, max_features=max_feat_dim)
    else:
        raise NotImplementedError(f"Unknown vectorizer type \"{vectorizer_type}\"")
    
    vectorizer.fit(df_train[O2O_COMMENT_KEY].values.tolist() + df_test[O2O_COMMENT_KEY].values.tolist())
    return vectorizer

def feat_extraction(
        df: DataFrame,
        vectorizer,
        val_split_rate: Optional[float] = None,
):
    """
    Returns:
        (x_train, x_val, y_train, y_val)
    """
    x = vectorizer.transform(df[O2O_COMMENT_KEY]).toarray()
    try:
        y = df[O2O_LABEL_KEY].values
    except KeyError:
        y = df[O2O_UID_KEY].values

    if val_split_rate is not None and 0 < val_split_rate < 1:
        N = x.shape[0]
        n_train = int(N * (1 - val_split_rate))

        rng = np.random.default_rng(seed=42)
        rand_indices = rng.permutation(x.shape[0])

        indices_train = rand_indices[:n_train]
        indices_val = rand_indices[n_train:]

        x_train, x_val = x[indices_train], x[indices_val]
        y_train, y_val = y[indices_train], y[indices_val]

        return x_train, x_val, y_train, y_val

    return x, None, y, None