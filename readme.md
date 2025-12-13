运行已训练的检查点：

```bash
python bagging_o2o_classifier.py -m model/bag_SXG_full.pkl -t /path/to/test/result/file
```

从头开始训练 & 评测：

```bash
python bagging_o2o_classifier.py -o /path/to/model/save/path -t /path/to/test/result/file
```
