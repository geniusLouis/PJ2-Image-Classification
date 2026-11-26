# PJ2 图像分类

复旦大学智能信息处理课程项目，组号 **2-2**。在 Food-101 六类子集和 Imagenette 320px 上，使用全局颜色特征和局部特征分别完成分类。每套数据使用相同的官方划分和统一的评估口径，对照两种特征的结果。

**提交截止** 2026 年 12 月 2 日 23:59（北京时间）。组 2-2 的课堂汇报安排在 PJ4。

## 数据与实验协议

| 数据集 | 固定划分 | 类别 |
| --- | --- | --- |
| Food-101 课程六类子集 | `meta/train.txt` 和 `meta/test.txt`，每类 750/250 张，共 4500/1500 张 | sushi、tacos、takoyaki、tiramisu、tuna_tartare、waffles |
| Imagenette 320px | 课程包原有 `train/` 和 `val/`，合计 9469/3925 张；将 `val/` 作为最终测试集 | tench、English springer、cassette player、chain saw、church、French horn、garbage truck、gas pump、golf ball、parachute |

从 Elearning 下载两份课程包 `PJ2--food_101_dataset.tgz` 和 `PJ2--imagenette2-320.tgz`。照片由各成员从课程或数据集发布渠道获取。来源见 [ETH Zurich Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) 和 [fastai Imagenette](https://github.com/fastai/imagenette)（[320px 发布包](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz)）。优先使用 Elearning 同名课程包，以匹配校验值。课程包的 SHA-256 见 `code/data/archives.sha256`。

在仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/setup_data.py "/存放两个课程数据压缩包的目录"
python code/prepare_splits.py
```

成功后生成 `code/data/raw/food-101-coursework-6classes/` 和 `code/data/raw/imagenette2-320/`。原图压缩包及解包目录保存在各成员本地。数据准备脚本先校验压缩包，再解包到约定路径。第 1 章的已核对清单、类别映射和审计结果位于 `code/results/data_audit/`。后续模块各自提供复现命令。

两条实验路线分别使用 **HSV 全图颜色直方图 → LinearSVC** 和 **SIFT 局部描述子 → K-means 视觉词袋（BoVW）→ LinearSVC**。两套数据分别训练，共形成四组主实验。分类器使用相同的训练集内三折分层验证策略、C 候选（0.1、1、10）和随机种子 42，以平均验证 Macro-F1 选参数。BoVW 在每个训练折独立学习词典和 TF-IDF 权重；验证图只用于变换及评分。BoVW 以 K=128、L1 为基线，比较 K=64、128、256 和四种直方图表示。每图最多保留 128 个 SIFT 描述子，每次从训练图保留的描述子中抽取最多 100,000 个拟合词典。最终测试集用于所选配置的一次评估。统一报告 Accuracy、Macro-F1、逐类指标、混淆矩阵和分阶段耗时。

## 六人分工

| 顺序与建议交付 | 组员 · 报告章节 | 读取 | 编写与实验 | 交付 |
| --- | --- | --- | --- | --- |
| 1 · 11 月 19 日 | 卢远卓 · 数据与划分 | 两个课程包及官方划分 | `prepare_splits.py`：生成两套清单，核对类别、图片解码、训练/测试交集，分析每类数量和尺寸分布 | 清单、类名映射、审计表及分布图，第 1 章 |
| 2 · 11 月 21 日 | 于智博 · 全局颜色特征 | 第 1 章清单和原图 | `extract_color.py`：计算 HSV 联合直方图与 L1 归一化；对比两种分箱配置的维数、耗时和特征示例 | 两套数据的颜色特征缓存、分箱配置与直方图示例，第 2 章 |
| 3 · 11 月 23 日 | 竺钱成 · 全局特征分类 | 第 1 章清单和第 2 章颜色特征 | `classify_color.py`：训练 LinearSVC，在训练集内选 C，评估两套数据；提供共用评分函数 | 全局预测 CSV、指标表、混淆矩阵及分类设置，第 3 章 |
| 4 · 11 月 25 日 | 张帅宇 · 局部描述子 | 第 1 章清单和原图，第 3 章确认的评估口径 | `extract_sift.py`：提取 SIFT 的 128 维描述子；比较关键点数量、无关键点比例和提取时间，展示代表图 | 描述子索引与缓存、关键点示例和数量统计，第 4 章 |
| 5 · 11 月 28 日 | 杨晓芳 · 局部特征分类 | 第 4 章描述子、第 1 章清单及第 3 章评分函数 | `classify_bovw.py`：每套数据各自用训练描述子拟合词典，量化并归一化词频，用同口径 LinearSVC 分类；记录词典规模和训练开销 | 局部预测 CSV、指标表、混淆矩阵和词袋设置，第 5 章 |
| 6 · 11 月 30 日 | 郑周锐 · 对比与误差分析 | 第 3、5 章对齐的测试预测、特征耗时和配置 | `compare_results.py`：校验样本一对一对应，比较两条路线的总体和逐类指标，统计互有胜负的样本并复核典型错例 | 四组结果汇总、对比图、错例及结论，第 6 章 |

每人交付可运行脚本、小体积统计表和代表图，以及同一 Word 对应章节。章节至少包含实验方法、关键函数原理和代码、实测结果、分析、运行命令。大型特征缓存与原图保存在各自本地，后续成员通过清单及脚本重建。

## 接口与交接

- 使用第 1 章交付的 `food101_split.csv` 和 `imagenette_split.csv`。每行含 `sample_id,split,path,label`，其中 `sample_id` 在同一数据集唯一，`path` 相对 `code/data/raw/`，`split` 统一为 `train` 或 `test`。
- 特征保存时保留 `sample_id`。预测 CSV 使用 `sample_id,y_true,y_pred`，每个最终测试样本恰好一行。比较脚本按 ID 对齐并验证真实标签。
- 实验脚本统一放在 `code/`，小体积实验产物放在 `code/results/`。每章结果记下运行命令、超参数、依赖版本、随机种子及测试样本数量。
- 接力顺序按分工表执行。上一位合并后，下一位同步最新 `main`，提交自己的代码、实验结果和对应 Word 章节。共享 Word 一次由一人编辑。

## 第 1 章核对结果

共验证 19,394 张图像，全部能解码。两个数据集均无跨划分的重复样本 ID 或相同文件字节。Food-101 的一张训练图和一张测试图解码后像素相同，详情见 `code/results/data_audit/exact_cross_split_matches.csv`。两条分类路线统一保留课程原划分。

## 报告与提交

组内维护一份汇总报告 `report/project2-report.docx`。各成员填写自己负责的章节，包含方法、所用函数的原理、关键代码、实测结果及分析。12 月 1 日复核四组实验结果。上传汇总 Word 时，按 Elearning 题面命名为 `Project2-学号-姓名-2-2.docx`，学号和姓名使用提交人信息。

## 第 3 章复现与交接

于智博的第 2 章已通过 PR #1 合并。竺钱成在两套数据上使用相同的三折分层验证规则，联合选择 HSV 分箱和 LinearSVC 的 C，再用全部训练样本拟合所选模型并评估官方测试集。

已有原图时，在仓库根目录运行以下命令。全新环境先按前文安装依赖并解压课程包。

```bash
python code/extract_color.py
python code/classify_color.py
```

已有与清单匹配的四份颜色缓存时，可以直接运行第二条命令。脚本从自身目录定位输入，支持在其他工作目录运行。缓存包含的清单 SHA-256、样本 ID、标签、划分、分箱、维数和 L1 归一化均会被核对。

### 分类设置与结果

两种分箱为 `8x8x8` 和 `16x8x8`，C 为 `0.1, 1, 10`。训练样本保持清单顺序，使用 `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`。每套数据的全部候选共用相同三折，以平均验证 Macro-F1 最大者胜出。完全相等时依次选择较低维数和较小 C。HSV 直方图沿用第 2 章的逐图 L1 归一化。

LinearSVC 使用 L2 正则、平方 hinge 损失、one-vs-rest、`dual="auto"`、`tol=1e-4`、`max_iter=10000`、`random_state=42`、`fit_intercept=True`、`intercept_scaling=1`、`class_weight=None`。每套数据最终只评估训练验证选出的一个配置。发生不收敛时脚本报错，保留问题供处理。

| 数据集 | 所选分箱与 C | 测试图数 | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- |
| Food-101 六类 | 16x8x8，C=10 | 1500 | 0.368667 | 0.365506 |
| Imagenette | 16x8x8，C=10 | 3925 | 0.393376 | 0.379191 |

所有小体积输出位于 `code/results/color_classification/`。

- `summary.csv`、`config.json` 记录所选配置、验证与测试指标、实测分类耗时、软件版本及清单校验值。耗时排除特征提取、文件读写和指标计算。
- `cv_search.csv` 记录每套数据六种候选的验证均值与标准差，标准差使用 `ddof=0`。`cv_folds.csv` 记录全部 36 次验证拟合，`{dataset}_folds.csv` 记录每个训练样本作为验证样本的折号。
- `{dataset}_predictions.csv` 使用 `sample_id,y_true,y_pred`，分别覆盖全部 1500 和 3925 个测试样本。第 6 章可按 ID 与局部路线一对一对齐。
- `{dataset}_per_class.csv` 包含逐类 precision、recall、F1 和 support。`{dataset}_confusion.csv` 为计数矩阵，行是真实类别，列是预测类别；同名 PNG 的颜色按行归一化，格内数字为计数。

原图、特征缓存和 `code/results/cache/color_models/` 中的数值模型保留本地，通过脚本重建。后者保存权重、截距、类别顺序、分箱和 C，可直接核对线性决策分数。

### 共用评分函数

`code/evaluate.py` 提供 `score_predictions(y_true, y_pred, labels)`。`labels` 按第 1 章对应 `*_labels.csv` 的 `label_id` 排序后传入，使用完整的类别列表。Accuracy 为预测正确比例；每类 F1 为 `2PR/(P+R)`，Macro-F1 是全部类别 F1 的等权平均，零分母项记为 0。返回值包含总体指标、逐类表和按固定顺序排列的混淆矩阵。

```python
from evaluate import score_predictions

scores = score_predictions(y_test, predictions, labels)
print(scores["accuracy"], scores["macro_f1"])
```

张帅宇可按第 1 章原清单提取 SIFT。杨晓芳可复用 `classify_color.py` 的 `training_folds(y_train)`、`C_VALUES`、`SVC_SETTINGS` 和共用评分函数，使训练验证与分类器设置一致。BoVW 词典须在每折训练样本的描述子上拟合，再编码该折验证图像。正式分类器和词典最终用全量训练集拟合，测试集用于所选配置的最终评估。

## 第 5 章实验与交接

从仓库根目录运行以下命令。已有第 4 章描述子缓存时使用第一条。只有解包后的原图和划分清单时，使用第二条按第 4 章设置提取并保存截取后的缓存。

```bash
python code/classify_bovw.py
python code/classify_bovw.py --extract-missing --sift-workers 8
```

`code/results/bovw_classification/` 包含汇总指标、折内搜索、分阶段耗时、逐类指标、混淆矩阵和图 5-1 至图 5-4。第 6 章可读取 `summary.csv`、`config.json` 及两份 `{dataset}_predictions.csv`。预测按测试清单顺序保存，列名为 `sample_id,y_true,y_pred`，Food-101 六类和 Imagenette 分别有 1,500 行和 3,925 行。
