# 泰坦尼克号生存预测分析

本项目是国际课程《生物统计学：应用于流行病学与生物医药的数据科学》的结课项目。基于 Kaggle 经典竞赛“泰坦尼克号：机器学习从灾难中学习”，通过数据清洗、特征工程与多种机器学习模型的对比，预测乘客的生存情况。项目最终获得 **89 分（A）**。


## 🎯 项目目标

- 对泰坦尼克号乘客数据进行探索性分析
- 处理缺失值并进行特征工程
- 训练并对比多种分类模型
- 选择最优模型对测试集进行预测并生成提交文件

## 📊 数据集描述

训练集包含 891 条记录，测试集包含 418 条记录。主要字段包括：

| 字段 | 说明 | 类型 |
|------|------|------|
| PassengerId | 乘客 ID | 数值 |
| Survived | 生存（1）或死亡（0） | 数值（目标） |
| Pclass | 客舱等级（1/2/3） | 数值 |
| Sex | 性别 | 字符 |
| Age | 年龄 | 数值 |
| SibSp | 兄弟姐妹/配偶数量 | 数值 |
| Parch | 父母/子女数量 | 数值 |
| Ticket | 船票号码 | 字符 |
| Fare | 票价 | 数值 |
| Cabin | 客舱号 | 字符 |
| Embarked | 登船港口 | 字符 |

## 🔧 数据处理与特征工程

1. **缺失值处理**  
   - 删除对预测贡献较小且缺失严重的字段：`Cabin`、`Name`、`Ticket`、`Embarked`、`Fare`  
   - `Age` 的缺失值用中位数填充  
   - 删除剩余含有缺失值的行

2. **特征编码**  
   - 将 `Sex` 用 `LabelEncoder` 转换为数值（0 = 女性，1 = 男性）

3. **可视化分析**  
   - 使用 Seaborn 绘制缺失值热图、性别与生存关系、年龄分布等，辅助理解数据。

## 🤖 模型训练与评估

### 使用的模型

- 逻辑回归（Logistic Regression）
- 随机森林（Random Forest）
- 梯度提升树（Gradient Boosting）
- 决策树（Decision Tree）
- K近邻（K-Nearest Neighbors）
- 朴素贝叶斯（Gaussian Naive Bayes）
- 支持向量机（SVC）

### 评估结果

| 模型 | 准确率 |
|------|--------|
| Logistic Regression | 0.8045 |
| Random Forest | **0.8315** |
| Gradient Boosting | 0.8258 |
| Decision Tree | 0.7978 |
| K-Nearest Neighbors | 0.7697 |
| Gaussian Naive Bayes | 0.7640 |
| SVC | 0.8202 |

随机森林在验证集上表现最佳，被选为最终模型。

## 📈 最终预测

使用随机森林对测试集进行预测，生成 `submission.csv` 文件，kaggle得分0.69

## 🛠️ 技术栈

- Python 3.7+
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn
