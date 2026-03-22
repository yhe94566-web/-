# 导入所需的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 读取数据集
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 显示训练集的前几行
print(train.head())

# 查看训练集的基本信息
print(train.info())

# 描述统计，查看数据的分布
print(train.describe())

# 查看缺失值
print(train.isnull().sum())

# 绘制缺失值的热图
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.show()

# 数据清理函数
def clean_data(df):
    """
    清理数据：删除不必要的列并填充缺失值
    """
    # 删除不必要的列
    df.drop(['Cabin', 'Name', 'Ticket', 'Embarked', 'Fare'], axis=1, inplace=True)
    
    # 填充缺失的Age列：用中位数填充
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # 处理任何剩余的缺失值：删除包含缺失值的行
    df.dropna(inplace=True)
    
    return df

# 清理训练集和测试集
train = clean_data(train)
test = clean_data(test)

# 检查是否还有缺失值
print(train.isnull().sum())

# 特征工程：将Sex列从字符串转换为数字
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

# 查看清理后的数据
print("清理后：")
print(train.head())

# 数据分析：相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(train.corr(), annot=True, fmt='0.1f', linewidths=0.8)
plt.show()

# 将 Survived 列的数据类型转换为字符串
train['Survived'] = train['Survived'].astype(str)

# 绘制存活与性别的关系
sns.countplot(x='Sex', hue='Survived', data=train)
plt.show()

# 查看年龄的分布
sns.histplot(train['Age'], kde=True)
plt.show()

# 绘制存活情况与年龄的关系
sns.boxplot(x='Age', y='Survived', data=train)
plt.show()

# 分离特征与标签
X = train.drop('Survived', axis=1)
y = train['Survived']

# 将数据集分为训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型列表
models = [
    LogisticRegression(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC()
]

# 存储模型和准确率
accuracies = []

# 训练并评估每个模型的函数
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    训练模型并返回准确率
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 逐个评估模型
for model in models:
    evaluate_model(model, X_train, X_test, y_train, y_test)

# 比较不同模型的准确率
model_names = [model.__class__.__name__ for model in models]
accuracy_df = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
print(accuracy_df)

# 选择表现最好的模型
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

# 使用最佳模型对测试集进行预测
test_predictions = best_model.predict(test)

# 准备提交文件
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)