# ============================================================================
# 模板⑧：特殊数据加载/合并场景
#   A. SQLite 数据库读取 + 多表合并（202503 多标签题）
#   B. 多张 CSV 表按主键 merge 成一张宽表（202412 金融欺诈题）
# ============================================================================
import pandas as pd
import numpy as np

# ---------- A. 多表 CSV 读取 + merge（202503 多标签题） ----------
# 读四个 CSV，在 pandas 里"创建"四张表（对应原数据库的四张表，不依赖 sqlite3）
questions = pd.read_csv('Filtered_Questions.csv', encoding='utf-8-sig')   # 问题表：Id, Title, Body
answers   = pd.read_csv('Filtered_Answers.csv',  encoding='utf-8-sig')   # 答案表：Id, ParentId, Body
train_tag = pd.read_csv('Train_data.csv',        encoding='utf-8-sig')   # 训练标签表：Id, Tag
test_ids  = pd.read_csv('Test_data.csv',         encoding='utf-8-sig')   # 测试表：Id

# 训练数据：标签表 ⋈ 问题表(Id) ⋈ 答案表(ParentId=Id)，等价 SQL 的两次 LEFT JOIN
training = (train_tag.merge(questions[['Id', 'Title', 'Body']], on='Id', how='left')
                      .merge(answers[['ParentId', 'Body']].rename(columns={'Body': 'Answer'}),
                             left_on='Id', right_on='ParentId', how='left')
                      .drop(columns='ParentId'))

# 测试数据：没有标签列
test = (test_ids.merge(questions[['Id', 'Title', 'Body']], on='Id', how='left')
                 .merge(answers[['ParentId', 'Body']].rename(columns={'Body': 'Answer'}),
                        left_on='Id', right_on='ParentId', how='left')
                 .drop(columns='ParentId'))

# 因为一个 Id 有多条答案，按 Id 聚合成一行（多个答案拼成一段文本）
# ⚠️ 标签必须在聚合时一起合并，不能先 drop_duplicates 再聚，否则会丢掉多标签
def group_join(df, id_col, text_cols, tag_col=None):
    agg = df.groupby(id_col)[text_cols].agg(
        lambda s: ' '.join(x for x in s.dropna().astype(str).unique() if x.strip())
    ).reset_index()
    if tag_col and tag_col in df.columns:
        tags = df.groupby(id_col)[tag_col].apply(
            lambda s: ','.join(sorted(set(s.dropna().astype(str))))).reset_index(name=tag_col)
        agg = agg.merge(tags, on=id_col)
    return agg

training = group_join(training, 'Id', ['Title', 'Body', 'Answer'], tag_col='Tag')
test = group_join(test, 'Id', ['Title', 'Body', 'Answer'])

# 拼接成一段文本喂给文本分类模板②/③
training['text'] = training[['Title', 'Body', 'Answer']].apply(lambda x: ' '.join(x), axis=1)
test['text'] = test[['Title', 'Body', 'Answer']].apply(lambda x: ' '.join(x), axis=1)

# 标签已逗号分隔在 Tag 列，之后接 03_模板_多标签分类.py 的做法 B


# ---------- B. 多张 CSV merge 成一张宽表（202412 金融欺诈） ----------
# 思路：以交易为主线，把 交易/顾客/商家/可疑 四类表按主键 join 起来
def load_fraud(dir):
    account  = pd.read_csv(f'{dir}/Customer Profiles/account_activity.csv')
    customer = pd.read_csv(f'{dir}/Customer Profiles/customer_data.csv')
    susp     = pd.read_csv(f'{dir}/Fraudulent Patterns/suspicious_activity.csv')
    merchant = pd.read_csv(f'{dir}/Merchant Information/merchant_data.csv')
    tran_cat = pd.read_csv(f'{dir}/Merchant Information/transaction_category_labels.csv')
    tran_meta= pd.read_csv(f'{dir}/Transaction Data/transaction_metadata.csv')
    tran_rec = pd.read_csv(f'{dir}/Transaction Data/transaction_records.csv')
    train    = pd.read_csv(f'{dir}/train_data.csv')   # TransactionID, FraudIndicator

    # 顾客表：customer + account + suspicious 按 CustomerID 合并
    cust = customer.merge(account, on='CustomerID').merge(susp, on='CustomerID')
    # 交易表：train + 类别 + 元数据 + 金额记录 按 TransactionID 合并
    trans = train.merge(tran_cat, on='TransactionID') \
                 .merge(tran_meta, on='TransactionID') \
                 .merge(tran_rec, on='TransactionID')
    # 交易 + 顾客 按 CustomerID 合并成最终宽表
    data = trans.merge(cust, on='CustomerID')
    return data

# 训练/测试分别 load，然后时间特征工程 + 类别编码（接模板①）
# train_data = load_fraud('data/')
# train_data['Hour'] = pd.to_datetime(train_data['Timestamp']).dt.hour
# train_data['gap']  = (pd.to_datetime(train_data['Timestamp']) - pd.to_datetime(train_data['LastLogin'])).dt.days.abs()
