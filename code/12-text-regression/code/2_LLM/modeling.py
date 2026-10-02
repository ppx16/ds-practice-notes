import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# 1. 加载特征数据     
# ----------------------
def load_features():
    """加载特征工程生成的训练集和测试集特征"""
    train_feat = pd.read_csv("train_features.csv")
    test_feat = pd.read_csv("test_features.csv")
    return train_feat, test_feat

# ----------------------
# 2. 数据预处理（缺失值处理、特征筛选）
# ----------------------
def preprocess_features(train_feat, test_feat):
    """处理缺失值、无穷大值并分离特征与目标变量"""
    # 获取数值特征列
    numeric_cols = train_feat.columns.drop(["student_id", "article_id", "score"])
    
    # 处理无穷大值和缺失值
    for col in numeric_cols:
        # 检查并处理无穷大值
        if np.isinf(train_feat[col]).any():
            print(f"发现特征 {col} 包含无穷大值，正在处理...")
            if col == 'perplexity':
                # 对于困惑度特征，无穷大通常表示很高的困惑度，用一个大的有限值替代
                max_finite = train_feat[col][np.isfinite(train_feat[col])].max()
                if np.isnan(max_finite):  # 如果没有有限值，使用默认值
                    replacement_value = 100.0
                else:
                    replacement_value = max_finite * 2  # 使用最大有限值的2倍
                train_feat[col] = train_feat[col].replace([np.inf, -np.inf], replacement_value)
                test_feat[col] = test_feat[col].replace([np.inf, -np.inf], replacement_value)
            else:
                # 对于其他特征，用列的第99百分位数替代
                percentile_99 = np.percentile(train_feat[col][np.isfinite(train_feat[col])], 99)
                if np.isnan(percentile_99):
                    percentile_99 = 1.0  # 默认值
                train_feat[col] = train_feat[col].replace([np.inf, -np.inf], percentile_99)
                test_feat[col] = test_feat[col].replace([np.inf, -np.inf], percentile_99)
        
        # 处理缺失值（均值填充）
        if train_feat[col].isnull().any():
            mean_val = train_feat[col].mean()
            train_feat[col].fillna(mean_val, inplace=True)
            test_feat[col].fillna(mean_val, inplace=True)
    
    # 分离特征（X）和目标变量（y）
    X_train = train_feat[numeric_cols]
    y_train = train_feat["score"]
    X_test = test_feat[numeric_cols]
    
    # 保存ID信息（用于最终结果拼接）
    test_ids = test_feat[["student_id", "article_id"]]
    
    # 验证处理结果
    print("数据预处理完成:")
    print(f"  训练集特征形状: {X_train.shape}")
    print(f"  测试集特征形状: {X_test.shape}")
    print(f"  是否还有无穷大值: {np.isinf(X_train).any().any() or np.isinf(X_test).any().any()}")
    print(f"  是否还有缺失值: {X_train.isnull().any().any() or X_test.isnull().any().any()}")
    
    return X_train, y_train, X_test, test_ids

# ----------------------
# 3. 模型定义（机器学习+简单深度学习模型）
# ----------------------
class MLModels:
    @staticmethod
    def get_lightgbm():
        """LightGBM回归模型（适合表格数据，效率高）"""
        return lgb.LGBMRegressor(
            objective="regression",
            metric="mse",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            min_child_weight=0.001,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=1,  # 单线程避免某些问题
            verbosity=-1  # 减少输出
        )
    
    @staticmethod
    def get_random_forest():
        """随机森林回归模型（抗过拟合）"""
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

class SimpleNN(nn.Module):
    """简单深度学习模型（适配小规模数据，避免效能问题）"""
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# ----------------------
# 4. 交叉验证（评估模型稳定性）
# ----------------------
def cross_validate_model(model, X, y, cv=5):
    """K折交叉验证，返回平均MSE"""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mse_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        model.fit(X_tr, y_tr)
        
        # 验证并计算MSE
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    
    avg_mse = np.mean(mse_scores)
    print(f"交叉验证平均MSE: {avg_mse:.4f}")
    return avg_mse, model

# ----------------------
# 5. 模型调优（超参数搜索）
# ----------------------
def tune_model(model, param_dist, X, y):
    """随机搜索调优超参数"""
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,  # 搜索迭代次数（平衡时间与效果）
        scoring="neg_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)
    
    print(f"最优参数: {search.best_params_}")
    print(f"最优交叉验证MSE: {(-search.best_score_):.4f}")
    return search.best_estimator_

# 模型超参数搜索空间
param_spaces = {
    "lightgbm": {
        "n_estimators": randint(50, 300),
        "learning_rate": uniform(0.01, 0.3),
        "max_depth": randint(3, 10),
        "num_leaves": randint(20, 150),
        "subsample": uniform(0.6, 0.4)
    },
    "random_forest": {
        "n_estimators": randint(50, 300),
        "max_depth": randint(5, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10)
    }
}

# ----------------------
# 6. 深度学习模型训练（适配PyTorch 2.4.1+cu118）
# ----------------------
def train_nn_model(X_train, y_train, X_test, epochs=50, batch_size=32):
    """训练简单神经网络模型"""
    # 转换为Tensor并适配GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    
    # 数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型、损失函数、优化器
    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train MSE: {avg_loss:.4f}")
    
    # 预测测试集
    model.eval()
    with torch.no_grad():
        y_pred_nn = model(X_test_tensor).cpu().numpy().flatten()
    
    return y_pred_nn

# ----------------------
# 7. 模型集成与最终预测
# ----------------------
def ensemble_predictions(models, X_test):
    """集成多个模型的预测结果（简单平均）"""
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    return np.mean(predictions, axis=0)

def save_results(test_ids, y_pred, filename="邓雨_results.csv"):
    """保存预测结果为要求的CSV格式"""
    result_df = pd.DataFrame({
        "student_id": test_ids["student_id"],
        "article_id": test_ids["article_id"],
        "score": y_pred
    })
    result_df.to_csv(filename, index=False)
    print(f"预测结果已保存至 {filename}")
    return result_df

# ----------------------
# 8. 主流程执行
# ----------------------
if __name__ == "__main__":
    print("=== 开始建模流程 ===")
    
    # 加载并预处理数据
    print("加载并预处理数据...")
    train_feat, test_feat = load_features()
    X_train, y_train, X_test, test_ids = preprocess_features(train_feat, test_feat)
    
    # 步骤1：训练基本模型（不进行超参数搜索）
    print("\n----- 训练LightGBM模型 -----")
    lgb_model = MLModels.get_lightgbm()
    try:
        cv_mse_lgb, lgb_model_final = cross_validate_model(lgb_model, X_train, y_train)
        lgb_success = True
    except Exception as e:
        print(f"LightGBM训练失败: {e}")
        lgb_success = False
    
    print("\n----- 训练随机森林模型 -----")
    rf_model = MLModels.get_random_forest()
    cv_mse_rf, rf_model_final = cross_validate_model(rf_model, X_train, y_train)
    
    # 步骤2：选择最佳模型
    print("\n----- 选择最佳模型 -----")
    if lgb_success and cv_mse_lgb < cv_mse_rf:
        best_model = lgb_model_final
        best_model_name = "LightGBM"
        best_mse = cv_mse_lgb
    else:
        best_model = rf_model_final
        best_model_name = "Random Forest"
        best_mse = cv_mse_rf
    
    print(f"选择的最佳模型: {best_model_name} (MSE: {best_mse:.4f})")
    
    # 步骤3：训练深度学习模型（可选）
    print("\n----- 训练简单神经网络模型 -----")
    try:
        y_pred_nn = train_nn_model(X_train, y_train, X_test, epochs=30)
        nn_success = True
    except Exception as e:
        print(f"神经网络训练失败: {e}")
        nn_success = False
    
    # 步骤4：生成最终预测
    print("\n----- 生成最终预测 -----")
    # 用最佳模型预测
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    
    # 如果有多个成功的模型，进行集成
    if lgb_success and nn_success:
        models = [lgb_model_final, rf_model_final]
        ensemble_pred = ensemble_predictions(models, X_test)
        print("使用LightGBM和随机森林集成预测")
        final_pred = ensemble_pred
    else:
        print(f"使用单一模型预测: {best_model_name}")
        final_pred = y_pred_best
    
    # 步骤5：保存结果
    print(f"\n预测完成，预测值范围: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
    save_results(test_ids, final_pred)
    print("=== 建模流程完成 ===")