# Kaggle-Optiver: 收盘交易竞赛解决方案

![image](https://github.com/user-attachments/assets/6c4ede83-9923-4998-bd2b-6c94cc8f440a)


## 竞赛概览
### 背景
本仓库包含[Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close)竞赛的解决方案。该挑战聚焦于预测纳斯达克10分钟收盘拍卖期间的短期价格波动，这一关键过程占据了日均交易量的近10%。

核心概念：
- ​**收盘交叉拍卖**：纳斯达克确定官方收盘价的价格发现机制
- ​**订单簿**：连续交易与拍卖订单簿的结合
- ​**价格确定**：最大化匹配量同时考虑失衡的"解交叉"价格

### 竞赛任务
- 使用以下数据预测股价变动（`target`）：
  - 订单簿数据（买卖报价及量）
  - 拍卖信息（失衡量、参考价等）
  - 市场指标（加权均价、近/远价格）
- 评估指标：平均绝对误差（MAE）

## 解决方案架构
### 特征工程
核心特征：
1. ​**市场动态**
   - 失衡比率：`imbalance_size / matched_size`
   - 买卖价差：`ask_price - bid_price`
   - 中间价：`(ask_price + bid_price)/2`

2. ​**价格关系**
   - 6个关键价格指标的两两差值
   - 基于最大-最小-中间值的三价交互关系

3. ​**历史特征**
   - 股票特定统计量（买卖量中位数、标准差、极值）
   - 当前成交量与历史基准比较

4. ​**市场状态**
   - 高量标志：`bid_plus_ask_sizes > stock_median`
   - 买卖量差异：`ask_size - bid_size`

### 模型实现
- ​**LightGBM**梯度提升模型（4折交叉验证）
- 关键参数：
  ```python
  {
    'learning_rate': 0.018,
    'max_depth': 9,
    'n_estimators': 600,
    'num_leaves': 440,
    'objective': 'mae',
    'reg_alpha': 0.01,
    'reg_lambda': 0.01
  }
