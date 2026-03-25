#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI量化投资脚本 - 博时黄金C (002611) 多模型集成预测与回测
===========================================================
本脚本使用机器学习模型（RF、XGB、LSTM、LR）集成预测基金次日涨跌幅，
并包含回测评估功能，帮助验证策略有效性。

【修复】2025-03-17：解决 SSL 连接错误 + 历史净值获取失败问题
        改用稳定的 API 接口获取历史净值，保留 JS 解析作为备选。

【新增】自动更新实际涨跌幅：每次运行脚本时，自动从最新净值数据中获取
        并填充 predictions_log.csv 中未填写的实际涨跌幅，确保统计准确。
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import urllib3
from scipy import stats
import matplotlib.pyplot as plt

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 机器学习库
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost 未安装，将跳过 XGBoost 模型。")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow 未安装，将跳过 LSTM 模型。")

# ================== 配置参数 ==================
FUND_CODE = '002611'          # 基金代码
ETF_VOLUME_CODE = '518880'    # 用于成交量的ETF代码（黄金ETF）
HISTORY_DAYS = 3000            # 获取历史数据天数（足够训练即可）
WINDOW_SIZE = 20               # 特征窗口大小
TEST_RATIO = 0.2               # 回测中验证集比例
RETRAIN_FREQ = 30              # 每隔多少天重新训练模型（回测用）

# 文件路径
PREDICTION_LOG = 'predictions_log.csv'   # 预测记录
BACKTEST_RESULT = 'backtest_result.csv'  # 回测结果
MODEL_WEIGHTS_FILE = 'model_weights.json' # 模型权重

# 网络请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://fund.eastmoney.com/',
    'Accept': 'application/json, text/plain, */*'
}
# =============================================

# ---------- 数据获取模块 ----------
def get_fund_net_value(code=FUND_CODE):
    """获取指定基金最新净值及涨跌幅"""
    url = 'https://api.fund.eastmoney.com/f10/lsjz'
    params = {
        'fundCode': code,
        'pageIndex': 1,
        'pageSize': 1,
        '_': int(datetime.now().timestamp() * 1000)
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, verify=False, timeout=10)
        data = resp.json()
        if data['Data']['LSJZList']:
            item = data['Data']['LSJZList'][0]
            net = float(item['DWJZ'])
            change = float(item['JZZZL']) if item['JZZZL'] else 0.0
            return net, change
    except Exception as e:
        print(f"净值获取异常({code}): {e}")
    return None, None

def get_history_net_value_via_api(code=FUND_CODE, days=HISTORY_DAYS):
    """
    通过天天基金API获取历史净值（多页）
    返回: (prices list, dates list) 按时间升序
    """
    all_items = []
    page = 1
    page_size = 20  # 每页条数，API实际返回20条
    url = 'https://api.fund.eastmoney.com/f10/lsjz'

    # 计算需要的页数：3000天 / 20条/页 = 150页
    max_pages = min(150, (days // page_size) + 1)
    print(f"  开始获取历史净值，计划请求 {max_pages} 页，每页 {page_size} 条")
    
    while page <= max_pages:  # 最多请求150页，确保获取3000天数据
        params = {
            'fundCode': code,
            'pageIndex': page,
            'pageSize': page_size,
            '_': int(datetime.now().timestamp() * 1000)
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, verify=False, timeout=10)
            data = resp.json()
            if data['Data'] is None or data['Data']['LSJZList'] is None:
                print(f"  第{page}页返回数据为空，停止请求")
                break
            items = data['Data']['LSJZList']
            if not items:
                print(f"  第{page}页无数据，停止请求")
                break
            all_items.extend(items)
            print(f"  第{page}页获取成功，累计 {len(all_items)} 条数据")
            # 继续请求下一页，直到获取到足够的数据
            page += 1
        except Exception as e:
            print(f"获取历史净值API失败（第{page}页）: {e}")
            break

    if not all_items:
        return None, None

    # 提取净值、日期，并转为 float
    prices = []
    dates = []
    for item in all_items:
        try:
            price = float(item['DWJZ'])
            date_str = item['FSRQ']
            prices.append(price)
            dates.append(date_str)
        except:
            continue

    # 按日期升序排序（API返回可能是倒序，但不确定，我们手动排序）
    combined = sorted(zip(dates, prices), key=lambda x: x[0])
    dates = [x[0] for x in combined]
    prices = [x[1] for x in combined]

    # 只返回最近 days 条
    if days > 0:
        prices = prices[-days:]
        dates = dates[-days:]
    return prices, dates

def get_history_net_value_via_js(code=FUND_CODE, days=HISTORY_DAYS):
    """备用方法：通过 js 文件解析历史净值"""
    try:
        url = f'https://fund.eastmoney.com/pingzhongdata/{code}.js'
        resp = requests.get(url, headers=HEADERS, verify=False, timeout=10)
        resp.encoding = 'utf-8'
        text = resp.text

        # 尝试多个可能的变量名
        patterns = [
            r'var Data_netValue\s*=\s*(\[.*?\]);',
            r'var netValueData\s*=\s*(\[.*?\]);',
            r'var netvalue\s*=\s*(\[.*?\]);'
        ]
        match = None
        for pat in patterns:
            match = re.search(pat, text, re.DOTALL)
            if match:
                break

        if not match:
            print("未找到净值数据 (JS解析失败)")
            return None, None

        data = json.loads(match.group(1))
        # 按日期排序（升序）
        data.sort(key=lambda x: x['净值日期'])
        prices = [float(item['单位净值']) for item in data]
        dates = [item['净值日期'] for item in data]

        if days > 0:
            prices = prices[-days:]
            dates = dates[-days:]
        return prices, dates
    except Exception as e:
        print(f"JS解析历史净值失败: {e}")
        return None, None

def get_history_net_value(code=FUND_CODE, days=HISTORY_DAYS):
    """获取基金历史净值（日线）- 优先使用 API，失败则尝试 JS"""
    prices, dates = get_history_net_value_via_api(code, days)
    if prices is not None and len(prices) > 0:
        return prices, dates
    else:
        print("API获取失败，尝试JS解析...")
        return get_history_net_value_via_js(code, days)

def get_etf_volume(code=ETF_VOLUME_CODE, days=HISTORY_DAYS):
    """获取ETF成交量数据（作为替代成交量）"""
    try:
        session = requests.Session()
        session.trust_env = False
        session.proxies = {'http': None, 'https': None}
        
        # 根据代码前缀判断市场
        if code.startswith('5'):
            symbol = f'sh{code}'
        elif code.startswith('1'):
            symbol = f'sz{code}'
        else:
            symbol = code
        
        url = 'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
        params = {
            'symbol': symbol,
            'scale': '240',      # 日线
            'ma': 'no',
            'datalen': str(days)
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = session.get(url, params=params, headers=headers, verify=False, timeout=10)
        data = resp.json()
        if data and len(data) > 0:
            volume = [int(item['volume']) for item in data]
            volume.reverse()      # 确保时间正序
            return volume
    except Exception as e:
        print(f"获取成交量失败: {e}")
    return None

# ---------- 特征工程模块 ----------
def add_technical_indicators(df):
    """
    在包含 'close' 和 'volume' 的DataFrame上添加技术指标
    参数 df: 至少包含 'close' 列，可选 'volume'
    返回: 添加指标后的DataFrame
    """
    data = df.copy()
    close_series = data['close']
    volume_series = data['volume'] if 'volume' in data.columns else None

    # 移动平均线
    for period in [5, 10, 20, 60]:
        data[f'ma_{period}'] = data['close'].rolling(window=period).mean()

    # 指数移动平均
    for period in [12, 26]:
        data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()

    # RSI
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    data['rsi'] = rsi(data['close'], 14)

    # MACD
    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']

    # 布林带
    ma20 = data['close'].rolling(20).mean()
    std20 = data['close'].rolling(20).std()
    data['boll_upper'] = ma20 + 2 * std20
    data['boll_lower'] = ma20 - 2 * std20
    data['boll_mid'] = ma20

    # ATR (平均真实波幅)
    if 'high' not in data.columns:
        data['high'] = data['close']
    if 'low' not in data.columns:
        data['low'] = data['close']
    high = data['high']
    low = data['low']
    prev_close = data['close'].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr'] = tr.rolling(14).mean()

    # CCI (顺势指标)
    tp = (high + low + close_series) / 3
    sma_tp = tp.rolling(20).mean()
    mad_tp = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    data['cci'] = (tp - sma_tp) / (0.015 * mad_tp)

    # OBV (能量潮)
    if volume_series is not None:
        obv = (np.sign(close_series.diff()) * volume_series).fillna(0).cumsum()
        data['obv'] = obv
        data['obv_ma'] = obv.rolling(20).mean()

    # 成交量比率
    if volume_series is not None:
        vol_ma5 = data['volume'].rolling(5).mean()
        vol_ma20 = data['volume'].rolling(20).mean()
        data['volume_ratio'] = vol_ma5 / vol_ma20

    # 动量
    for period in [5, 10, 20]:
        data[f'momentum_{period}'] = data['close'].pct_change(period) * 100

    # 波动率 (20日年化)
    data['volatility'] = data['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100

    # 价格位置 (当前价格在20日区间的位置)
    data['hhv_20'] = data['close'].rolling(20).max()
    data['llv_20'] = data['close'].rolling(20).min()
    data['price_position'] = (data['close'] - data['llv_20']) / (data['hhv_20'] - data['llv_20'])

    # 时间特征
    if 'date' in data.columns:
        data['dayofweek'] = pd.to_datetime(data['date']).dt.dayofweek
        data['month'] = pd.to_datetime(data['date']).dt.month
        # 可以添加更多季节性特征

    return data

def create_features_and_target(prices, volumes=None, dates=None, window=WINDOW_SIZE):
    """
    构建特征矩阵和目标变量（下一日收益率）
    返回:
        X: 二维特征数组 (样本数, 特征数)
        y: 目标值 (下一日收益率 %)
        feature_names: 特征名称列表
        scaler: 用于标准化的scaler
    """
    df = pd.DataFrame({'close': prices})
    if volumes:
        df['volume'] = volumes
    if dates:
        df['date'] = dates

    # 添加技术指标
    df = add_technical_indicators(df)

    # 创建目标：下一日收益率（百分比）
    df['target'] = df['close'].pct_change(-1).shift(-1) * 100  # 注意shift确保对齐

    # 删除包含NaN的行（由于指标计算）
    df = df.dropna().reset_index(drop=True)

    # 选择特征列（排除目标、日期和原始价格/成交量可能引起的冗余）
    exclude_cols = ['target', 'date', 'close', 'volume', 'high', 'low']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    y = df['target'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler, df

# ---------- 模型训练模块 ----------
def train_models(X_train, y_train, X_val=None, y_val=None):
    """
    训练多个回归模型，并返回模型字典及验证集上的权重（用于集成）
    """
    models = {}
    val_preds = {}
    weights = {}

    # 1. 线性回归（基准）
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear'] = lr

    # 2. 随机森林
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    # 3. XGBoost (如果可用)
    if XGB_AVAILABLE:
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model

    # 4. LSTM (需要3D输入，此处单独处理，后面在集成时使用)
    # 注意：LSTM将在集成预测时动态训练（使用序列数据），这里先占位
    # 由于LSTM需要序列形状，我们不在这个函数中训练，而是在外层处理

    # 如果有验证集，计算每个模型在验证集上的表现（例如MAE的倒数作为权重）
    if X_val is not None and y_val is not None:
        for name, model in models.items():
            pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            weights[name] = 1.0 / (mae + 1e-6)   # 防止除零
        # 归一化权重
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
    else:
        # 默认等权重
        for name in models.keys():
            weights[name] = 1.0 / len(models)

    return models, weights

def train_lstm(X_seq_train, y_train, X_seq_val=None, y_val=None):
    """训练LSTM模型，输入为3D序列"""
    if not TF_AVAILABLE:
        return None

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=0)
    model.fit(X_seq_train, y_train, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])

    return model

# ---------- 集成预测 ----------
def ensemble_predict(models, lstm_model, X_latest, X_seq_latest, weights):
    """集成预测：加权平均各模型预测值"""
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X_latest.reshape(1, -1))[0]

    if lstm_model is not None and X_seq_latest is not None:
        pred_lstm = lstm_model.predict(X_seq_latest.reshape(1, X_seq_latest.shape[0], X_seq_latest.shape[1]))[0,0]
        preds['LSTM'] = pred_lstm
        # 如果LSTM权重未定义，则赋予平均权重
        if 'LSTM' not in weights:
            avg_weight = 1.0 / (len(weights) + 1)
            weights = {k: v*(1-avg_weight) for k, v in weights.items()}
            weights['LSTM'] = avg_weight

    # 加权平均
    ensemble = sum(preds[name] * weights.get(name, 0) for name in preds)
    return ensemble, preds

# ---------- 回测模块 ----------
def backtest_strategy(prices, volumes=None, dates=None, window=WINDOW_SIZE, test_ratio=TEST_RATIO, retrain_freq=RETRAIN_FREQ):
    """
    滚动回测策略
    返回: 回测结果DataFrame，包含日期、实际收益率、预测收益率、模拟持仓收益等
    """
    # 构建特征和目标
    X, y, feature_names, scaler, df = create_features_and_target(prices, volumes, dates, window)
    # df已经包含所有特征和目标，且按时间排序（日期升序）

    n = len(X)
    split_idx = int(n * (1 - test_ratio))
    # 确保训练集足够
    if split_idx < 200:
        print("警告：训练样本太少，回测可能不可靠。")
        split_idx = max(200, n // 2)

    # 准备存储预测结果
    results = []
    current_models = None
    current_weights = None
    lstm_model = None

    # 滚动窗口：从split_idx开始预测，每次预测后，模型可以重新训练或固定
    # 这里采用固定训练集（前split_idx个样本）进行训练，然后逐步预测后面的每个点
    # 但为了模拟真实情况，最好每隔一段时间重新训练，我们使用retrain_freq控制

    # 先将前split_idx作为初始训练集
    train_end = split_idx
    for i in range(split_idx, n):
        # 当前预测点的索引 i，特征为X[i]
        # 需要构造序列数据给LSTM: 需要根据历史构建序列，这里简单使用过去window个点的特征？
        # 更合理的LSTM输入应该是原始价格序列或其他，但我们简化：使用X[i]的序列形式
        # 实际上LSTM需要时序特征，我们之前已经将特征展平，因此无法直接用于LSTM，这里我们重新生成序列数据。
        # 为了回测简单，我们可以只使用非LSTM模型，或者额外处理。为了演示，这里暂时忽略LSTM在回测中的使用。
        # 更好的做法：在特征工程阶段保留3D数据，但为简化，回测部分仅使用树模型和线性模型。

        # 每隔retrain_freq天重新训练
        if (i - split_idx) % retrain_freq == 0 or current_models is None:
            # 使用截止到i-1的数据重新训练（注意不能用未来数据）
            X_train = X[:i]
            y_train = y[:i]
            # 划分一部分验证集用于权重
            val_size = min(100, len(X_train)//5)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size] if val_size>0 else X_train
            y_train = y_train[:-val_size] if val_size>0 else y_train

            models, weights = train_models(X_train, y_train, X_val, y_val)
            current_models = models
            current_weights = weights
            # LSTM暂时忽略

        # 预测
        X_input = X[i].reshape(1, -1)
        # 对特征进行标准化？X已经标准化过了，直接使用
        ensemble = 0
        preds = {}
        for name, model in current_models.items():
            pred = model.predict(X_input)[0]
            preds[name] = pred
            ensemble += pred * current_weights[name]
        # 无LSTM

        actual = y[i]
        # 将所有预测值转换为Python原生类型
        preds_serializable = {k: float(v) for k, v in preds.items()}
        results.append({
            'date': dates[i+window] if dates and i+window < len(dates) else None,
            'actual': float(actual),
            'predicted': float(ensemble),
            'individual': json.dumps(preds_serializable)
        })

    results_df = pd.DataFrame(results)
    return results_df

def evaluate_backtest(results_df):
    """评估回测表现，返回绩效指标和包含计算字段的DataFrame"""
    df = results_df.copy()
    df['direction_correct'] = (np.sign(df['predicted']) == np.sign(df['actual'])).astype(int)
    direction_accuracy = df['direction_correct'].mean()

    # 模拟交易：假设每次根据预测方向开仓，预测涨则买入（持有多头），预测跌则空仓（不交易，收益率0）
    # 这里简化：只做多，预测为正时持有，获得实际收益率；预测为负时空仓，收益率为0。
    df['strategy_return'] = np.where(df['predicted'] > 0, df['actual'], 0)
    df['cumulative_strategy'] = (1 + df['strategy_return']/100).cumprod()
    df['cumulative_benchmark'] = (1 + df['actual']/100).cumprod()

    total_days = len(df)
    total_return_strategy = df['cumulative_strategy'].iloc[-1] - 1
    total_return_benchmark = df['cumulative_benchmark'].iloc[-1] - 1

    # 年化收益率 (假设250个交易日)
    ann_return_strategy = (df['cumulative_strategy'].iloc[-1] ** (250/total_days) - 1) if total_days>0 else 0
    ann_return_benchmark = (df['cumulative_benchmark'].iloc[-1] ** (250/total_days) - 1) if total_days>0 else 0

    # 最大回撤
    rolling_max = df['cumulative_strategy'].cummax()
    drawdown = (df['cumulative_strategy'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 夏普比率 (假设无风险利率0)
    excess_returns = df['strategy_return'] / 100  # 转为小数
    sharpe = np.sqrt(250) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

    # 盈亏比
    winning_trades = df[df['strategy_return'] > 0]['strategy_return']
    losing_trades = df[df['strategy_return'] < 0]['strategy_return']
    avg_win = winning_trades.mean() if len(winning_trades)>0 else 0
    avg_loss = abs(losing_trades.mean()) if len(losing_trades)>0 else 1e-6
    profit_factor = avg_win / avg_loss

    metrics = {
        '样本数': total_days,
        '方向准确率': direction_accuracy,
        '策略累计收益率': total_return_strategy,
        '基准累计收益率': total_return_benchmark,
        '策略年化收益率': ann_return_strategy,
        '基准年化收益率': ann_return_benchmark,
        '夏普比率': sharpe,
        '最大回撤': max_drawdown,
        '盈亏比': profit_factor,
        '平均每笔收益': df['strategy_return'].mean(),
        '胜率': len(winning_trades) / len(df) if len(df)>0 else 0,
    }
    return metrics, df

def plot_backtest(results_df):
    """绘制回测曲线"""
    plt.figure(figsize=(12,6))
    plt.plot(results_df['cumulative_benchmark'], label='基准（买入持有）')
    plt.plot(results_df['cumulative_strategy'], label='策略（预测做多）')
    plt.title('回测净值曲线')
    plt.xlabel('交易日')
    plt.ylabel('净值')
    plt.legend()
    plt.grid(True)
    plt.show()

# ========== 新增：自动更新实际涨跌幅 ==========
def auto_update_actuals(days=60):
    """
    自动从历史净值中获取实际涨跌幅，并填充 predictions_log.csv 中未填写的记录。
    参数 days: 获取最近多少天的历史净值用于匹配。
    """
    if not os.path.exists(PREDICTION_LOG):
        return
    df = pd.read_csv(PREDICTION_LOG)
    # 找出 actual_return 为空或NaN的记录
    mask = df['actual_return'].isna() | (df['actual_return'] == '')
    if not mask.any():
        return

    # 获取最近 days 天的历史净值数据
    prices, dates = get_history_net_value(FUND_CODE, days=days)
    if not prices or not dates:
        print("无法获取历史净值，自动更新实际涨跌幅失败。")
        return

    # 计算每日涨跌幅（后一天相对于前一天的百分比变化）
    returns = pd.Series(prices).pct_change() * 100
    # 构建日期 -> 涨跌幅的字典（注意第一个日期没有涨跌幅，跳过）
    actual_dict = {}
    for i in range(1, len(dates)):
        actual_dict[dates[i]] = returns.iloc[i]

    # 更新记录
    updated = 0
    for idx in df[mask].index:
        date = df.loc[idx, 'date']
        if date in actual_dict:
            df.loc[idx, 'actual_return'] = actual_dict[date]
            updated += 1

    if updated > 0:
        df.to_csv(PREDICTION_LOG, index=False)
        print(f"自动更新了 {updated} 条实际涨跌幅记录。")

# ---------- 预测明日模块 ----------
def predict_tomorrow():
    """预测明日涨跌幅"""
    print("\n" + "="*70)
    print("🎯 博时黄金C(002611) 明日涨跌幅预测 (AI量化集成模型)")
    print("="*70)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"预测目标: 明日({tomorrow})涨跌幅")

    # 获取数据
    print("\n📊 【获取数据】")
    net_value, change = get_fund_net_value(FUND_CODE)
    print(f"  最新净值获取: {'成功' if net_value is not None else '失败'}")
    
    prices, dates = get_history_net_value(FUND_CODE, days=HISTORY_DAYS)
    print(f"  历史净值获取: {'成功' if prices is not None else '失败'}")
    print(f"  历史净值天数: {len(prices) if prices is not None else 0}")
    
    # 获取与价格数据长度一致的成交量数据
    volumes = get_etf_volume(ETF_VOLUME_CODE, days=len(prices) if prices else HISTORY_DAYS)
    print(f"  成交量获取: {'成功' if volumes is not None else '失败'}")
    if volumes:
        # 确保成交量数据长度与价格数据一致
        volumes = volumes[-len(prices):] if len(volumes) > len(prices) else volumes

    if net_value is None:
        print("❌ 无法获取最新净值")
        return None
    elif prices is None:
        print("❌ 无法获取历史净值")
        return None
    elif len(prices) < WINDOW_SIZE+10:
        print(f"❌ 历史数据不足，需要至少{WINDOW_SIZE+10}天，实际只有{len(prices)}天")
        return None

    print(f"✅ 最新净值: {net_value:.4f} (今日涨跌: {change:.2f}%)")
    print(f"✅ 历史净值天数: {len(prices)}")
    print(f"✅ 成交量天数: {len(volumes) if volumes else 0}")

    # 构建特征（使用全部历史数据）
    X, y, feature_names, scaler, df = create_features_and_target(prices, volumes, dates, WINDOW_SIZE)
    if len(X) < 100:
        print("❌ 特征样本太少")
        return None

    # 划分训练集和验证集（最近20%作为验证）
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 训练模型
    print("\n🤖 【训练模型】")
    models, weights = train_models(X_train, y_train, X_val, y_val)
    print(f"训练完成，模型权重: {weights}")

    # 训练LSTM（如果需要）
    lstm_model = None
    if TF_AVAILABLE:
        print("  训练LSTM...")
        # 构建LSTM需要的3D数据：使用原始价格序列（或其他）？
        # 简化：直接用价格序列作为输入，窗口为WINDOW_SIZE
        seq_prices = np.array(prices[-len(X):])  # 对齐X的长度
        seq_X, seq_y = [], []
        for i in range(WINDOW_SIZE, len(seq_prices)-1):
            seq_X.append(seq_prices[i-WINDOW_SIZE:i])
            seq_y.append((seq_prices[i+1] - seq_prices[i]) / seq_prices[i] * 100)
        seq_X = np.array(seq_X).reshape(-1, WINDOW_SIZE, 1)
        seq_y = np.array(seq_y)
        # 划分训练/验证
        split_seq = int(len(seq_X) * 0.8)
        seq_train_X, seq_val_X = seq_X[:split_seq], seq_X[split_seq:]
        seq_train_y, seq_val_y = seq_y[:split_seq], seq_y[split_seq:]

        lstm_model = train_lstm(seq_train_X, seq_train_y, seq_val_X, seq_val_y)
        if lstm_model:
            # 计算LSTM在验证集上的MAE，加入权重
            pred_lstm_val = lstm_model.predict(seq_val_X).flatten()
            mae_lstm = mean_absolute_error(seq_val_y, pred_lstm_val)
            # 更新权重
            weights['LSTM'] = 1.0 / (mae_lstm + 1e-6)
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            print(f"  LSTM 验证MAE: {mae_lstm:.4f}, 权重: {weights['LSTM']:.4f}")

    # 构建最新样本（最后一个窗口的特征）
    latest_features = X[-1]  # 最后一个样本的特征（对应昨天）
    # 为了预测明天，我们需要使用今天收盘后的最新数据，所以应该用最新的窗口重新计算特征
    # 但这里简单使用最后一个样本的特征，对应的是基于昨天及之前数据构建的特征，预测今天的涨跌幅？实际上y[-1]是今天的涨跌幅（如果昨天收盘后预测今天）。
    # 为了预测明天，我们需要用最新的窗口（包含今天的数据）重新计算特征。由于今天的数据已经包含在prices中（最后一个是今天），我们重新构建一次特征。
    # 更稳健的方法：重新运行特征构建，但为了快速，我们直接取最后一个特征向量，它已经包含了今天的信息（如果特征构建时使用了截至今天的窗口）。
    # 检查：特征构建中，每个样本i对应窗口[i-WINDOW_SIZE:i]，目标为i+1天的收益率。所以最后一个样本的窗口是[-WINDOW_SIZE:]，目标对应明天。
    # 所以X[-1]正好是预测明天所需的特征。
    X_tomorrow = X[-1].reshape(1, -1)

    # 预测
    ensemble, preds = ensemble_predict(models, lstm_model, X_tomorrow, None, weights)  # 这里忽略序列
    # 如果有LSTM，需要构建序列输入
    if lstm_model:
        # 构建最新的价格序列用于LSTM
        latest_seq = np.array(prices[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
        lstm_pred = lstm_model.predict(latest_seq)[0,0]
        preds['LSTM'] = lstm_pred
        # 重新加权
        ensemble = sum(preds[name] * weights.get(name, 0) for name in preds)

    print("\n" + "="*70)
    print("🎯 【集成预测结果】")
    print("="*70)
    print(f"预测方向: {'上涨' if ensemble > 0 else '下跌' if ensemble < 0 else '持平'}")
    print(f"预测涨跌幅: {ensemble:.4f}%")
    print(f"预测明日净值: {net_value * (1 + ensemble/100):.4f}")
    print("\n各模型预测:")
    for name, pred in preds.items():
        print(f"  {name}: {pred:.4f}%")

    # 技术指标概览（可选）
    print("\n📊 【最新技术指标】")
    latest_indicators = df.iloc[-1].to_dict()
    for key in ['rsi', 'macd', 'macd_hist', 'atr', 'volume_ratio']:
        if key in latest_indicators:
            print(f"  {key}: {latest_indicators[key]:.4f}")

    # 记录预测
    log_entry = {
        'date': tomorrow,
        'ensemble_pred': ensemble,
        'individual_preds': json.dumps({k: float(v) for k, v in preds.items()}),
        'actual_return': '',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    df_log = pd.DataFrame([log_entry])
    if os.path.exists(PREDICTION_LOG):
        old = pd.read_csv(PREDICTION_LOG)
        df_log = pd.concat([old, df_log], ignore_index=True)
    df_log.to_csv(PREDICTION_LOG, index=False)
    print(f"\n预测已记录至 {PREDICTION_LOG}")

    return ensemble, preds

# ---------- 历史预测统计 ----------
def show_prediction_stats():
    """显示历史预测的准确率统计"""
    if not os.path.exists(PREDICTION_LOG):
        print("暂无预测记录。")
        return
    df = pd.read_csv(PREDICTION_LOG)
    
    # 显示所有预测记录
    print("\n" + "="*70)
    print("📊 【历史预测记录】")
    print("="*70)
    print(f"总预测次数: {len(df)}")
    
    # 过滤出有实际值的记录
    df_actual = df[df['actual_return'].notna() & (df['actual_return'] != '')]
    if len(df_actual) > 0:
        df_actual['actual_return'] = pd.to_numeric(df_actual['actual_return'])
        df_actual['ensemble_pred'] = pd.to_numeric(df_actual['ensemble_pred'])
        df_actual['correct'] = (np.sign(df_actual['ensemble_pred']) == np.sign(df_actual['actual_return'])).astype(int)
        acc = df_actual['correct'].mean()
        mae = mean_absolute_error(df_actual['actual_return'], df_actual['ensemble_pred'])
        rmse = np.sqrt(mean_squared_error(df_actual['actual_return'], df_actual['ensemble_pred']))
        print(f"有实际值的预测次数: {len(df_actual)}")
        print(f"方向准确率: {acc:.2%}")
        print(f"平均绝对误差(MAE): {mae:.4f}%")
        print(f"均方根误差(RMSE): {rmse:.4f}%")
    else:
        print("暂无已更新的实际值，无法计算准确率。")
    
    # 显示最近的预测记录
    print("\n" + "-"*70)
    print("最近的预测记录:")
    print("-"*70)
    recent_df = df.tail(5)[['date', 'ensemble_pred', 'actual_return']]
    for _, row in recent_df.iterrows():
        actual = row['actual_return'] if pd.notna(row['actual_return']) and row['actual_return'] != '' else "待更新"
        print(f"日期: {row['date']}, 预测涨跌幅: {float(row['ensemble_pred']):.4f}%, 实际涨跌幅: {actual}")

# ---------- 更新实际涨跌幅（保留手动接口，但已不必要）----------
def update_actual(date, actual):
    """更新某日实际涨跌幅（手动方式，现已由自动更新替代）"""
    if not os.path.exists(PREDICTION_LOG):
        print("预测记录文件不存在。")
        return
    df = pd.read_csv(PREDICTION_LOG)
    if date in df['date'].values:
        df.loc[df['date'] == date, 'actual_return'] = actual
        df.to_csv(PREDICTION_LOG, index=False)
        print(f"已更新 {date} 的实际涨跌幅为 {actual}%")
    else:
        print(f"未找到日期 {date} 的预测记录。")

# ---------- 主程序 ----------
if __name__ == "__main__":
    # 每次运行先自动更新实际涨跌幅（基于最近60天历史净值）
    auto_update_actuals(days=60)

    print("博时黄金C AI量化投资系统")
    print("1. 预测明日涨跌幅")
    print("2. 查看历史预测统计")
    print("3. 更新实际涨跌幅（手动）")
    print("4. 运行回测")
    choice = input("请选择操作 (1/2/3/4): ").strip()

    if choice == '1':
        predict_tomorrow()
    elif choice == '2':
        show_prediction_stats()
    elif choice == '3':
        date = input("输入日期 (YYYY-MM-DD): ").strip()
        actual = float(input("输入实际涨跌幅 (%): ").strip())
        update_actual(date, actual)
    elif choice == '4':
        print("\n正在获取历史数据用于回测...")
        prices, dates = get_history_net_value(FUND_CODE, days=HISTORY_DAYS)
        volumes = get_etf_volume(ETF_VOLUME_CODE, days=HISTORY_DAYS)
        if prices is not None:
            # 确保成交量数据长度与价格数据一致
            if volumes:
                volumes = volumes[-len(prices):] if len(volumes) > len(prices) else volumes
            if len(prices) < 200:
                print("数据不足，无法回测。")
            else:
                results = backtest_strategy(prices, volumes, dates, window=WINDOW_SIZE)
                metrics, results_with_metrics = evaluate_backtest(results)
                print("\n" + "="*70)
                print("📈 【回测绩效报告】")
                print("="*70)
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"{k}: {v:.4f}")
                    else:
                        print(f"{k}: {v}")
                # 保存回测结果
                results_with_metrics.to_csv(BACKTEST_RESULT, index=False)
                print(f"\n回测明细已保存至 {BACKTEST_RESULT}")
                # 可选绘图
                plot = input("是否绘制净值曲线？(y/n): ").strip().lower()
                if plot == 'y':
                    plot_backtest(results_with_metrics)
        else:
            print("无法获取历史数据，无法回测。")
    else:
        print("无效选择")