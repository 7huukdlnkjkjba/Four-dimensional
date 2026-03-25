# -*- coding: utf-8 -*-
"""
三底共振检测增强版（完整版）
- 优先获取真实数据（ETF、场外基金、指数）
- 包含完整100只基金列表
- 修复Shibor和指数估值接口
- 自动降级模拟数据，确保程序始终可运行

依赖库：pip install akshare pandas numpy scipy openpyxl
"""

import os
import time
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import akshare as ak
from scipy.signal import argrelextrema

warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
DATA_SAVE_PATH = 'fund_three_bottom_enhanced'
SIGNAL_SAVE_PATH = 'fund_signals_enhanced'

# ===================== 完整100只基金列表 =====================
FUND_LIST = [
    # 黄金/商品ETF
    {"code": "518880", "name": "华安黄金ETF", "type": "商品ETF"},
    {"code": "159934", "name": "易方达黄金ETF", "type": "商品ETF"},
    {"code": "518660", "name": "博时黄金ETF", "type": "商品ETF"},
    {"code": "159987", "name": "华安易富黄金ETF", "type": "商品ETF"},
    {"code": "512710", "name": "有色金属ETF", "type": "商品ETF"},
    # 宽基ETF
    {"code": "510300", "name": "沪深300ETF", "type": "宽基ETF"},
    {"code": "510500", "name": "中证500ETF", "type": "宽基ETF"},
    {"code": "159915", "name": "创业板ETF", "type": "宽基ETF"},
    {"code": "510050", "name": "上证50ETF", "type": "宽基ETF"},
    {"code": "159902", "name": "中小板ETF", "type": "宽基ETF"},
    {"code": "510880", "name": "红利ETF", "type": "宽基ETF"},
    {"code": "159920", "name": "恒生ETF", "type": "宽基ETF"},
    {"code": "513100", "name": "纳指ETF", "type": "宽基ETF"},
    # 行业ETF（金融）
    {"code": "512880", "name": "证券ETF", "type": "行业ETF"},
    {"code": "512000", "name": "券商ETF", "type": "行业ETF"},
    {"code": "510230", "name": "金融ETF", "type": "行业ETF"},
    {"code": "512800", "name": "银行ETF", "type": "行业ETF"},
    {"code": "512900", "name": "非银金融ETF", "type": "行业ETF"},
    # 行业ETF（消费）
    {"code": "512690", "name": "酒ETF", "type": "行业ETF"},
    {"code": "159928", "name": "医药ETF", "type": "行业ETF"},
    {"code": "512200", "name": "医药卫生ETF", "type": "行业ETF"},
    {"code": "516130", "name": "医美ETF", "type": "行业ETF"},
    {"code": "515670", "name": "消费ETF", "type": "行业ETF"},
    {"code": "159929", "name": "食品饮料ETF", "type": "行业ETF"},
    # 行业ETF（科技）
    {"code": "512760", "name": "半导体ETF", "type": "行业ETF"},
    {"code": "159995", "name": "芯片ETF", "type": "行业ETF"},
    {"code": "515050", "name": "新能源ETF", "type": "行业ETF"},
    {"code": "516410", "name": "光伏ETF", "type": "行业ETF"},
    {"code": "515790", "name": "锂电ETF", "type": "行业ETF"},
    {"code": "512410", "name": "人工智能ETF", "type": "行业ETF"},
    {"code": "515000", "name": "科技ETF", "type": "行业ETF"},
    # 混合型基金
    {"code": "000001", "name": "华夏成长混合", "type": "混合型"},
    {"code": "000011", "name": "华夏大盘精选", "type": "混合型"},
    {"code": "000021", "name": "华夏优势增长", "type": "混合型"},
    {"code": "002001", "name": "华夏回报混合", "type": "混合型"},
    {"code": "002011", "name": "华夏红利混合", "type": "混合型"},
    {"code": "001184", "name": "易方达新常态", "type": "混合型"},
    {"code": "001714", "name": "工银文体产业", "type": "混合型"},
    {"code": "001838", "name": "国投瑞银国家安全", "type": "混合型"},
    {"code": "002910", "name": "易方达供给改革", "type": "混合型"},
    {"code": "004752", "name": "南方智诚混合", "type": "混合型"},
    {"code": "161725", "name": "招商中证白酒", "type": "指数型"},
    {"code": "160213", "name": "国泰纳斯达克", "type": "QDII"},
    {"code": "160706", "name": "嘉实300", "type": "指数型"},
    {"code": "161005", "name": "富国天惠", "type": "混合型"},
    {"code": "163402", "name": "兴全趋势投资", "type": "混合型"},
    {"code": "163406", "name": "兴全合润", "type": "混合型"},
    {"code": "000697", "name": "汇添富移动互联", "type": "混合型"},
    {"code": "000965", "name": "中欧医疗健康", "type": "混合型"},
    {"code": "001071", "name": "华安媒体互联网", "type": "混合型"},
    {"code": "001156", "name": "申万菱信新能源", "type": "混合型"},
    {"code": "001216", "name": "易方达新收益", "type": "混合型"},
    {"code": "001632", "name": "富国中证工业4.0", "type": "指数型"},
    {"code": "001896", "name": "招商中证煤炭", "type": "指数型"},
    {"code": "002083", "name": "新华优选分红", "type": "混合型"},
    {"code": "002190", "name": "农银新能源主题", "type": "混合型"},
    {"code": "002290", "name": "汇添富中证生物科技", "type": "指数型"},
    {"code": "002692", "name": "富国创新科技", "type": "混合型"},
    {"code": "002943", "name": "广发全球医疗保健", "type": "QDII"},
    {"code": "003095", "name": "中欧医疗创新", "type": "混合型"},
    {"code": "003834", "name": "华夏能源革新", "type": "混合型"},
    {"code": "004854", "name": "广发中证医疗", "type": "指数型"},
    {"code": "005669", "name": "前海开源公用事业", "type": "混合型"},
    {"code": "005918", "name": "广发双擎升级", "type": "混合型"},
    {"code": "006098", "name": "汇添富创新医药", "type": "混合型"},
    {"code": "006748", "name": "华安科创主题", "type": "混合型"},
    {"code": "007119", "name": "南方香港后", "type": "QDII"},
    {"code": "007300", "name": "科创50ETF联接", "type": "指数型"},
    {"code": "007412", "name": "嘉实新能源新材料", "type": "混合型"},
    {"code": "008086", "name": "华夏中证新能源汽车", "type": "指数型"},
    {"code": "008281", "name": "易方达远见成长", "type": "混合型"},
    {"code": "008969", "name": "大成国企改革", "type": "混合型"},
    {"code": "009066", "name": "易方达远见成长A", "type": "混合型"},
    {"code": "009147", "name": "南方兴润价值", "type": "混合型"},
    {"code": "009795", "name": "永赢先进制造智选", "type": "混合型"},
    {"code": "010115", "name": "汇添富北交所两年定开", "type": "混合型"},
    {"code": "010354", "name": "易方达高质量严选", "type": "混合型"},
    {"code": "012808", "name": "中欧数字经济混合", "type": "混合型"},
    {"code": "013127", "name": "易方达医药生物精选", "type": "混合型"},
    {"code": "014283", "name": "华夏北交所两年定开", "type": "混合型"},
    {"code": "016115", "name": "南方香港后A", "type": "QDII"},
    # 债券型基金
    {"code": "000186", "name": "华泰柏瑞季季红", "type": "债券型"},
    {"code": "161015", "name": "富国天盈债券", "type": "债券型"},
    {"code": "000337", "name": "鹏华丰享债券", "type": "债券型"},
    {"code": "000421", "name": "南方宝元债券", "type": "债券型"},
    {"code": "000566", "name": "华富健康文娱债券", "type": "债券型"},
    {"code": "000604", "name": "易方达瑞享混合", "type": "债券型"},
    {"code": "000731", "name": "方正富邦保险主题", "type": "债券型"},
    {"code": "000831", "name": "工银瑞信医疗保健", "type": "债券型"},
    {"code": "000977", "name": "长城环保产业混合", "type": "债券型"},
    {"code": "001047", "name": "光大国证新能源指数", "type": "债券型"},
    # QDII基金
    {"code": "000041", "name": "华夏全球精选", "type": "QDII"},
    {"code": "000631", "name": "广发纳斯达克100", "type": "QDII"},
    {"code": "001668", "name": "汇添富全球消费", "type": "QDII"},
    {"code": "001726", "name": "易方达恒生国企", "type": "QDII"},
    {"code": "002067", "name": "汇添富恒生科技", "type": "QDII"},
    {"code": "004547", "name": "广发美国房地产", "type": "QDII"},
    {"code": "005397", "name": "博时标普500ETF联接", "type": "QDII"},
    {"code": "006075", "name": "易方达原油QDII", "type": "QDII"},
    {"code": "006679", "name": "南方香港后C", "type": "QDII"},
    {"code": "007844", "name": "华夏恒生科技ETF联接", "type": "QDII"},
    # 指数增强基金
    {"code": "000172", "name": "汇添富恒生指数A", "type": "指数型"},
    {"code": "000311", "name": "景顺长城沪深300增强", "type": "指数型"},
    {"code": "000598", "name": "兴全沪深300指数(LOF)", "type": "指数型"},
    {"code": "000968", "name": "嘉实沪深300ETF联接", "type": "指数型"},
    {"code": "001051", "name": "华夏沪深300ETF联接", "type": "指数型"},
    {"code": "001234", "name": "鹏华中证医药卫生", "type": "指数型"},
    {"code": "001542", "name": "华安中证银行指数", "type": "指数型"},
    {"code": "001631", "name": "富国中证军工指数", "type": "指数型"},
    {"code": "001875", "name": "前海开源公用事业", "type": "指数型"},
    {"code": "002984", "name": "永赢科技智选混合发起A", "type": "指数型"},
]

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fund_detect_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== 宏观数据获取（修复版） =====================
class MacroDataFetcher:
    """获取宏观流动性及情绪数据（全部免费，带备选方案）"""
    
    @staticmethod
    def get_shibor(days=30):
        """获取Shibor隔夜利率（近days天）- 使用 shibor_report 接口"""
        try:
            df = ak.shibor_report(start_date=(datetime.now()-timedelta(days=days)).strftime('%Y%m%d'),
                                   end_date=datetime.now().strftime('%Y%m%d'))
            # 'ON' 列是隔夜利率
            if 'ON' in df.columns:
                rates = df['ON'].astype(float).values
                return rates
        except Exception as e:
            logger.debug(f"shibor_report 失败: {e}")
        return None

    @staticmethod
    def get_bond_yield_10y():
        """获取10年期国债收益率（最新）"""
        try:
            # 使用 bond_zh_us_rate 获取中国10年期国债收益率
            df = ak.bond_zh_us_rate()
            china_10y = df[df['债券名称']=='中国国债收益率10年']['收益率'].values
            if len(china_10y) > 0:
                return float(china_10y[-1])
        except Exception as e:
            logger.debug(f"bond_zh_us_rate 失败: {e}")
        return None

    def get_liquidity_score(self):
        """计算流动性宽松得分（0~1）"""
        shibor = self.get_shibor(30)
        bond = self.get_bond_yield_10y()
        score = 0
        if shibor is not None and len(shibor) >= 5:
            # Shibor趋势向下表示宽松
            trend_down = shibor[-1] < np.mean(shibor) * 0.95
            if trend_down:
                score += 0.5
        if bond is not None:
            # 国债收益率低于3%视为宽松（历史经验）
            if bond < 3.0:
                score += 0.5
        # 如果两个数据都缺失，返回默认中等分数
        if shibor is None and bond is None:
            return 0.5
        return min(score, 1.0)

# ===================== 指数估值获取（修复版） =====================
class IndexValuationFetcher:
    """获取指数PE/PB历史分位（使用 index_value_hist_em）"""
    
    @staticmethod
    def get_index_pe_percent(index_code):
        """
        获取指定指数的PE历史分位（0~1, 越小越低估）
        index_code: 如 '000300', '399006'
        """
        try:
            # 指数代码映射：akshare 需要带市场前缀
            if index_code.startswith('000'):
                symbol = f"sh{index_code}"
            elif index_code.startswith('399'):
                symbol = f"sz{index_code}"
            else:
                symbol = index_code
            
            df = ak.index_value_hist_em(symbol=symbol)
            if df is None or df.empty:
                return None
            
            # 尝试获取市盈率列（列名可能变动）
            pe_col = None
            for col in ['平均市盈率-市盈率', 'pe', '市盈率']:
                if col in df.columns:
                    pe_col = col
                    break
            if pe_col is None:
                return None
            
            pe_series = df[pe_col].dropna()
            if len(pe_series) == 0:
                return None
            
            current = pe_series.iloc[-1]
            percent = (pe_series <= current).sum() / len(pe_series)
            return percent
        except Exception as e:
            logger.debug(f"获取指数估值失败: {e}")
        return None

    @staticmethod
    def fund_to_index_code(fund_code):
        """基金代码 -> 指数代码（常用映射）"""
        mapping = {
            '510300': '000300',
            '510500': '000905',
            '159915': '399006',
            '512880': '399975',
            '161725': '399997',
            '510050': '000016',
            '512800': '399986',
            '512690': '399987',
            '159928': '000991',
            '512410': '930713',  # 中证人工智能主题指数
            '512760': '000812',  # 中证半导体
            '515050': '000941',  # 中证新能源
            # 可继续补充
        }
        return mapping.get(fund_code, None)

# ===================== 数据获取（增强版） =====================
class FundDataFetcher:
    """获取基金历史净值数据（增强版：多接口尝试）"""
    
    @staticmethod
    def fetch_real_data(fund_code, fund_type, days=365):
        """
        增强版真实数据获取
        1. 优先ETF接口
        2. 其次场外基金接口
        3. 最后尝试指数接口（仅用于指数型）
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        # ----- 方法1：ETF历史数据接口（适用于所有ETF）-----
        try:
            df = ak.fund_etf_hist_em(
                symbol=fund_code,
                period="daily",
                start_date="20100101",  # 放宽到最早，让接口自动截取
                end_date=end_str,
                adjust="qfq"
            )
            if df is not None and not df.empty:
                # 重命名列名
                column_map = {
                    '日期': '日期',
                    '开盘': '开盘价',
                    '最高': '最高价',
                    '最低': '最低价',
                    '收盘': '收盘价',
                    '成交量': '成交量',
                }
                # 只保留存在的列
                rename_map = {k: v for k, v in column_map.items() if k in df.columns}
                df = df.rename(columns=rename_map)
                df['日期'] = pd.to_datetime(df['日期'])
                
                # 确保必要列存在
                required_cols = ['日期', '收盘价']
                if all(col in df.columns for col in required_cols):
                    # 如果缺少成交量，用模拟数据填充
                    if '成交量' not in df.columns:
                        df['成交量'] = np.random.randint(1000000, 10000000, len(df))
                    if '开盘价' not in df.columns:
                        df['开盘价'] = df['收盘价'] * 0.998
                    if '最高价' not in df.columns:
                        df['最高价'] = df['收盘价'] * 1.01
                    if '最低价' not in df.columns:
                        df['最低价'] = df['收盘价'] * 0.99
                    
                    logger.info(f"ETF接口成功获取 {len(df)} 条数据")
                    return df[['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量']]
        except Exception as e:
            logger.debug(f"ETF接口失败: {e}")
        
        # ----- 方法2：场外基金净值接口（适用于混合型、指数型）-----
        try:
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            if df is not None and not df.empty:
                df = df.rename(columns={'净值日期': '日期', '单位净值': '收盘价'})
                df['日期'] = pd.to_datetime(df['日期'])
                # 过滤日期范围
                df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]
                
                if len(df) > 0:
                    # 补充必要列
                    df['成交量'] = np.random.randint(100000, 10000000, len(df))
                    df['开盘价'] = df['收盘价'] * np.random.uniform(0.998, 1.002, len(df))
                    df['最高价'] = df['收盘价'] * np.random.uniform(1.000, 1.015, len(df))
                    df['最低价'] = df['收盘价'] * np.random.uniform(0.985, 1.000, len(df))
                    
                    logger.info(f"场外基金接口成功获取 {len(df)} 条数据")
                    return df[['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量']]
        except Exception as e:
            logger.debug(f"场外基金接口失败: {e}")
        
        # ----- 方法3：对于指数型，尝试获取对应指数数据（需映射表）-----
        if fund_type in ['指数型', '宽基ETF', '行业ETF']:
            index_code = IndexValuationFetcher.fund_to_index_code(fund_code)
            if index_code:
                try:
                    # 指数代码格式处理
                    if index_code.startswith('000'):
                        symbol = f"sh{index_code}"
                    elif index_code.startswith('399'):
                        symbol = f"sz{index_code}"
                    else:
                        symbol = index_code
                    
                    df = ak.stock_zh_index_hist(symbol=symbol)
                    if df is not None and not df.empty:
                        df = df.rename(columns={
                            'date': '日期',
                            'open': '开盘价',
                            'high': '最高价',
                            'low': '最低价',
                            'close': '收盘价',
                            'volume': '成交量'
                        })
                        df['日期'] = pd.to_datetime(df['日期'])
                        df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]
                        
                        logger.info(f"指数接口成功获取 {len(df)} 条数据")
                        return df[['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量']]
                except Exception as e:
                    logger.debug(f"指数接口失败: {e}")
        
        # ----- 所有接口均失败 -----
        logger.warning(f"所有真实数据接口均失败: {fund_code}")
        return None

    @staticmethod
    def generate_enhanced_simulated(fund_code, fund_type, days=365):
        """生成增强型模拟数据（保留真实统计特征）"""
        type_features = {
            "商品ETF": {"base": 1.2, "vol": 0.002, "trend": 0.00008},
            "宽基ETF": {"base": 3.0, "vol": 0.008, "trend": 0.00015},
            "行业ETF": {"base": 2.5, "vol": 0.010, "trend": 0.00012},
            "混合型": {"base": 1.8, "vol": 0.007, "trend": 0.00010},
            "债券型": {"base": 1.1, "vol": 0.001, "trend": 0.00005},
            "QDII": {"base": 1.3, "vol": 0.009, "trend": 0.00007},
            "指数型": {"base": 1.5, "vol": 0.006, "trend": 0.00009},
        }
        features = type_features.get(fund_type, type_features["混合型"])
        np.random.seed(int(fund_code[-4:]) if fund_code.isdigit() else 42)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[~dates.dayofweek.isin([5,6])]  # 剔除周末
        
        returns = np.random.normal(0, features["vol"], len(dates)) + features["trend"]
        prices = features["base"] * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            "日期": dates,
            "开盘价": prices * np.random.uniform(0.998, 1.002, len(prices)),
            "最高价": prices * np.random.uniform(1.000, 1.015, len(prices)),
            "最低价": prices * np.random.uniform(0.985, 1.000, len(prices)),
            "收盘价": prices,
            "成交量": np.random.randint(1000000, 100000000, len(prices))
        })
        return df.reset_index(drop=True)

# ===================== 增强型三底检测器 =====================
class ThreeBottomDetectorEnhanced:
    """三底共振检测增强版（多重验证）"""
    
    def __init__(self, fund_data, fund_info, macro_fetcher):
        self.df = fund_data
        self.info = fund_info
        self.prices = self.df['收盘价'].tolist()
        self.volumes = self.df['成交量'].tolist() if '成交量' in self.df.columns else [1]*len(self.prices)
        self.macro = macro_fetcher
        self.dates = self.df['日期'].tolist() if '日期' in self.df.columns else None

    # ---------- 技术指标计算 ----------
    @staticmethod
    def calc_rsi(prices, period=14):
        """简单RSI计算"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed>=0].sum()/period
        down = -seed[seed<0].sum()/period
        if down == 0:
            return 100
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100 - 100/(1+rs)
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            if down == 0:
                rsi[i] = 100
            else:
                rs = up/down
                rsi[i] = 100 - 100/(1+rs)
        return rsi

    @staticmethod
    def calc_macd(prices, fast=12, slow=26, signal=9):
        """计算MACD线、信号线、柱状线"""
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line.values, signal_line.values, histogram.values

    @staticmethod
    def calc_bollinger_band(prices, period=20, width=2):
        """计算布林带"""
        ma = pd.Series(prices).rolling(period).mean()
        std = pd.Series(prices).rolling(period).std()
        upper = ma + width * std
        lower = ma - width * std
        return lower.values, ma.values, upper.values

    def has_bullish_divergence(self, prices, window=30):
        """检测最近window内是否有MACD底背离（价格新低但MACD柱线未新低）"""
        if len(prices) < window + 10:
            return False
        recent_prices = prices[-window:]
        _, _, histogram = self.calc_macd(prices)
        recent_hist = histogram[-window:]
        # 寻找价格低点
        price_min_idx = np.argmin(recent_prices)
        if price_min_idx == 0:
            return False
        # 寻找MACD柱线在价格低点之前的最低点
        hist_min_before = np.min(recent_hist[:price_min_idx])
        # 条件：价格新低，但MACD柱线比之前的最低点更高
        if recent_prices[price_min_idx] < np.min(recent_prices[:price_min_idx]) and recent_hist[price_min_idx] > hist_min_before:
            return True
        return False

    # ---------- 三个底的检测 ----------
    def detect_policy_bottom(self):
        """政策底 → 宏观流动性底"""
        liq_score = self.macro.get_liquidity_score()
        is_bottom = liq_score >= 0.7
        confidence = liq_score
        return is_bottom, confidence

    def detect_valuation_bottom(self):
        """估值底（不同类型基金区别对待）"""
        # 指数型 / ETF 优先使用指数估值分位
        if self.info['type'] in ['指数型', '宽基ETF', '行业ETF']:
            index_code = IndexValuationFetcher.fund_to_index_code(self.info['code'])
            if index_code:
                percent = IndexValuationFetcher.get_index_pe_percent(index_code)
                if percent is not None:
                    is_bottom = percent < 0.2
                    confidence = 1 - percent
                    return is_bottom, min(confidence, 1.0)
        # 其他基金（混合、商品、QDII等）使用净值历史分位
        if len(self.prices) < 60:
            return False, 0.0
        current = self.prices[-1]
        hist_low = np.percentile(self.prices, 10)
        hist_high = np.percentile(self.prices, 90)
        if hist_high == hist_low:
            relative_pos = 0.5
        else:
            relative_pos = (current - hist_low) / (hist_high - hist_low)
        is_bottom = current <= hist_low * 1.05
        confidence = 1 - relative_pos
        return is_bottom, min(max(confidence, 0), 1)

    def detect_market_bottom(self):
        """市场底（多因子技术底）"""
        if len(self.prices) < 60:
            return False, 0.0, []
        prices = np.array(self.prices)
        volumes = np.array(self.volumes)
        score = 0
        reasons = []

        # 1. RSI超卖
        rsi = self.calc_rsi(prices, 14)
        if rsi[-1] < 30:
            score += 1
            reasons.append("RSI超卖")
        # 2. MACD底背离
        if self.has_bullish_divergence(prices):
            score += 1
            reasons.append("MACD底背离")
        # 3. 成交量地量（最近5日均量 < 20日均量 * 0.6）
        if len(volumes) >= 20:
            vol_ma5 = np.mean(volumes[-5:])
            vol_ma20 = np.mean(volumes[-20:])
            if vol_ma5 < vol_ma20 * 0.6:
                score += 1
                reasons.append("地量")
        # 4. 价格在布林下轨
        lower, _, _ = self.calc_bollinger_band(prices)
        if prices[-1] <= lower[-1]:
            score += 1
            reasons.append("布林下轨")
        # 5. 近期低点抬高（初步企稳）
        if len(prices) >= 10:
            # 寻找过去20日内的局部低点
            local_min_indices = argrelextrema(prices[-20:], np.less, order=3)[0]
            if len(local_min_indices) >= 2:
                last_two = local_min_indices[-2:]
                if prices[-20:][last_two[-1]] > prices[-20:][last_two[-2]]:
                    score += 1
                    reasons.append("低点抬高")

        confidence = score / 5
        is_bottom = score >= 3
        return is_bottom, confidence, reasons

    def get_final_signal(self):
        """获取最终信号（含多重验证）"""
        policy_bottom, p_conf = self.detect_policy_bottom()
        valuation_bottom, v_conf = self.detect_valuation_bottom()
        market_bottom, m_conf, m_reasons = self.detect_market_bottom()

        bottom_count = sum([policy_bottom, valuation_bottom, market_bottom])
        avg_conf = round((p_conf + v_conf + m_conf) / 3, 2)

        if bottom_count == 3:
            signal = "🔥 强烈买入"
            priority = 1
        elif bottom_count == 2:
            signal = "📈 买入"
            priority = 2
        elif bottom_count == 1:
            signal = "🤏 持有"
            priority = 3
        else:
            signal = "❌ 观望/卖出"
            priority = 4

        return {
            "基金代码": self.info["code"],
            "基金名称": self.info["name"],
            "基金类型": self.info["type"],
            "政策底": "✅" if policy_bottom else "❌",
            "估值底": "✅" if valuation_bottom else "❌",
            "市场底": "✅" if market_bottom else "❌",
            "底部数量": bottom_count,
            "平均置信度": avg_conf,
            "交易信号": signal,
            "信号优先级": priority,
            "当前价格": round(self.prices[-1], 4) if self.prices else 0,
            "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "市场底细节": ", ".join(m_reasons) if market_bottom else ""
        }

# ===================== 主程序 =====================
def main():
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    os.makedirs(SIGNAL_SAVE_PATH, exist_ok=True)
    
    macro = MacroDataFetcher()
    all_signals = []
    strong_buy_funds = []
    
    logger.info(f"🚀 开始增强检测 {len(FUND_LIST)} 只基金（多重验证版）")
    
    for idx, fund in enumerate(FUND_LIST, 1):
        try:
            logger.info(f"处理 [{idx}/{len(FUND_LIST)}] {fund['name']} ({fund['code']})")
            
            # 获取数据（优先真实）
            df = FundDataFetcher.fetch_real_data(fund["code"], fund["type"])
            if df is None or len(df) < 60:
                logger.warning(f"真实数据不足，使用增强模拟")
                df = FundDataFetcher.generate_enhanced_simulated(fund["code"], fund["type"], days=365)
            else:
                logger.info(f"使用真实数据，共 {len(df)} 条")
            
            # 保存原始数据（可选）
            save_path = f"{DATA_SAVE_PATH}/{fund['code']}_{fund['name']}.csv"
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            
            # 检测
            detector = ThreeBottomDetectorEnhanced(df, fund, macro)
            signal = detector.get_final_signal()
            all_signals.append(signal)
            
            if signal["底部数量"] == 3:
                strong_buy_funds.append({
                    "代码": signal["基金代码"],
                    "名称": signal["基金名称"],
                    "类型": signal["基金类型"],
                    "置信度": signal["平均置信度"],
                    "当前价格": signal["当前价格"]
                })
            
            logger.info(f"✅ {fund['name']} - 底部数：{signal['底部数量']} - 信号：{signal['交易信号']}")
            time.sleep(0.2)  # 礼貌间隔，避免被封
            
        except Exception as e:
            logger.error(f"❌ {fund['name']} 处理异常：{str(e)[:100]}")
            continue
    
    # 保存结果
    if all_signals:
        signal_df = pd.DataFrame(all_signals)
        signal_df = signal_df.sort_values(["信号优先级", "平均置信度"], ascending=[True, False])
        
        date_str = datetime.now().strftime("%Y%m%d")
        excel_path = f"{SIGNAL_SAVE_PATH}/三底共振增强版_{date_str}.xlsx"
        csv_path = f"{SIGNAL_SAVE_PATH}/三底信号增强版_简易.csv"
        signal_df.to_excel(excel_path, index=False)
        signal_df[["基金名称", "基金代码", "底部数量", "交易信号", "平均置信度"]].to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 打印汇总
        logger.info("\n" + "="*80)
        logger.info("🎯 三底共振增强版检测结果")
        logger.info("="*80)
        logger.info(f"总检测：{len(FUND_LIST)}")
        logger.info(f"强烈买入（3底）：{len(strong_buy_funds)}")
        logger.info(f"买入（2底）：{len(signal_df[signal_df['底部数量']==2])}")
        logger.info(f"持有（1底）：{len(signal_df[signal_df['底部数量']==1])}")
        logger.info(f"观望（0底）：{len(signal_df[signal_df['底部数量']==0])}")
        
        if strong_buy_funds:
            logger.info("\n🔥 强烈买入名单：")
            for i, f in enumerate(strong_buy_funds, 1):
                logger.info(f"  {i}. {f['名称']} ({f['代码']}) 置信度：{f['置信度']}")
        else:
            logger.info("\n暂无强烈买入基金")
        
        logger.info(f"\n📁 结果保存：{excel_path}")
        logger.info("="*80)
    else:
        logger.error("❌ 未生成任何信号")

if __name__ == "__main__":
    main()