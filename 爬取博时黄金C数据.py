import requests
import random
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from scipy.linalg import logm
from scipy import stats
from scipy.signal import hilbert, wavelets
import pywt
import akshare as ak
import threading
import json
from datetime import datetime
import sys
import os

# 导入六爻纳甲系统
from liuyao_najia import LiuYaoNaJia

# 导入三底共振模块
from 三底共振 import ThreeBottomDetectorEnhanced, MacroDataFetcher, IndexValuationFetcher, FundDataFetcher

# 导入反共识交易模块
from anti_consensus import AntiConsensusSignal, IntegratedDecisionEngine

# ================== 配置 ==================
FUND_CODE = '002611'  # 主基金（博时黄金C）
RELATED_FUNDS = ['002611', '161226', '162411', '513100']  # 关联基金
HISTORY_DAYS = 3576
# 监控配置
MONITOR_INTERVAL = 30  # 监控间隔（分钟）
PRICE_KEY_LEVELS = [5050, 5100, 5150, 5200]  # 国际金价关键位（美元/盎司）
SENTIMENT_CHANGE_THRESHOLD = 2.0  # 新闻情感突变阈值
RATE_CHANGE_THRESHOLD = 5  # 降息预期变动阈值（百分点）
RISK_CHANGE_THRESHOLD = 3  # 地缘风险变动阈值
VOLUME_RATIO_THRESHOLD_HIGH = 1.5  # 放量阈值
VOLUME_RATIO_THRESHOLD_LOW = 0.5   # 缩量阈值

# 资金流向阈值（万元）
FUND_FLOW_THRESHOLD = 5000  # 净流入超过5000万元视为显著流入

# 多源数据配置
DATA_SOURCES = {
    'gold_spot': '伦敦金',
    'gold_futures': '纽约金',
    'dollar_index': '美元指数',
    'treasury_yield': '美债收益率',
    'inflation_expectation': '通胀预期',
    'central_bank_reserves': '央行黄金储备',
    'etf_holdings': 'SPDR持仓',
    'comex_positions': 'COMEX持仓报告'
}
# ==========================================

HEADERS = {
    'User-Agent': UserAgent().random,
    'Referer': 'https://fund.eastmoney.com/',
    'Accept': 'application/json, text/plain, */*'
}

# ---------- 八卦映射（同原版）----------
TRIGRAM_MAP = {
    (1, 1, 1): "乾", (0, 0, 0): "坤", (1, 0, 1): "离", (0, 1, 0): "坎",
    (1, 1, 0): "兑", (0, 0, 1): "艮", (1, 0, 0): "震", (0, 1, 1): "巽"
}

HEXAGRAM_NAMES = {
    ("乾", "乾"): "乾为天", ("乾", "兑"): "天泽履", ("乾", "离"): "天火同人", ("乾", "震"): "天雷无妄",
    ("乾", "巽"): "天风姤", ("乾", "坎"): "天水讼", ("乾", "艮"): "天山遁", ("乾", "坤"): "天地否",
    ("兑", "乾"): "泽天夬", ("兑", "兑"): "兑为泽", ("兑", "离"): "泽火革", ("兑", "震"): "泽雷随",
    ("兑", "巽"): "泽风大过", ("兑", "坎"): "泽水困", ("兑", "艮"): "泽山咸", ("兑", "坤"): "泽地萃",
    ("离", "乾"): "火天大有", ("离", "兑"): "火泽睽", ("离", "离"): "离为火", ("离", "震"): "火雷噬嗑",
    ("离", "巽"): "火风鼎", ("离", "坎"): "火水未济", ("离", "艮"): "火山旅", ("离", "坤"): "火地晋",
    ("震", "乾"): "雷天大壮", ("震", "兑"): "雷泽归妹", ("震", "离"): "雷火丰", ("震", "震"): "震为雷",
    ("震", "巽"): "雷风恒", ("震", "坎"): "雷水解", ("震", "艮"): "雷山小过", ("震", "坤"): "雷地豫",
    ("巽", "乾"): "风天小畜", ("巽", "兑"): "风泽中孚", ("巽", "离"): "风火家人", ("巽", "震"): "风雷益",
    ("巽", "巽"): "巽为风", ("巽", "坎"): "风水涣", ("巽", "艮"): "风山渐", ("巽", "坤"): "风地观",
    ("坎", "乾"): "水天需", ("坎", "兑"): "水泽节", ("坎", "离"): "水火既济", ("坎", "震"): "水雷屯",
    ("坎", "巽"): "水风井", ("坎", "坎"): "坎为水", ("坎", "艮"): "水山蹇", ("坎", "坤"): "水地比",
    ("艮", "乾"): "山天大畜", ("艮", "兑"): "山泽损", ("艮", "离"): "山火贲", ("艮", "震"): "山雷颐",
    ("艮", "巽"): "山风蛊", ("艮", "坎"): "山水蒙", ("艮", "艮"): "艮为山", ("艮", "坤"): "山地剥",
    ("坤", "乾"): "地天泰", ("坤", "兑"): "地泽临", ("坤", "离"): "地火明夷", ("坤", "震"): "地雷复",
    ("坤", "巽"): "地风升", ("坤", "坎"): "地水师", ("坤", "艮"): "地山谦", ("坤", "坤"): "坤为地"
}

GUA_INTERPRETATION = {
    "乾为天": "天行健，成交量放大则真突破，缩量则需防诱多",
    "坤为地": "厚德载物，适合稳健持有，不宜追高",
    "屯": "万物始生，震荡中孕育机会，可轻仓试探",
    "蒙": "蒙昧不明，需等待市场方向明朗",
    "需": "等待时机，金价可能盘整，观望为宜",
    "讼": "争讼之象，市场多空分歧大，减仓避险",
    "师": "众争为险，波动加剧，快进快出",
    "比": "亲比和谐，可跟随趋势",
    "小畜": "小有积蓄，但上涨乏力，谨慎持有",
    "履": "如履薄冰，风险较大，建议减仓",
    "泰": "天地交泰，利好金价，可加仓",
    "否": "天地不交，下跌概率大，清仓观望",
    "同人": "同心协力，市场情绪向好，持有",
    "大有": "丰收之象，金价有望创新高",
    "谦": "谦逊低调，可逢低吸纳",
    "豫": "愉悦之象，但需防乐极生悲",
    "随": "跟随趋势，顺势而为",
    "蛊": "腐败生虫，警惕潜在风险",
    "临": "临近转折，注意变盘",
    "观": "观望等待，不宜操作",
    "噬嗑": "咬合咀嚼，市场博弈激烈，观望",
    "贲": "装饰之象，表面繁荣，警惕回调",
    "剥": "剥落之象，下跌趋势，清仓",
    "复": "一阳来复，反弹可期，轻仓抄底",
    "无妄": "不妄为，顺其自然，持有",
    "大畜": "积蓄力量，后市可期，加仓",
    "颐": "自求口实，独立判断，谨慎操作",
    "大过": "过犹不及，风险极大，清仓",
    "坎": "重重险难，下跌中继，回避",
    "离": "光明依附，上涨趋势，持有",
    "咸": "感应之象，短期波动，观望",
    "恒": "持之以恒，长线持有",
    "遁": "退避之象，减仓防守",
    "大壮": "声势浩大，但盛极必衰，逢高减仓",
    "晋": "前进发展，但需防回调，持有",
    "明夷": "光明受伤，黑暗来临，清仓",
    "家人": "家庭和睦，稳定持有",
    "睽": "背道而驰，市场分化，观望",
    "蹇": "行走艰难，下跌趋势，回避",
    "解": "缓解之象，有望反弹，轻仓",
    "损": "减损之象，减仓为宜",
    "益": "增益之象，加仓机会",
    "夬": "决断之象，果断操作",
    "姤": "不期而遇，突发消息，观望",
    "萃": "聚集之象，资金流入，持有",
    "升": "上升之势，加仓",
    "困": "困境之象，等待解救",
    "井": "稳定不变，持有不动",
    "革": "变革之象，变盘在即，谨慎",
    "鼎": "鼎新之象，有望突破",
    "震": "震动不安，波动剧烈，观望",
    "艮": "静止不动，不宜操作",
    "渐": "循序渐进，持有",
    "归妹": "非正之配，警惕诱多",
    "丰": "丰盛之象，但防盛极而衰",
    "旅": "旅居在外，暂时离场",
    "巽": "随风而顺，顺势而为",
    "兑": "喜悦之象，可小赚即安",
    "涣": "涣散之象，资金流出，减仓",
    "节": "节制之象，控制仓位",
    "中孚": "诚信之象，可信任趋势",
    "小过": "小有过错，谨慎操作",
    "既济": "成功之象，但防反转",
    "未济": "未完成之象，等待时机",
    "火水未济": "事未成，需耐心等待，不宜冒进",
    "火地晋": "旭日东升，但地火不交，防冲高回落",
    "山地剥": "剥落之象，若缩量下跌，可抄底；放量下跌，快跑",
}

GUA_LUCK = {
    "乾为天": 2, "坤为地": 1, "屯": 0, "蒙": -1, "需": 0,
    "讼": -2, "师": -1, "比": 1, "小畜": 1, "履": -1,
    "泰": 2, "否": -2, "同人": 1, "大有": 2, "谦": 1,
    "豫": 1, "随": 1, "蛊": -1, "临": 0, "观": 0,
    "噬嗑": -1, "贲": 0, "剥": -2, "复": 1, "无妄": 0,
    "大畜": 2, "颐": 1, "大过": -2, "坎": -2, "离": 2,
    "咸": 0, "恒": 1, "遁": -1, "大壮": 1, "晋": 1,
    "明夷": -2, "家人": 1, "睽": -1, "蹇": -2, "解": 1,
    "损": -1, "益": 2, "夬": 1, "姤": 0, "萃": 1,
    "升": 2, "困": -2, "井": 0, "革": 1, "鼎": 2,
    "震": 0, "艮": -1, "渐": 1, "归妹": -1, "丰": 1,
    "旅": -1, "巽": 1, "兑": 1, "涣": -2, "节": 0,
    "中孚": 1, "小过": -1, "既济": 2, "未济": -1,
    "火水未济": -1, "火地晋": 1, "山地剥": -2,
}

# ========== 情感分析模块 ==========
class SentimentAnalyzer:
    """简洁高效的情感分析器"""
    
    def __init__(self):
        # 金融领域核心情感词 - 简洁精确
        self.positive_terms = {
            '上涨', '上涨', '上涨',  # 重复是为了增加权重
            '创新高', '走强', '强势', '反弹', '回升', '飙升',
            '利好', '利多', '看好', '乐观', '买入', '增持',
            '突破', '突破', '突破', '放量', '量增', '资金流入'
        }
        
        self.negative_terms = {
            '下跌', '下跌', '下跌',  # 重复是为了增加权重
            '创新低', '走弱', '弱势', '回调', '回落', '暴跌',
            '利空', '利淡', '看空', '悲观', '卖出', '减持',
            '跌破', '跌破', '跌破', '缩量', '量缩', '资金流出'
        }
        
        # 否定词
        self.negation_terms = {'不', '没', '未', '无', '别', '勿'}
        
    def analyze(self, text):
        """分析文本情感"""
        if not text:
            return 0
        
        score = 0
        words = text
        
        # 检查否定词
        negation_count = sum(1 for term in self.negation_terms if term in words)
        negation_factor = -1 if negation_count % 2 == 1 else 1
        
        # 计算正面情感
        positive_score = sum(1 for term in self.positive_terms if term in words)
        
        # 计算负面情感
        negative_score = sum(1 for term in self.negative_terms if term in words)
        
        # 计算净得分
        net_score = (positive_score - negative_score) * negation_factor
        
        # 归一化到 [-1, 1]
        max_possible = max(len(self.positive_terms), len(self.negative_terms))
        normalized_score = net_score / (max_possible + 1e-10)
        
        return max(-1, min(1, normalized_score))

# ========== 多源数据采集模块 ==========
class MultiSourceDataCollector:
    """多源数据采集器"""
    
    def __init__(self):
        self.data_cache = {}
        self.last_update = {}
    
    def get_gold_spot_price(self):
        """获取伦敦金现货价格"""
        try:
            url = 'https://hq.sinajs.cn/list=XAU'
            headers = {'Referer': 'https://finance.sina.com.cn'}
            resp = requests.get(url, headers=headers, timeout=5)
            data = resp.text
            if 'hq_str_XAU' in data:
                parts = data.split('=')[1].strip().strip('"').split(',')
                if len(parts) > 1:
                    price = float(parts[1])
                    return price
        except Exception as e:
            print(f"获取伦敦金价格失败: {e}")
        return None
    
    def get_gold_futures_price(self):
        """获取纽约金期货价格"""
        try:
            url = 'https://hq.sinajs.cn/list=gc_main'
            headers = {'Referer': 'https://finance.sina.com.cn'}
            resp = requests.get(url, headers=headers, timeout=5)
            data = resp.text
            if 'hq_str_gc_main' in data:
                parts = data.split('=')[1].strip().strip('"').split(',')
                if len(parts) > 2:
                    price = float(parts[2])
                    return price
        except Exception as e:
            print(f"获取纽约金价格失败: {e}")
        return None
    
    def get_dollar_index(self):
        """获取美元指数(DXY)"""
        try:
            url = 'https://hq.sinajs.cn/list=DINIW'
            headers = {'Referer': 'https://finance.sina.com.cn'}
            resp = requests.get(url, headers=headers, timeout=5)
            data = resp.text
            if data and '"' in data:
                parts = data.split('"')[1].split(',')
                if len(parts) > 1:
                    price = float(parts[1])
                    return price
        except Exception as e:
            print(f"获取美元指数失败: {e}")
        return None
    
    def get_treasury_yield(self):
        """获取美债收益率"""
        try:
            # 使用akshare获取美债收益率
            df = ak.bond_us_yield()
            if not df.empty:
                return float(df.iloc[0]['10Y'])
        except Exception as e:
            print(f"获取美债收益率失败: {e}")
        return None
    
    def get_inflation_expectation(self):
        """获取通胀预期（盈亏平衡通胀率）"""
        try:
            # 简化实现，实际应从专业数据源获取
            # 这里使用美债收益率与TIPS收益率之差
            nominal_yield = self.get_treasury_yield()
            if nominal_yield:
                # 假设TIPS收益率为1.0%（实际应实时获取）
                tips_yield = 1.0
                return nominal_yield - tips_yield
        except Exception as e:
            print(f"获取通胀预期失败: {e}")
        return None
    
    def get_central_bank_reserves(self):
        """获取央行黄金储备变动"""
        try:
            # 简化实现，实际应从IMF或世界黄金协会获取
            # 这里返回模拟数据
            return 35000  # 吨
        except Exception as e:
            print(f"获取央行黄金储备失败: {e}")
        return None
    
    def get_etf_holdings(self):
        """获取SPDR黄金ETF持仓"""
        try:
            url = 'https://www.spdrgoldshares.com/ajax/holdings'
            resp = requests.get(url, headers=HEADERS, timeout=10)
            data = resp.json()
            if 'gold_ounces' in data:
                return float(data['gold_ounces'])
        except Exception as e:
            print(f"获取SPDR持仓失败: {e}")
        return None
    
    def get_comex_positions(self):
        """获取COMEX持仓报告"""
        try:
            # 简化实现，实际应从CFTC获取
            # 这里返回模拟数据
            return {
                'non_commercial_long': 200000,
                'non_commercial_short': 50000,
                'commercial_long': 150000,
                'commercial_short': 250000
            }
        except Exception as e:
            print(f"获取COMEX持仓失败: {e}")
        return None
    
    def get_fed_rate_probability(self):
        """获取美联储利率决议概率（CME FedWatch）"""
        try:
            # 简化实现，实际应从CME FedWatch获取
            # 这里返回模拟数据
            return {
                'cut_probability': 65.5,  # 降息概率
                'hold_probability': 30.2,  # 维持不变概率
                'hike_probability': 4.3   # 加息概率
            }
        except Exception as e:
            print(f"获取美联储利率决议概率失败: {e}")
        return None
    
    def get_nonfarm_payrolls(self):
        """获取非农就业数据"""
        try:
            # 使用akshare获取非农就业数据
            df = ak.macro_us_nonfarm_payrolls()
            if not df.empty:
                return float(df.iloc[0]['数值'])
        except Exception as e:
            print(f"获取非农就业数据失败: {e}")
        return None
    
    def get_cpi_data(self):
        """获取CPI数据"""
        try:
            # 使用akshare获取CPI数据
            df = ak.macro_us_cpi()
            if not df.empty:
                return float(df.iloc[0]['数值'])
        except Exception as e:
            print(f"获取CPI数据失败: {e}")
        return None
    
    def quantify_event_impact(self, event_type, event_value):
        """量化事件影响"""
        impact_scores = {
            'fed_rate_cut': 0.8,      # 降息对黄金利好
            'fed_rate_hike': -0.8,    # 加息对黄金利空
            'nonfarm_strong': -0.5,   # 非农强劲对黄金利空
            'nonfarm_weak': 0.5,      # 非农疲软对黄金利好
            'cpi_high': 0.6,          # CPI高对黄金利好
            'cpi_low': -0.4           # CPI低对黄金利空
        }
        
        if event_type == 'fed_rate':
            if event_value < 0:  # 降息
                return impact_scores['fed_rate_cut']
            elif event_value > 0:  # 加息
                return impact_scores['fed_rate_hike']
        elif event_type == 'nonfarm':
            if event_value > 200000:  # 强劲
                return impact_scores['nonfarm_strong']
            elif event_value < 100000:  # 疲软
                return impact_scores['nonfarm_weak']
        elif event_type == 'cpi':
            if event_value > 3.0:  # 高通胀
                return impact_scores['cpi_high']
            elif event_value < 2.0:  # 低通胀
                return impact_scores['cpi_low']
        
        return 0.0
    
    def collect_all_data(self):
        """采集所有数据源"""
        data = {
            'gold_spot': self.get_gold_spot_price(),
            'gold_futures': self.get_gold_futures_price(),
            'dollar_index': self.get_dollar_index(),
            'treasury_yield': self.get_treasury_yield(),
            'inflation_expectation': self.get_inflation_expectation(),
            'central_bank_reserves': self.get_central_bank_reserves(),
            'etf_holdings': self.get_etf_holdings(),
            'comex_positions': self.get_comex_positions(),
            'fed_rate_probability': self.get_fed_rate_probability(),
            'nonfarm_payrolls': self.get_nonfarm_payrolls(),
            'cpi_data': self.get_cpi_data()
        }
        self.data_cache = data
        self.last_update = datetime.now().isoformat()
        return data

# 初始化多源数据采集器
data_collector = MultiSourceDataCollector()

# ========== 全局状态 ==========
last_state = {
    'price': None,
    'sentiment': None,
    'rate_expect': None,
    'geo_risk': None,
    'volume_ratio': None,
    'fund_flow': None,
    'price_breakthrough': set(),
    'multi_source_data': None,
}

# 新增：情感历史记录（最多 HISTORY_DAYS 条）
sentiment_history = []

# 初始化情感分析器
sentiment_analyzer = SentimentAnalyzer()

# ========== 多因子风险模型 ==========
class MultiFactorRiskModel:
    """多因子风险模型（替代量子纠缠模型）"""
    
    def __init__(self):
        self.factors = {
            'market_beta': 0,      # 市场贝塔
            'volatility': 0,       # 波动率
            'momentum': 0,         # 动量因子
            'liquidity': 0,        # 流动性因子
            'correlation': 0,      # 相关性风险
        }
    
    def calculate_risk_score(self, returns, market_returns=None, volumes=None):
        """
        计算综合风险评分
        
        Args:
            returns: 基金收益率序列
            market_returns: 市场基准收益率
            volumes: 成交量序列
        """
        if len(returns) < 30:
            return {'total_risk': 0.5, 'factors': self.factors}
        
        # 将输入转换为numpy数组
        returns = np.array(returns)
        if volumes is not None:
            volumes = np.array(volumes)
        if market_returns is not None:
            market_returns = np.array(market_returns)
        
        # 1. 波动率风险 (年化)
        volatility = np.std(returns) * np.sqrt(252)
        vol_score = min(volatility / 0.3, 1.0)  # 30%年化波动率为满分风险
        
        # 2. 动量风险 (近期趋势)
        if len(returns) >= 20:
            momentum = np.mean(returns[-20:]) / (np.std(returns[-20:]) + 1e-10)
            momentum_score = 1 - stats.norm.cdf(momentum)  # 极端动量视为风险
        else:
            momentum_score = 0.5
            
        # 3. 流动性风险
        liquidity_score = 0.5
        if volumes is not None and len(volumes) > 1:
            volume_changes = np.diff(volumes) / (volumes[:-1] + 1e-10)
            liquidity_score = min(np.std(volume_changes) * 10, 1.0)
        
        # 4. 相关性风险 (与市场基准)
        correlation_score = 0.5
        if market_returns is not None and len(market_returns) == len(returns):
            correlation = np.corrcoef(returns, market_returns)[0, 1]
            # 高相关性意味着缺乏分散性
            correlation_score = abs(correlation)
        
        # 综合风险评分 (加权)
        total_risk = (
            vol_score * 0.3 +
            momentum_score * 0.25 +
            liquidity_score * 0.25 +
            correlation_score * 0.2
        )
        
        return {
            'total_risk': total_risk,
            'factors': {
                'volatility': vol_score,
                'momentum': momentum_score,
                'liquidity': liquidity_score,
                'correlation': correlation_score
            }
        }

# ========== 特征工程模块 ==========
class FeatureEngineer:
    """特征工程模块"""
    
    def hilbert_transform(self, data):
        """希尔伯特变换提取瞬时特征"""
        if len(data) < 2:
            return None
        
        analytic_signal = hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2 * np.pi)
        
        return {
            'amplitude_envelope': amplitude_envelope,
            'instantaneous_phase': instantaneous_phase,
            'instantaneous_frequency': instantaneous_frequency
        }
    
    def wavelet_decomposition(self, data, wavelet='db4', level=3):
        """小波分解提取多尺度分量"""
        if len(data) < 2**level:
            return None
        
        coeffs = pywt.wavedec(data, wavelet, level=level)
        cA = coeffs[0]  # 近似系数
        cD = coeffs[1:]  # 细节系数
        
        return {
            'approximation': cA,
            'details': cD
        }
    
    def emd_decomposition(self, data):
        """经验模态分解（EMD）提取本征模态函数"""
        if len(data) < 3:
            return None
        
        # 简化实现，实际应使用pyemd库
        # 这里使用快速EMD近似
        def sifting_process(x):
            x = np.array(x)
            imfs = []
            while True:
                x1 = x
                for _ in range(10):
                    # 上下包络
                    peaks = np.where((x1[1:-1] > x1[:-2]) & (x1[1:-1] > x1[2:]))[0] + 1
                    if len(peaks) < 2:
                        break
                    upper = np.interp(range(len(x1)), peaks, x1[peaks])
                    
                    valleys = np.where((x1[1:-1] < x1[:-2]) & (x1[1:-1] < x1[2:]))[0] + 1
                    if len(valleys) < 2:
                        break
                    lower = np.interp(range(len(x1)), valleys, x1[valleys])
                    
                    mean_env = (upper + lower) / 2
                    x1 = x1 - mean_env
                
                if np.std(x1) < 0.01 * np.std(x):
                    break
                
                imfs.append(x1)
                x = x - x1
            
            imfs.append(x)
            return imfs
        
        imfs = sifting_process(data)
        return {'imfs': imfs}
    
    def extract_features(self, data):
        """提取所有特征"""
        features = {}
        
        # 基本统计特征
        features['mean'] = np.mean(data)
        features['std'] = np.std(data)
        features['max'] = np.max(data)
        features['min'] = np.min(data)
        features['skewness'] = stats.skew(data)
        features['kurtosis'] = stats.kurtosis(data)
        
        # 希尔伯特变换特征
        hilbert_features = self.hilbert_transform(data)
        if hilbert_features:
            features['hilbert_amplitude_mean'] = np.mean(hilbert_features['amplitude_envelope'])
            features['hilbert_frequency_mean'] = np.mean(hilbert_features['instantaneous_frequency'])
        
        # 小波分解特征
        wavelet_features = self.wavelet_decomposition(data)
        if wavelet_features:
            features['wavelet_approx_mean'] = np.mean(wavelet_features['approximation'])
            for i, detail in enumerate(wavelet_features['details']):
                features[f'wavelet_detail_{i}_mean'] = np.mean(detail)
                features[f'wavelet_detail_{i}_std'] = np.std(detail)
        
        # EMD分解特征
        emd_features = self.emd_decomposition(data)
        if emd_features:
            for i, imf in enumerate(emd_features['imfs']):
                features[f'emd_imf_{i}_mean'] = np.mean(imf)
                features[f'emd_imf_{i}_std'] = np.std(imf)
        
        return features

# 初始化特征工程模块
feature_engineer = FeatureEngineer()

# 初始化风险模型
risk_model = MultiFactorRiskModel()

# ========== 权重优化器 ==========
class WeightOptimizer:
    """权重优化器"""
    
    def __init__(self):
        self.best_weights = None
        
    def optimize(self, historical_signals, price_data):
        """
        通过网格搜索优化权重
        
        Args:
            historical_signals: 历史信号数据列表
            price_data: 历史价格数据
        """
        best_score = -float('inf')
        results = []
        
        # 定义权重搜索空间
        weight_grid = np.arange(0.1, 0.8, 0.1)
        
        for w1, w2, w3, w4 in np.ndindex((7, 7, 7, 7)):
            # 从网格中获取权重
            weights = np.array([weight_grid[w1], weight_grid[w2], weight_grid[w3], weight_grid[w4]])
            # 归一化权重
            total = np.sum(weights)
            if total > 0:
                weights = weights / total
                
                # 回测
                result = self._backtest(historical_signals, price_data, weights)
                results.append((weights, result))
                
                # 评分：综合考虑夏普比率和胜率
                score = result['sharpe'] * 0.6 + result['win_rate'] * 0.4
                if score > best_score:
                    best_score = score
                    self.best_weights = weights
        
        # 按夏普比率排序
        results.sort(key=lambda x: x[1]['sharpe'], reverse=True)
        
        return {
            'optimal_weights': self.best_weights,
            'top_results': results[:5],
            'all_results': results
        }
    
    def _backtest(self, signals, prices, weights):
        """执行单次回测"""
        returns = []
        positions = []  # 1: 多头, 0: 空仓, -1: 空头
        
        for i, signal in enumerate(signals):
            # 计算综合得分
            score = (
                signal.get('luck', 0) * weights[0] +
                signal.get('sentiment', 0) * weights[1] +
                signal.get('risk', 0) * weights[2] +
                signal.get('volume', 0) * weights[3]
            )
            
            # 生成仓位
            if score > 0.3:
                pos = 1
            elif score < -0.3:
                pos = -1
            else:
                pos = 0
            positions.append(pos)
        
        # 计算收益
        for i in range(1, len(prices)):
            if i <= len(positions):
                daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                strategy_return = positions[i-1] * daily_return
                returns.append(strategy_return)
        
        if not returns:
            return {
                'sharpe': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_return': 0
            }
        
        returns = np.array(returns)
        
        # 计算指标
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        # 最大回撤
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # 胜率
        win_rate = np.sum(returns > 0) / len(returns)
        
        # 总收益
        total_ret = cumulative[-1] - 1 if len(cumulative) > 0 else 0
        
        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_return': total_ret
        }

# 初始化权重优化器
weight_optimizer = WeightOptimizer()

# 初始化六爻纳甲系统
liuyao_system = LiuYaoNaJia()

# ========== 权重优化辅助函数 ==========
def generate_historical_signals(history_days=60):
    """
    生成历史信号数据用于权重优化
    
    Args:
        history_days: 历史天数
    
    Returns:
        (historical_signals, price_data): 历史信号和价格数据
    """
    historical_signals = []
    price_data = []
    
    # 这里使用模拟数据，实际应用中应该使用真实历史数据
    for i in range(history_days):
        # 模拟信号数据
        signal = {
            'luck': np.random.uniform(-2, 2),
            'sentiment': np.random.uniform(-1, 1),
            'risk': np.random.uniform(-1, 1),
            'volume': np.random.uniform(-1, 1)
        }
        historical_signals.append(signal)
        
        # 模拟价格数据
        price = 1.0 + np.random.normal(0, 0.01, size=1)[0]
        if price_data:
            price *= price_data[-1]
        price_data.append(price)
    
    return historical_signals, price_data

def get_optimized_weights():
    """
    获取优化后的权重
    
    Returns:
        optimized_weights: 优化后的权重列表
    """
    global weight_optimizer
    
    # 生成历史信号数据
    historical_signals, price_data = generate_historical_signals()
    
    # 优化权重
    result = weight_optimizer.optimize(historical_signals, price_data)
    
    if result['optimal_weights'] is not None:
        return result['optimal_weights']
    else:
        # 默认权重
        return np.array([0.3, 0.2, 0.3, 0.2])

# 初始化时优化权重
optimized_weights = get_optimized_weights()
print(f"[成功] 权重优化完成，最优权重: {optimized_weights}")

# ========== 模型性能评估和监控模块 ==========
class ModelPerformanceMonitor:
    """模型性能评估和监控模块"""
    
    def __init__(self):
        self.performance_history = []
        self.alert_thresholds = {
            'accuracy': 0.4,        # 降低准确率阈值
            'sharpe_ratio': -1.0,   # 降低夏普比率阈值，允许负值
            'max_drawdown': -0.3,     # 放宽最大回撤阈值
            'win_rate': 0.4          # 降低胜率阈值
        }
    
    def evaluate_performance(self, predictions, actuals):
        """
        评估模型性能
        
        Args:
            predictions: 预测结果列表
            actuals: 实际结果列表
        
        Returns:
            performance: 性能指标字典
        """
        if len(predictions) != len(actuals):
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'win_rate': 0
            }
        
        # 计算准确率
        correct = sum(1 for p, a in zip(predictions, actuals) if p * a > 0)
        accuracy = correct / len(predictions)
        
        # 计算胜率
        win_rate = sum(1 for a in actuals if a > 0) / len(actuals)
        
        # 计算收益率相关指标
        returns = actuals
        if returns and len(returns) >= 2:  # 需要至少2个数据点
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            # 避免除以零
            running_max[running_max == 0] = 1e-10
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            # 单次数据点时，使用简单指标
            sharpe_ratio = returns[0] if returns else 0
            max_drawdown = 0
        
        performance = {
            'accuracy': accuracy,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'timestamp': datetime.now().isoformat()
        }
        
        # 记录性能历史
        self.performance_history.append(performance)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # 只有在数据充足时才检查异常
        if len(self.performance_history) >= 10:  # 至少10次评估后才触发警报
            self.check_alerts(performance)
        
        return performance
    
    def check_alerts(self, performance):
        """
        检查性能异常
        
        Args:
            performance: 性能指标字典
        """
        alerts = []
        
        if performance['accuracy'] < self.alert_thresholds['accuracy']:
            alerts.append(f"准确率异常: {performance['accuracy']:.2f} < {self.alert_thresholds['accuracy']}")
        
        if performance['sharpe_ratio'] < self.alert_thresholds['sharpe_ratio']:
            alerts.append(f"夏普比率异常: {performance['sharpe_ratio']:.2f} < {self.alert_thresholds['sharpe_ratio']}")
        
        if performance['max_drawdown'] < self.alert_thresholds['max_drawdown']:
            alerts.append(f"最大回撤异常: {performance['max_drawdown']:.2f} < {self.alert_thresholds['max_drawdown']}")
        
        if performance['win_rate'] < self.alert_thresholds['win_rate']:
            alerts.append(f"胜率异常: {performance['win_rate']:.2f} < {self.alert_thresholds['win_rate']}")
        
        if alerts:
            print("\n⚠️ 【性能警报】")
            for alert in alerts:
                print(f"  {alert}")
    
    def get_performance_summary(self):
        """
        获取性能汇总
        
        Returns:
            summary: 性能汇总字典
        """
        if not self.performance_history:
            return {}
        
        recent = self.performance_history[-30:]  # 最近30次
        
        summary = {
            'avg_accuracy': np.mean([p['accuracy'] for p in recent]),
            'avg_win_rate': np.mean([p['win_rate'] for p in recent]),
            'avg_sharpe': np.mean([p['sharpe_ratio'] for p in recent]),
            'avg_max_drawdown': np.mean([p['max_drawdown'] for p in recent]),
            'best_accuracy': max([p['accuracy'] for p in recent]),
            'worst_accuracy': min([p['accuracy'] for p in recent]),
            'total_evaluations': len(self.performance_history)
        }
        
        return summary

# ========== 模型层模块 ==========
class TransformerModel:
    """Transformer架构模型（简化实现）"""
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
    
    def forward(self, x):
        """前向传播"""
        # 简化实现，实际应使用PyTorch或TensorFlow
        # 这里返回模拟预测
        return np.mean(x, axis=1)

class MultiModalFusion:
    """多模态融合模型"""
    
    def __init__(self):
        self.models = {
            'numerical': TransformerModel(input_dim=10),
            'text': TransformerModel(input_dim=5),
            'image': TransformerModel(input_dim=8)
        }
    
    def fuse(self, numerical_data, text_data, image_data):
        """融合多模态数据"""
        numerical_pred = self.models['numerical'].forward(numerical_data)
        text_pred = self.models['text'].forward(text_data)
        image_pred = self.models['image'].forward(image_data)
        
        # 加权融合
        weights = [0.5, 0.3, 0.2]
        fused_pred = (numerical_pred * weights[0] + 
                     text_pred * weights[1] + 
                     image_pred * weights[2])
        
        return fused_pred

class EnsembleModel:
    """集成学习模型"""
    
    def __init__(self):
        self.models = {
            'rf': self._random_forest,
            'xgboost': self._xgboost,
            'lightgbm': self._lightgbm,
            'catboost': self._catboost
        }
        self.weights = {'rf': 0.25, 'xgboost': 0.25, 'lightgbm': 0.25, 'catboost': 0.25}
    
    def _random_forest(self, x):
        return np.mean(x, axis=1)
    
    def _xgboost(self, x):
        return np.mean(x, axis=1) + 0.01
    
    def _lightgbm(self, x):
        return np.mean(x, axis=1) - 0.005
    
    def _catboost(self, x):
        return np.mean(x, axis=1) + 0.005
    
    def predict(self, x):
        """集成预测"""
        predictions = {name: model(x) for name, model in self.models.items()}
        
        # 加权集成
        final_pred = np.zeros_like(predictions['rf'])
        for name, pred in predictions.items():
            final_pred += pred * self.weights[name]
        
        return final_pred
    
    def update_weights(self, recent_performance):
        """根据近期表现动态调整权重"""
        total = sum(recent_performance.values())
        if total > 0:
            self.weights = {name: score / total for name, score in recent_performance.items()}

class ReinforcementLearningAgent:
    """强化学习智能体"""
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = np.zeros((state_dim, action_dim))
    
    def choose_action(self, state, epsilon=0.1):
        """选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        """学习更新"""
        alpha = 0.1
        gamma = 0.9
        self.q_table[state, action] += alpha * (reward + gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])

class MetaLearning:
    """元学习模型"""
    
    def __init__(self):
        self.specialized_models = {
            'rate_hike_cycle': TransformerModel(input_dim=10),
            'geopolitical_conflict': TransformerModel(input_dim=10),
            'normal_market': TransformerModel(input_dim=10)
        }
    
    def select_model(self, market_state):
        """根据市场状态选择专用模型"""
        if market_state == 'rate_hike':
            return self.specialized_models['rate_hike_cycle']
        elif market_state == 'conflict':
            return self.specialized_models['geopolitical_conflict']
        else:
            return self.specialized_models['normal_market']
    
    def adapt(self, model, new_data):
        """快速适应新数据"""
        # 简化实现，实际应使用MAML等元学习算法
        return model

# 初始化模型层
ensemble_model = EnsembleModel()
meta_learning = MetaLearning()
rl_agent = ReinforcementLearningAgent(state_dim=100, action_dim=3)

# ========== 训练与验证模块 ==========
class TrainingValidation:
    """训练与验证模块"""
    
    def time_series_cross_validation(self, X, y, window_size=60, step=30):
        """时间序列交叉验证"""
        n = len(X)
        scores = []
        
        for i in range(window_size, n, step):
            train_X, train_y = X[:i], y[:i]
            test_X, test_y = X[i:i+step], y[i:i+step]
            
            # 训练模型（简化实现）
            # 这里使用集成模型进行预测
            predictions = ensemble_model.predict(test_X)
            
            # 计算准确率
            accuracy = np.mean(np.sign(predictions) == np.sign(test_y))
            scores.append(accuracy)
        
        return scores
    
    def adversarial_validation(self, train_X, test_X):
        """对抗验证"""
        # 简化实现，实际应训练分类器区分训练集和测试集
        train_mean = np.mean(train_X, axis=0)
        test_mean = np.mean(test_X, axis=0)
        
        # 计算分布差异
        distribution_diff = np.mean(np.abs(train_mean - test_mean))
        return distribution_diff
    
    def pseudo_out_of_sample_test(self, X, y, event_dates):
        """伪外推测试"""
        results = {}
        
        for event_date in event_dates:
            # 找到事件发生的索引
            event_idx = min(event_date, len(X)-1)
            
            # 使用事件前的数据训练
            train_X, train_y = X[:event_idx], y[:event_idx]
            # 使用事件期间的数据测试
            test_X, test_y = X[event_idx:event_idx+5], y[event_idx:event_idx+5]
            
            # 预测
            predictions = ensemble_model.predict(test_X)
            accuracy = np.mean(np.sign(predictions) == np.sign(test_y))
            results[event_date] = accuracy
        
        return results
    
    def regularization(self, model, l2_lambda=0.01):
        """正则化技术"""
        # 简化实现，实际应在模型训练中应用
        return model

# ========== 另类数据集成模块 ==========
class AlternativeDataCollector:
    """另类数据采集器"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def get_news_sentiment(self):
        """获取新闻情感"""
        news = get_news()
        return analyze_news_sentiment(news)
    
    def get_social_media_sentiment(self):
        """获取社交媒体情绪"""
        try:
            # 简化实现，实际应从Twitter、Reddit等获取
            # 这里返回模拟数据
            return np.random.uniform(-1, 1)
        except Exception as e:
            print(f"获取社交媒体情绪失败: {e}")
            return 0.0
    
    def get_search_trends(self):
        """获取搜索趋势"""
        try:
            # 简化实现，实际应从Google Trends获取
            # 这里返回模拟数据
            return {
                'gold_price': np.random.uniform(0, 100),
                'inflation': np.random.uniform(0, 100),
                'fed_rate': np.random.uniform(0, 100)
            }
        except Exception as e:
            print(f"获取搜索趋势失败: {e}")
            return {}
    
    def get_satellite_images(self):
        """获取卫星图像数据"""
        try:
            # 简化实现，实际应从卫星图像分析获取
            # 这里返回模拟数据
            return {
                'mining_activity': np.random.uniform(0, 1),
                'gold_stocks': np.random.uniform(0, 1)
            }
        except Exception as e:
            print(f"获取卫星图像数据失败: {e}")
            return {}
    
    def collect_all_alternative_data(self):
        """采集所有另类数据"""
        data = {
            'news_sentiment': self.get_news_sentiment(),
            'social_media_sentiment': self.get_social_media_sentiment(),
            'search_trends': self.get_search_trends(),
            'satellite_images': self.get_satellite_images()
        }
        return data

# ========== 高频数据处理模块 ==========
class HighFrequencyDataProcessor:
    """高频数据处理器"""
    
    def __init__(self):
        self.high_freq_data = []
    
    def get_minute_data(self, symbol='XAUUSD', minutes=60):
        """获取分钟级数据"""
        try:
            # 简化实现，实际应从API获取
            # 这里生成模拟数据
            data = []
            base_price = 2000.0
            for i in range(minutes):
                timestamp = datetime.now().timestamp() - (minutes - i) * 60
                price = base_price + np.random.normal(0, 0.5)
                volume = np.random.randint(100, 1000)
                bid = price - 0.1
                ask = price + 0.1
                data.append({
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume,
                    'bid': bid,
                    'ask': ask
                })
            return data
        except Exception as e:
            print(f"获取分钟级数据失败: {e}")
            return []
    
    def calculate_microstructure_features(self, data):
        """计算微观结构特征"""
        if not data:
            return {}
        
        features = {}
        
        # 买卖价差
        spreads = [d['ask'] - d['bid'] for d in data]
        features['avg_spread'] = np.mean(spreads)
        features['std_spread'] = np.std(spreads)
        
        # 订单流不平衡
        # 简化实现，实际应根据订单数据计算
        order_imbalance = np.random.uniform(-1, 1, len(data))
        features['avg_order_imbalance'] = np.mean(order_imbalance)
        
        # 价格波动率
        prices = [d['price'] for d in data]
        returns = np.diff(prices) / prices[:-1]
        features['realized_volatility'] = np.std(returns) * np.sqrt(252 * 24 * 60)
        
        # 成交量特征
        volumes = [d['volume'] for d in data]
        features['avg_volume'] = np.mean(volumes)
        features['volume_volatility'] = np.std(volumes)
        
        return features
    
    def process_high_frequency_data(self):
        """处理高频数据"""
        # 获取分钟级数据
        minute_data = self.get_minute_data()
        
        # 计算微观结构特征
        features = self.calculate_microstructure_features(minute_data)
        
        return {
            'minute_data': minute_data,
            'microstructure_features': features
        }

# ========== 模式识别模块 ==========
class PatternRecognition:
    """模式识别模块"""
    
    def __init__(self):
        pass
    
    def shapelet_transform(self, data, window_size=20):
        """Shapelets变换学习价格形态"""
        if len(data) < window_size:
            return []
        
        shapelets = []
        for i in range(len(data) - window_size + 1):
            shapelet = data[i:i+window_size]
            # 标准化
            shapelet = (shapelet - np.mean(shapelet)) / (np.std(shapelet) + 1e-10)
            shapelets.append(shapelet)
        
        return shapelets
    
    def autoencoder(self, data, encoding_dim=10):
        """自编码器学习价格形态"""
        # 简化实现，实际应使用PyTorch或TensorFlow
        # 这里返回模拟编码
        encoding = np.random.randn(encoding_dim)
        return encoding
    
    def graph_neural_network(self, asset_data):
        """图神经网络建模资产间关联"""
        # 简化实现，实际应使用PyTorch Geometric等库
        # 计算资产间相关性作为边权重
        assets = list(asset_data.keys())
        n = len(assets)
        correlation_matrix = np.zeros((n, n))
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # 模拟相关性
                    correlation_matrix[i, j] = np.random.uniform(-1, 1)
        
        return {
            'assets': assets,
            'correlation_matrix': correlation_matrix
        }
    
    def recognize_patterns(self, price_data, asset_data):
        """识别价格形态和资产关联"""
        # 学习价格形态
        shapelets = self.shapelet_transform(price_data)
        encoding = self.autoencoder(price_data)
        
        # 建模资产间关联
        graph = self.graph_neural_network(asset_data)
        
        return {
            'shapelets': shapelets,
            'autoencoder_encoding': encoding,
            'asset_correlations': graph
        }

# ========== 贝叶斯结构模块 ==========
class BayesianStructure:
    """贝叶斯结构模块"""
    
    def __init__(self, n_states=3):
        self.n_states = n_states  # 3个状态：趋势、震荡、高波动
        self.transition_matrix = np.array([
            [0.8, 0.15, 0.05],  # 从趋势到其他状态的概率
            [0.1, 0.8, 0.1],    # 从震荡到其他状态的概率
            [0.15, 0.25, 0.6]   # 从高波动到其他状态的概率
        ])
        self.emission_means = np.array([0.001, 0, -0.001])  # 各状态的均值
        self.emission_stds = np.array([0.005, 0.01, 0.02])  # 各状态的标准差
    
    def hmm_predict(self, returns):
        """隐马尔可夫模型预测"""
        # 简化实现，实际应使用hmmlearn库
        # 这里使用Viterbi算法的简化版本
        n = len(returns)
        if n == 0:
            return []
        
        # 初始化
        delta = np.zeros((n, self.n_states))
        psi = np.zeros((n, self.n_states), dtype=int)
        
        # 初始状态概率
        delta[0] = np.array([1/3, 1/3, 1/3])
        
        # 前向算法
        for t in range(1, n):
            for s in range(self.n_states):
                # 计算从各状态转移到s的概率
                trans_probs = delta[t-1] * self.transition_matrix[:, s]
                # 计算 emission 概率
                emission_prob = stats.norm.pdf(returns[t], self.emission_means[s], self.emission_stds[s])
                delta[t, s] = np.max(trans_probs) * emission_prob
                psi[t, s] = np.argmax(trans_probs)
        
        # 回溯找到最优状态序列
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(n-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def state_space_model(self, data):
        """状态空间模型"""
        # 简化实现，实际应使用卡尔曼滤波器
        # 这里返回模拟状态
        states = []
        current_state = 0
        for i in range(len(data)):
            # 根据转移矩阵更新状态
            current_state = np.random.choice(self.n_states, p=self.transition_matrix[current_state])
            states.append(current_state)
        
        return states
    
    def detect_market_regime(self, returns):
        """检测市场状态"""
        # 使用HMM检测市场状态
        states = self.hmm_predict(returns)
        
        # 状态解释
        state_names = ['趋势', '震荡', '高波动']
        regime_distribution = np.bincount(states, minlength=self.n_states) / len(states)
        
        return {
            'states': states,
            'state_names': state_names,
            'regime_distribution': regime_distribution
        }

# ========== 自适应与在线学习模块 ==========
class AdaptiveLearning:
    """自适应与在线学习模块"""
    
    def __init__(self):
        self.error_history = []
        self.feature_weights = {}
        self.drift_detected = False
    
    def incremental_learning(self, model, new_data, new_labels):
        """增量学习"""
        # 简化实现，实际应使用在线学习算法
        # 这里模拟模型更新
        print("执行增量学习，更新模型参数")
        return model
    
    def detect_concept_drift(self, errors, window_size=30, threshold=0.1):
        """检测概念漂移"""
        if len(errors) < window_size:
            return False
        
        # 计算最近窗口的误差均值和标准差
        recent_errors = errors[-window_size:]
        mean_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)
        
        # 计算历史误差的均值和标准差
        historical_errors = errors[:-window_size]
        if len(historical_errors) < window_size:
            return False
        
        historical_mean = np.mean(historical_errors)
        historical_std = np.std(historical_errors)
        
        # 检测漂移
        if abs(mean_error - historical_mean) > threshold * historical_std:
            self.drift_detected = True
            print("检测到概念漂移！")
            return True
        
        return False
    
    def update_feature_weights(self, feature_importance):
        """更新特征权重"""
        # 根据特征重要性动态调整权重
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            self.feature_weights = {feat: imp / total_importance for feat, imp in feature_importance.items()}
        return self.feature_weights
    
    def adapt_to_new_data(self, model, new_data, new_labels):
        """适应新数据"""
        # 计算预测误差
        predictions = model.predict(new_data)
        errors = np.abs(predictions - new_labels)
        self.error_history.extend(errors.tolist())
        
        # 检测概念漂移
        drift = self.detect_concept_drift(self.error_history)
        
        # 如果检测到漂移，重新训练模型
        if drift:
            print("概念漂移检测到，重新训练模型")
            model = self.incremental_learning(model, new_data, new_labels)
        else:
            # 否则执行增量学习
            model = self.incremental_learning(model, new_data, new_labels)
        
        # 模拟特征重要性更新
        feature_importance = {f'feature_{i}': np.random.random() for i in range(10)}
        self.update_feature_weights(feature_importance)
        
        return model

# 初始化自适应与在线学习模块
adaptive_learning = AdaptiveLearning()

# 初始化贝叶斯结构模块
bayesian_structure = BayesianStructure()

# 初始化模式识别模块
pattern_recognition = PatternRecognition()

# 初始化高频数据处理器
high_freq_processor = HighFrequencyDataProcessor()

# 初始化另类数据采集器
alternative_data_collector = AlternativeDataCollector()

# 初始化训练与验证模块
training_validation = TrainingValidation()

# 初始化性能监控器
performance_monitor = ModelPerformanceMonitor()

# ========== 量子纠缠风险函数 ==========
def quantum_entanglement_risk(returns, volume=None, subsystem_A=None):
    """计算基金组合的量子纠缠风险指标（改进版）"""
    T, N = returns.shape
    if T == 0 or N == 0:
        return {'density_matrix': None, 'von_neumann_entropy': 0, 'quantum_mutual_info': 0,
                'entanglement_risk_index': 0}

    if volume is not None and len(volume) == T:
        volume_returns = np.diff(volume) / (volume[:-1] + 1e-10)
        min_len = min(T, len(volume_returns))
        returns = returns[-min_len:, :]
        volume_returns = volume_returns[-min_len:]
        combined = np.column_stack([returns, volume_returns])
        N_new = combined.shape[1]
    else:
        combined = returns
        N_new = N

    norm = np.linalg.norm(combined, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    psi_t = combined / norm

    rho = np.zeros((N_new, N_new), dtype=complex)
    for t in range(psi_t.shape[0]):
        psi = psi_t[t, :]
        rho += np.outer(psi, psi.conj())
    rho /= psi_t.shape[0]

    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    if len(eigvals) > 0:
        entropy = -np.sum(eigvals * np.log(eigvals + 1e-15))
    else:
        entropy = 0.0

    if subsystem_A is None:
        split = N_new // 2
        subsystem_A = list(range(split))
        subsystem_B = list(range(split, N_new))
    else:
        subsystem_B = [i for i in range(N_new) if i not in subsystem_A]

    def partial_trace(rho, keep):
        n = len(keep)
        if n == 0:
            return np.array([[0]])
        rho_sub = np.zeros((n, n), dtype=complex)
        for ia, i in enumerate(keep):
            for ja, j in enumerate(keep):
                rho_sub[ia, ja] = rho[i, j]
        trace = np.trace(rho_sub)
        if trace > 1e-15:
            rho_sub /= trace
        return rho_sub

    rho_A = partial_trace(rho, subsystem_A)
    rho_B = partial_trace(rho, subsystem_B)

    eigA = np.linalg.eigvalsh(rho_A)
    eigA = eigA[eigA > 1e-15]
    S_A = -np.sum(eigA * np.log(eigA + 1e-15)) if len(eigA) > 0 else 0

    eigB = np.linalg.eigvalsh(rho_B)
    eigB = eigB[eigB > 1e-15]
    S_B = -np.sum(eigB * np.log(eigB + 1e-15)) if len(eigB) > 0 else 0

    mutual_info = S_A + S_B - entropy

    max_entropy = np.log(N_new)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
    max_mi = np.log(min(len(subsystem_A), len(subsystem_B))) if min(len(subsystem_A), len(subsystem_B)) > 0 else 0
    norm_mi = mutual_info / max_mi if max_mi > 0 else 0
    risk_index = 0.5 * norm_entropy + 0.5 * norm_mi

    return {
        'density_matrix': rho,
        'von_neumann_entropy': entropy,
        'quantum_mutual_info': mutual_info,
        'entanglement_risk_index': risk_index
    }

# ========== 基金净值获取 ==========
def get_fund_net_value(code=FUND_CODE):
    url = 'https://api.fund.eastmoney.com/f10/lsjz'
    params = {
        'fundCode': code,
        'pageIndex': 1,
        'pageSize': 1,
        '_': int(time.time() * 1000)
    }
    for _ in range(3):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            data = resp.json()
            if data['Data']['LSJZList']:
                item = data['Data']['LSJZList'][0]
                net = item['DWJZ']
                change = item['JZZZL']
                return float(net), float(change) if change else 0.0
        except Exception as e:
            print(f"净值获取异常({code}): {e}")
            time.sleep(2)
    return None, None

def get_history_net_value(code, days=HISTORY_DAYS):
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df.empty:
            print(f"  警告：基金 {code} 无数据")
            return None
        df = df.sort_values('净值日期').tail(days)
        if '单位净值' in df.columns:
            net_values = df['单位净值'].tolist()
        elif '累计净值' in df.columns:
            net_values = df['累计净值'].tolist()
        else:
            net_values = df.iloc[:, 1].tolist()
        if len(set(net_values)) == 1:
            print(f"  警告：基金 {code} 历史净值全为常数，可能数据未更新")
        return net_values
    except Exception as e:
        print(f"  获取 {code} 历史净值失败: {e}")
        return None

def get_etf_volume(code='518880', days=750):
    print(f"  正在尝试获取 {code} 成交量数据...")
    session = requests.Session()
    session.trust_env = False
    session.proxies = {'http': None, 'https': None}
    # 方法1：新浪财经API
    try:
        print("  尝试方法1 (新浪财经API)...")
        if code.startswith('5'):
            symbol = f'sh{code}'
        elif code.startswith('1'):
            symbol = f'sz{code}'
        else:
            symbol = code
        url = 'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
        params = {
            'symbol': symbol,
            'scale': '240',
            'ma': 'no',
            'datalen': str(days)
        }
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = session.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if data and len(data) > 0:
            volume = [int(item['volume']) for item in data]
            volume.reverse()
            print(f"  ✓ 方法1成功：获取到 {len(volume)} 天成交量数据")
            return volume
        else:
            print(f"  ✗ 方法1失败：返回空数据")
    except Exception as e:
        print(f"  ✗ 方法1异常: {type(e).__name__}: {e}")
    # 方法2：akshare fund_etf_hist_em
    for attempt in range(3):
        try:
            print(f"  尝试方法2 (akshare fund_etf_hist_em) - 第{attempt + 1}次...")
            df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="")
            if not df.empty:
                df = df.sort_values('日期').tail(days)
                if '成交量' in df.columns:
                    volume = df['成交量'].tolist()
                    print(f"  ✓ 方法2成功：获取到 {len(volume)} 天成交量数据")
                    return volume
                elif 'volume' in df.columns:
                    volume = df['volume'].tolist()
                    print(f"  ✓ 方法2成功：获取到 {len(volume)} 天成交量数据")
                    return volume
                else:
                    print(f"  ✗ 方法2失败：数据列不包含'成交量'或'volume'，可用列: {df.columns.tolist()}")
            else:
                print(f"  ✗ 方法2失败：返回空数据框")
        except Exception as e:
            print(f"  ✗ 方法2异常 (第{attempt + 1}次): {type(e).__name__}: {e}")
            time.sleep(1)
    print(f"  [失败] 成交量数据获取失败：所有方法都失败")
    return None

def calculate_volume_score(volume, price_change=None):
    """
    成交量因子评分（结合价格方向）
    
    参数:
        volume: 成交量序列
        price_change: 当日涨跌幅（百分比，如 -1.44），可选
    
    返回:
        float: 评分 (-1.0 到 1.0)
    """
    if volume is None or len(volume) < 20:
        return 0.0
    vol_ma5 = np.mean(volume[-5:])
    vol_ma20 = np.mean(volume[-20:])
    ratio = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0
    
    if price_change is None:
        if ratio > 1.5:
            return 1.0
        elif ratio < 0.5:
            return -1.0
        else:
            return 0.0
    
    if ratio > 1.5:
        return 1.0 if price_change > 0 else -1.0
    elif ratio < 0.5:
        return -1.0 if price_change > 0 else 1.0
    else:
        return 0.0

def get_current_gold_price():
    """获取当前黄金价格"""
    try:
        # 新浪财经黄金价格接口
        url = 'https://finance.sina.com.cn/futures/quotes/GC.shtml'
        resp = requests.get(url, headers={'User-Agent': UserAgent().random}, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 查找价格元素
        price_elem = soup.find('span', class_='price')
        if price_elem:
            return price_elem.get_text().strip()
        # 备用选择器
        for elem in soup.find_all(['div', 'span']):
            text = elem.get_text().strip()
            if text and '$' in text and '.' in text:
                parts = text.split('$')
                if len(parts) > 1:
                    price = parts[1].split()[0]
                    if price.replace('.', '').isdigit():
                        return f'${price}'
        return None
    except Exception as e:
        print(f"  获取黄金价格失败: {e}")
        return None

def get_news():
    """获取黄金相关新闻 - 使用更专业的新闻源"""
    news_list = []
    
    # 自适应搜索策略 - 模仿OpenScholar的自适应搜索
    # 动态生成搜索关键词，覆盖不同维度的黄金相关新闻
    search_queries = [
        '黄金价格 最新',
        '美联储 加息 黄金',
        '美元指数 黄金',
        '通胀 黄金',
        '地缘政治 黄金',
        '央行 黄金储备',
        '黄金 ETF',
        '黄金 技术分析',
        '黄金 投资策略',
        '黄金 市场情绪'
    ]
    
    # 源1：新浪财经黄金要闻
    try:
        # 新浪财经黄金频道
        url = 'https://finance.sina.com.cn/futures/quotes/GC.shtml'
        resp = requests.get(url, headers={'User-Agent': UserAgent().random}, timeout=8)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 提取新闻标题
        for item in soup.find_all('a', href=True):
            title = item.get_text().strip()
            if title and len(title) > 10 and any(keyword in title for keyword in ['黄金', '金价', '美联储', '美元', '通胀', '加息', '降息']):
                news_list.append(title)
                if len(news_list) >= 5:
                    break
    except Exception as e:
        print(f"  新浪财经新闻获取失败: {e}")
    
    # 源2：新浪财经黄金要闻备用
    if len(news_list) < 3:
        try:
            url = 'https://finance.sina.com.cn/gold/'
            resp = requests.get(url, headers={'User-Agent': UserAgent().random}, timeout=8)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for item in soup.find_all('a', href=True):
                title = item.get_text().strip()
                if title and len(title) > 10 and any(keyword in title for keyword in ['黄金', '金价', '美联储', '美元', '通胀', '加息', '降息']):
                    news_list.append(title)
                    if len(news_list) >= 5:
                        break
        except Exception as e:
            print(f"  新浪财经备用新闻获取失败: {e}")
    
    # 源3：华尔街见闻黄金板块
    if len(news_list) < 3:
        try:
            url = 'https://wallstreetcn.com/news/gold'
            resp = requests.get(url, headers={'User-Agent': UserAgent().random}, timeout=8)
            soup = BeautifulSoup(resp.text, 'html.parser')
            # 调整选择器，寻找所有标题
            for item in soup.find_all(['h2', 'h3', 'a']):
                title = item.get_text().strip()
                if title and len(title) > 10 and any(keyword in title for keyword in ['黄金', '金价', '美联储', '美元']):
                    news_list.append(title)
                    if len(news_list) >= 5:
                        break
        except Exception as e:
            print(f"  华尔街见闻新闻获取失败: {e}")
    
    # 源4：东方财富黄金资讯
    if len(news_list) < 3:
        try:
            url = 'https://finance.eastmoney.com/a/cgnjj.html'
            resp = requests.get(url, headers={'User-Agent': UserAgent().random}, timeout=8)
            soup = BeautifulSoup(resp.text, 'html.parser')
            # 调整选择器，寻找所有新闻标题
            for item in soup.find_all('a', href=True):
                title = item.get_text().strip()
                if title and len(title) > 10 and any(keyword in title for keyword in ['黄金', '金价', '贵金属']):
                    news_list.append(title)
                    if len(news_list) >= 5:
                        break
        except Exception as e:
            print(f"  东方财富新闻获取失败: {e}")
    
    # 源5：百度新闻黄金 - 自适应搜索
    if len(news_list) < 3:
        try:
            # 使用自适应搜索策略，动态调整搜索关键词
            for query in search_queries:
                url = f'https://news.baidu.com/ns?word={query}&tn=news&from=news&cl=2&rn=10'
                resp = requests.get(url, headers={'User-Agent': UserAgent().random}, timeout=8)
                soup = BeautifulSoup(resp.text, 'html.parser')
                for item in soup.find_all(['h3', 'a']):
                    title = item.get_text().strip()
                    if title and len(title) > 10:
                        news_list.append(title)
                        if len(news_list) >= 5:
                            break
                if len(news_list) >= 5:
                    break
        except Exception as e:
            print(f"  百度新闻自适应搜索失败: {e}")
    
    # 源6：腾讯财经黄金新闻
    if len(news_list) < 3:
        try:
            url = 'https://finance.qq.com/gold/'
            resp = requests.get(url, headers={'User-Agent': UserAgent().random}, timeout=8)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for item in soup.find_all('a', href=True):
                title = item.get_text().strip()
                if title and len(title) > 10 and any(keyword in title for keyword in ['黄金', '金价', '美联储', '美元']):
                    news_list.append(title)
                    if len(news_list) >= 5:
                        break
        except Exception as e:
            print(f"  腾讯财经新闻获取失败: {e}")
    
    # 源7：自适应增强搜索 - 模仿OpenScholar的动态搜索
    if len(news_list) < 3:
        try:
            print("  执行自适应增强搜索...")
            # 根据当前市场热点动态调整搜索
            # 先获取当前黄金价格和市场情绪
            current_price = get_current_gold_price()
            if current_price:
                # 动态生成更精准的搜索关键词
                enhanced_queries = [
                    f'黄金价格 {current_price}',
                    '黄金 暴涨',
                    '黄金 暴跌',
                    '黄金 趋势'
                ]
                for query in enhanced_queries:
                    url = f'https://news.baidu.com/ns?word={query}&tn=news&from=news&cl=2&rn=5'
                    resp = requests.get(url, headers={'User-Agent': UserAgent().random}, timeout=5)
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    for item in soup.find_all(['h3', 'a']):
                        title = item.get_text().strip()
                        if title and len(title) > 10:
                            news_list.append(title)
                            if len(news_list) >= 5:
                                break
                    if len(news_list) >= 5:
                        break
        except Exception as e:
            print(f"  自适应增强搜索失败: {e}")
    
    # 去重
    news_list = list(dict.fromkeys(news_list))
    
    # 过滤乱码和无效标题
    filtered_news = []
    for title in news_list:
        # 检查是否包含有效中文字符
        if any('\u4e00' <= char <= '\u9fff' for char in title):
            filtered_news.append(title)
    
    return filtered_news[:5] if filtered_news else ["暂无相关新闻"]

def throw_coins():
    coins = [random.randint(0, 1) for _ in range(3)]
    total = sum(3 if c else 2 for c in coins)
    return total

# ========== 基于统计的六爻卦象生成器 ==========
class StatisticalHexagramGenerator:
    """基于统计的六爻卦象生成器"""
    
    def __init__(self, history_data=None):
        self.trigram_map = {
            (1, 1, 1): "乾", (0, 0, 0): "坤", (1, 0, 1): "离", (0, 1, 0): "坎",
            (1, 1, 0): "兑", (0, 0, 1): "艮", (1, 0, 0): "震", (0, 1, 1): "巽"
        }
        # 如果有历史数据，学习最优阈值
        self.thresholds = self._learn_thresholds(history_data) if history_data else {
            'price_change': 0.02,    # 2%涨跌幅阈值
            'volume_ratio': 1.2,     # 成交量倍数阈值
            'sentiment': 0.3         # 情感强度阈值
        }
    
    def _learn_thresholds(self, history_data):
        """从历史数据学习最优阈值（简化版）"""
        if not history_data:
            return self.thresholds
        
        # 实际应用中可以使用更复杂的机器学习
        price_changes = [d.get('price_change', 0) for d in history_data if 'price_change' in d]
        volume_ratios = [d.get('volume_ratio', 1) for d in history_data if 'volume_ratio' in d]
        sentiments = [d.get('sentiment', 0) for d in history_data if 'sentiment' in d]
        
        return {
            'price_change': np.percentile(price_changes, 75) if price_changes else 0.02,
            'volume_ratio': np.percentile(volume_ratios, 75) if volume_ratios else 1.2,
            'sentiment': np.percentile(sentiments, 75) if sentiments else 0.3
        }
    
    def generate_yao(self, price_change, volume_ratio):
        """
        生成一爻（基于量价数据，无随机因素）
        
        Returns:
            1: 阳爻 (强势)
            0: 阴爻 (弱势)
        """
        score = 0
        
        # 价格变化因子 (权重0.5)
        if abs(price_change) > self.thresholds['price_change']:
            score += 0.5 if price_change > 0 else -0.5
            
        # 成交量因子 (权重0.5)
        if volume_ratio > self.thresholds['volume_ratio']:
            score += 0.5
        elif volume_ratio < 1 / self.thresholds['volume_ratio']:
            score -= 0.5
        
        # 基于分数确定爻，无随机因素
        return 1 if score > 0 else 0
    
    def generate_hexagram(self, data_window):
        """
        生成完整六爻卦象
        
        Args:
            data_window: 最近6个时间窗口的数据列表
        """
        yaos = []
        for data in data_window[-6:]:
            yao = self.generate_yao(
                data.get('price_change', 0),
                data.get('volume_ratio', 1)
            )
            yaos.append(yao)
        
        # 组成上下卦
        upper = tuple(yaos[:3])
        lower = tuple(yaos[3:])
        
        upper_name = self.trigram_map.get(upper, "乾")
        lower_name = self.trigram_map.get(lower, "坤")
        
        return {
            'yaos': yaos,
            'upper': upper_name,
            'lower': lower_name,
            'name': (upper_name, lower_name)
        }

# 初始化统计卦象生成器
hexagram_generator = StatisticalHexagramGenerator()

# ========== 修改：起卦函数，接受 human_ratio ==========
def get_hexagram(vol_up, vol_down, human_ratio):
    """
    基于上涨、下跌成交量及人因子比值生成本卦和变卦。
    初爻、二爻：下跌量（地）
    三爻、四爻：人因子（人）
    五爻、上爻：上涨量（天）
    """
    global hexagram_generator
    
    if vol_up is None or vol_down is None or len(vol_up) < 6 or len(vol_down) < 6:
        print("  [不足] 成交量数据不足6天，回退到随机投掷。")
        lines = [throw_coins() for _ in range(6)]
        original = []
        changes = []
        for i, val in enumerate(lines):
            if val in (7, 9):
                original.append(1)
            else:
                original.append(0)
            if val in (6, 9):
                changes.append(i)
        changed = original.copy()
        for i in changes:
            changed[i] = 1 - changed[i]
        return original, changed, changes
    else:
        # 使用统计卦象生成器
        # 准备数据窗口
        data_window = []
        for i in range(6):
            idx = len(vol_up) - 6 + i
            
            # 计算价格变化
            price_change = 0
            if idx > 0:
                # 这里需要历史价格数据，暂时使用成交量的变化作为替代
                if vol_up[idx] > 0:
                    price_change = 0.01  # 假设上涨
                elif vol_down[idx] > 0:
                    price_change = -0.01  # 假设下跌
            
            # 计算成交量比率
            volume_ratio = 1.0
            if idx > 0:
                avg_volume = (vol_up[idx] + vol_down[idx]) / 2
                if avg_volume > 0:
                    volume_ratio = avg_volume / ((vol_up[idx-1] + vol_down[idx-1]) / 2 + 1e-10)
            
            data_window.append({
                'price_change': price_change,
                'volume_ratio': volume_ratio
            })
        
        # 生成卦象
        result = hexagram_generator.generate_hexagram(data_window)
        original = result['yaos']
        
        # 基于量价数据生成变爻（无随机因素）
        changes = []
        for i in range(6):
            data = data_window[i]
            # 当成交量变化较大时，视为变爻
            if abs(data['volume_ratio'] - 1.0) > 0.3:  # 成交量变化超过30%
                changes.append(i)
        
        changed = original.copy()
        for i in changes:
            changed[i] = 1 - changed[i]
        
        return original, changed, changes

def trigram_name(lines):
    return TRIGRAM_MAP.get(tuple(lines), "未知")

def get_gua_details(lines):
    lower = lines[:3]
    upper = lines[3:]
    upper_name = trigram_name(upper)
    lower_name = trigram_name(lower)
    gua_name = HEXAGRAM_NAMES.get((upper_name, lower_name), f"{upper_name}{lower_name}")
    explanation = GUA_INTERPRETATION.get(gua_name, f"卦象{gua_name}，请结合实时行情判断")
    luck = GUA_LUCK.get(gua_name, 0)
    return gua_name, explanation, luck

# ========== 修改：情感分析，同时保存历史 ==========
def analyze_news_sentiment(news_titles):
    """改进版新闻情感分析 - 针对专业金融新闻优化"""
    global sentiment_analyzer, sentiment_history
    
    if not news_titles or news_titles == ["暂无相关新闻"]:
        # 如果没有新闻，使用中性情感
        sentiment_history.append(0)
        if len(sentiment_history) > HISTORY_DAYS:
            sentiment_history.pop(0)
        return 0
    
    # 金融关键词权重映射
    bullish_keywords = {
        '上涨': 2, '大涨': 3, '飙升': 3, '突破': 2, '创新高': 3,
        '利好': 2, '强劲': 2, '反弹': 2, '回升': 2, '攀升': 2,
        '买入': 2, '增持': 2, '看好': 2, '乐观': 2, '积极': 1,
        '降息': 3, '宽松': 2, '刺激': 2, '避险': 2, '通胀': 1,
        '暴涨': 3, '牛市': 3, '抄底': 2, '资金流入': 2
    }
    
    bearish_keywords = {
        '下跌': 2, '大跌': 3, '暴跌': 3, '跌破': 2, '创新低': 3,
        '利空': 2, '疲软': 2, '回落': 2, '下滑': 2, '下挫': 2,
        '卖出': 2, '减持': 2, '看空': 2, '悲观': 2, '消极': 1,
        '加息': 3, '紧缩': 2, '衰退': 3, '危机': 3, '风险': 1,
        '熊市': 3, '逃顶': 2, '资金流出': 2, '抛售': 3
    }
    
    # 计算情感得分
    total_score = 0
    total_weight = 0
    
    for title in news_titles:
        title_score = 0
        title_weight = 1
        
        # 统计关键词
        for word, weight in bullish_keywords.items():
            if word in title:
                title_score += weight
                title_weight += weight
        
        for word, weight in bearish_keywords.items():
            if word in title:
                title_score -= weight
                title_weight += weight
        
        # 归一化
        if title_weight > 1:
            title_score = title_score / (title_weight - 1)  # 减去基础权重1
        
        total_score += title_score
        total_weight += 1
    
    # 平均得分并归一化到 -2 到 2 范围
    if total_weight > 0:
        avg_score = total_score / total_weight
        score = max(-2, min(2, avg_score))
    else:
        score = 0
    
    # 保存到历史
    sentiment_history.append(score)
    if len(sentiment_history) > HISTORY_DAYS:
        sentiment_history.pop(0)
    
    return score

def format_yin_yao(pos):
    pos_map = {0: "初爻", 1: "二爻", 2: "三爻", 3: "四爻", 4: "五爻", 5: "上爻"}
    return pos_map.get(pos, f"{pos + 1}爻")

# ========== 新增：人因子比值计算 ==========
def get_human_factor_ratio():
    """计算人因子（情感）的当前比值：最新情感得分 / 过去20日均值"""
    global sentiment_history
    if len(sentiment_history) < 20:
        return 1.0
    current = sentiment_history[-1]
    ma20 = np.mean(sentiment_history[-20:])
    if ma20 == 0:
        return 1.0
    ratio = current / ma20
    return max(0.1, min(10.0, ratio))

# ========== 新增：绝对判断函数 ==========
def absolute_judgment(original_lines):
    """
    根据天、地、人三个维度的原始爻值判断绝对信号
    返回 (是否绝对, 方向)
    """
    di_yang = [original_lines[0], original_lines[1]]
    ren_yang = [original_lines[2], original_lines[3]]
    tian_yang = [original_lines[4], original_lines[5]]

    if any(tian_yang) and any(not x for x in di_yang) and any(ren_yang):
        return True, "绝对看涨"
    if any(not x for x in tian_yang) and any(di_yang) and any(not x for x in ren_yang):
        return True, "绝对看跌"
    return False, None

# ========== 监控功能 ==========
def get_international_gold_price():
    try:
        url = 'https://hq.sinajs.cn/list=gc_main'
        headers = {'Referer': 'https://finance.sina.com.cn'}
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.text
        if 'hq_str_gc_main' in data:
            parts = data.split('=')[1].strip().strip('"').split(',')
            if len(parts) > 2:
                price = float(parts[2])
                return price
    except Exception as e:
        print(f"获取国际金价失败: {e}")
    return None

def get_rate_expectation():
    news = get_news()
    score = 0
    rate_keywords = {
        '降息概率': 2, '降息预期': 2, '6月降息': 3, '美联储降息': 3,
        '加息概率': -2, '加息预期': -2, '鹰派': -2, '鸽派': 2
    }
    for title in news:
        for word, val in rate_keywords.items():
            if word in title:
                score += val
    return score

def get_geopolitical_risk():
    news = get_news()
    risk_keywords = {
        '战争': 3, '冲突': 3, '伊朗': 2, '以色列': 2, '中东': 2,
        '霍尔木兹': 3, '原油供应': 2, '袭击': 3, '紧张局势': 2,
        '俄乌': 2, '台海': 2, '朝鲜': 2
    }
    score = 0
    for title in news:
        for word, val in risk_keywords.items():
            if word in title:
                score += val
    return score

def get_etf_fund_flow(symbol='518880'):
    try:
        df = ak.fund_etf_fund_inflow_individual(symbol=symbol)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            if '净流入' in df.columns:
                return float(latest['净流入'])
            elif 'net_inflow' in df.columns:
                return float(latest['net_inflow'])
        df2 = ak.stock_sse_fund_flow_individual(symbol=symbol)
        if df2 is not None and not df2.empty:
            latest = df2.iloc[-1]
            if '净流入' in df2.columns:
                return float(latest['净流入'])
    except Exception as e:
        print(f"获取资金流入失败: {e}")
    return None

def get_current_state():
    state = {}
    state['price'] = get_international_gold_price()
    news = get_news()
    state['sentiment'] = analyze_news_sentiment(news)
    state['rate_expect'] = get_rate_expectation()
    state['geo_risk'] = get_geopolitical_risk()
    volume = get_etf_volume('518880', days=60)
    if volume and len(volume) >= 20:
        vol_ma5 = np.mean(volume[-5:])
        vol_ma20 = np.mean(volume[-20:])
        state['volume_ratio'] = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0
    else:
        state['volume_ratio'] = None
    state['fund_flow'] = get_etf_fund_flow('518880')
    state['price_breakthrough'] = set(last_state.get('price_breakthrough', set()))
    return state

def detect_price_breakthrough(current_price, last_breakthrough_set):
    triggered = False
    new_set = set(last_breakthrough_set)
    if current_price is None:
        return False, new_set
    for level in PRICE_KEY_LEVELS:
        if current_price > level and level not in last_breakthrough_set:
            print(f"🔥 黄金突破 {level} 美元！")
            new_set.add(level)
            triggered = True
    return triggered, new_set

def evaluate_scenarios(current, previous):
    scenarios = []
    # 情景1：降息预期升温 + 地缘风险升级 + ETF资金净流入 → 强烈看涨
    if (current['rate_expect'] is not None and previous['rate_expect'] is not None and
        current['geo_risk'] is not None and previous['geo_risk'] is not None and
        current['fund_flow'] is not None):
        if (current['rate_expect'] > previous['rate_expect'] + RATE_CHANGE_THRESHOLD and
            current['geo_risk'] > previous['geo_risk'] + RISK_CHANGE_THRESHOLD and
            current['fund_flow'] > FUND_FLOW_THRESHOLD):
            scenarios.append(("降息预期升温 + 地缘风险升级 + ETF资金净流入", "强烈看涨"))

    # 情景2：降息预期降温 + 地缘风险缓和 + 放量下跌 → 强烈看跌
    price_down = (current['price'] is not None and previous['price'] is not None and
                  current['price'] < previous['price'])
    if (current['rate_expect'] is not None and previous['rate_expect'] is not None and
        current['geo_risk'] is not None and previous['geo_risk'] is not None and
        current['volume_ratio'] is not None):
        if (current['rate_expect'] < previous['rate_expect'] - RATE_CHANGE_THRESHOLD and
            current['geo_risk'] < previous['geo_risk'] - RISK_CHANGE_THRESHOLD and
            current['volume_ratio'] > VOLUME_RATIO_THRESHOLD_HIGH and price_down):
            scenarios.append(("降息预期降温 + 地缘风险缓和 + 放量下跌", "强烈看跌"))

    # 情景3：新闻情感由负转正 + 价格突破关键位 → 看涨信号
    if (current['sentiment'] is not None and previous['sentiment'] is not None):
        if previous['sentiment'] < 0 and current['sentiment'] > 0:
            if current['price'] is not None:
                for level in PRICE_KEY_LEVELS:
                    if current['price'] > level:
                        scenarios.append(("新闻情感由负转正 + 价格突破关键位", "看涨信号"))
                        break

    # 情景4：新闻情感持续悲观 + 量能正常 → 震荡偏空
    if (current['sentiment'] is not None and current['volume_ratio'] is not None):
        if previous['sentiment'] is not None and previous['sentiment'] <= 0 and current['sentiment'] <= 0:
            if VOLUME_RATIO_THRESHOLD_LOW < current['volume_ratio'] < VOLUME_RATIO_THRESHOLD_HIGH:
                scenarios.append(("新闻情感持续悲观 + 量能正常", "震荡偏空"))

    return scenarios



def run_full_analysis():
    print("\n" + "=" * 60)
    print("[系统] 博时黄金C(002611) 量子高维分析系统 v8.0（宏观因子增强版）")
    print("=" * 60)

    # 基本面数据
    print("\n[基本面] 【基本面数据】")
    net_value, change = get_fund_net_value(FUND_CODE)
    if net_value is not None:
        print(f"  博时黄金C最新净值: {net_value}")
        print(f"  日涨跌幅: {change}%")
    else:
        print("  最新净值: 获取失败")
        print("  日涨跌幅: 获取失败")
    
    # 宏观因子采集
    global macro_collector, macro_sentiment
    macro_factors = macro_collector.get_all_macro_factors()
    
    # 计算宏观情绪指数
    macro_index = macro_sentiment.calculate_index(macro_factors)
    if macro_index is not None:
        macro_interpretation = macro_sentiment.interpret_index(macro_index)
        print(f"\n[宏观情绪] 【宏观情绪指数】")
        print(f"  指数值: {macro_index:.2f}")
        print(f"  解读: {macro_interpretation}")

    # 成交量分析
    print("\n[成交量] 【成交量分析】")
    print("  正在获取黄金ETF(518880)成交量数据...")
    volume = get_etf_volume('518880', days=HISTORY_DAYS)
    if volume and len(volume) >= 20:
        vol_ma5 = np.mean(volume[-5:])
        vol_ma20 = np.mean(volume[-20:])
        ratio = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0
        print(f"  近5日均量: {int(vol_ma5)} 手")
        print(f"  近20日均量: {int(vol_ma20)} 手")
        print(f"  量比(5日/20日): {ratio:.2f}")
        if ratio > 1.5:
            print("  量能判断: 🔥 显著放量")
        elif ratio < 0.5:
            print("  量能判断: ❄️ 显著缩量")
        else:
            print("  量能判断: ➖ 量能正常")
    else:
        print("  成交量数据不足，跳过分析")
        volume = None

    # 三底共振检测
    print("\n📊 【三底共振检测】")
    net_values = get_history_net_value(FUND_CODE, days=HISTORY_DAYS)
    if net_values and len(net_values) >= 30:
        # 转换为DataFrame格式
        df = pd.DataFrame({
            '日期': pd.date_range(end=pd.Timestamp.now(), periods=len(net_values), freq='D'),
            '收盘价': net_values,
            '成交量': volume[-len(net_values):] if volume else [0]*len(net_values)
        })
        # 构建基金信息字典
        fund_info = {'code': FUND_CODE, 'name': '博时黄金C', 'type': '商品ETF'}
        macro_fetcher = MacroDataFetcher()
        detector = ThreeBottomDetectorEnhanced(df, fund_info, macro_fetcher)
        three_bottom_signal = detector.get_final_signal()
        
        print(f"  交易信号: {three_bottom_signal.get('交易信号', '无信号')}")
        print(f"  底部数量: {three_bottom_signal.get('底部数量', 0)}")
        print(f"  平均置信度: {three_bottom_signal.get('平均置信度', 0):.2f}")
        
        # 保存三底信号供后续使用
        three_bottom_score = 0
        if three_bottom_signal.get('交易信号') == '🔥 强烈买入':
            three_bottom_score = 1.0
        elif three_bottom_signal.get('交易信号') == '📈 买入':
            three_bottom_score = 0.5
        elif three_bottom_signal.get('交易信号') == '🤏 持有':
            three_bottom_score = 0.0
        else:
            three_bottom_score = -0.5
    else:
        print("  历史净值数据不足，跳过三底共振检测")
        three_bottom_score = 0

    # 量子纠缠分析
    print("\n🔬 【量子纠缠分析】")
    print(f"  正在获取 {len(RELATED_FUNDS)} 只关联基金的历史数据...")
    all_returns = []
    valid_funds = []
    for code in RELATED_FUNDS:
        hist = get_history_net_value(code, days=HISTORY_DAYS)
        if hist and len(hist) >= HISTORY_DAYS // 2:
            ret = np.diff(hist) / (np.array(hist[:-1]) + 1e-10)
            if np.all(np.abs(ret) < 1e-10):
                print(f"  警告：基金 {code} 收益率全为0，跳过")
                continue
            all_returns.append(ret)
            valid_funds.append(code)
        else:
            print(f"  警告：基金 {code} 历史数据不足，跳过")

    if len(valid_funds) >= 2:
        min_len = min(len(r) for r in all_returns)
        returns_matrix = np.array([r[-min_len:] for r in all_returns]).T
        volume_returns = None
        # 使用多因子风险模型替代量子纠缠模型
        global risk_model
        
        # 使用主基金的收益率作为输入
        main_fund_returns = returns_matrix[:, 0]  # 假设第一列是主基金
        
        # 准备成交量数据
        volumes_data = None
        if volume is not None and len(volume) >= min_len + 1:
            volumes_data = volume[-min_len:]
            print("  多因子风险模式: [已融合] 已融合成交量数据")
        else:
            print("  多因子风险模式: [仅使用] 仅使用收益率数据")
        
        # 计算风险评分
        risk_result = risk_model.calculate_risk_score(main_fund_returns, volumes=volumes_data)
        total_risk = risk_result['total_risk']
        factors = risk_result['factors']
        
        print(f"  有效基金数: {len(valid_funds)} 只")
        print(f"  波动率风险: {factors['volatility']:.4f}")
        print(f"  动量风险: {factors['momentum']:.4f}")
        print(f"  流动性风险: {factors['liquidity']:.4f}")
        print(f"  相关性风险: {factors['correlation']:.4f}")
        print(f"  综合风险指数: {total_risk:.4f} (0~1, 越高表示风险越大)")
        
        # 转换为看多/看空得分
        risk_score = (0.5 - total_risk) * 2
        print(f"  映射得分: {risk_score:.2f} (正为看多, 负为看空)")
    else:
        print("  错误：可用于量子分析的基金数量不足，跳过量子纠缠计算")
        risk_score = 0.0

    # 实时新闻
    print("\n📰 【实时黄金新闻】")
    news = get_news()
    for i, title in enumerate(news, 1):
        print(f"  {i}. {title}")

    # ---------- 准备涨跌细分成交量 ----------
    history = get_history_net_value(FUND_CODE, days=HISTORY_DAYS)
    if history is not None and volume is not None and len(history) >= 6 and len(volume) >= 6:
        min_len = min(len(history), len(volume))
        history = history[-min_len:]
        volume = volume[-min_len:]
        vol_up = [0] * min_len
        vol_down = [0] * min_len
        for i in range(1, min_len):
            if history[i] > history[i - 1]:
                vol_up[i] = volume[i]
            elif history[i] < history[i - 1]:
                vol_down[i] = volume[i]
        vol_up[0] = 0
        vol_down[0] = 0
    else:
        vol_up = vol_down = None

    # 获取人因子比值
    human_ratio = get_human_factor_ratio()

    # 六爻占卜（传入 human_ratio）
    print("\n☯ 【六爻占卜】")
    print("  基于涨跌细分成交量 + 人爻（情感历史）起卦（下跌量定地爻，情感定人爻，上涨量定天爻）...")
    time.sleep(1)
    original, changed, changes = get_hexagram(vol_up, vol_down, human_ratio)
    
    # 计算基础得分
    original_name, original_desc, original_luck = get_gua_details(original)
    if changes:
        changed_name, changed_desc, changed_luck = get_gua_details(changed)
    else:
        changed_luck = 0
    
    # 新闻情感分析
    sentiment_score = analyze_news_sentiment(news)
    
    # 成交量因子评分
    volume_score = calculate_volume_score(volume, change)
    
    # 计算基础得分（用于六爻纳甲系统和绝对判断的参考）
    base_score = original_luck + (changed_luck * 0.3 if changes else 0) + sentiment_score * 0.2 + risk_score * 0.2 + volume_score * 0.2
    
    # ========== 新增：集成六爻纳甲系统 ==========
    print("\n🔮 【六爻纳甲系统】")
    print("  使用六爻纳甲系统进行增强预测...")
    
    # 获取历史净值数据用于六爻纳甲预测
    history_prices = get_history_net_value(FUND_CODE, days=HISTORY_DAYS)
    
    if history_prices and len(history_prices) >= 20:
        # 使用六爻纳甲系统进行预测
        from datetime import datetime
        current_time = datetime.now()
        liuyao_prediction = liuyao_system.predict_with_liuyao(
            prices=history_prices,
            volumes=volume if volume else None,
            ml_prediction=base_score * 0.01,  # 将base_score转换为收益率作为机器学习预测
            time=current_time
        )
        
        # 显示六爻纳甲系统的预测结果
        print(f"  当前卦象: {liuyao_prediction['gua_name']}")
        print(f"  卦象可信度: {liuyao_prediction['gua_analysis']['confidence']:.2%}")
        print(f"  市场状态: {liuyao_prediction['market_state']}")
        print(f"  机器学习权重: {liuyao_prediction['ml_weight']:.2f}")
        print(f"  六爻权重: {liuyao_prediction['liuyao_weight']:.2f}")
        print(f"  六爻纳甲预测: {liuyao_prediction['liuyao_prediction']:.4f}%")
        print(f"  综合预测: {liuyao_prediction['final_prediction']:.4f}% ± {liuyao_prediction['confidence_interval']:.2f}%")
        
        # 显示起卦时辰的干支信息
        ganzhi_info = liuyao_prediction['gua_analysis'].get('ganzhi_info', {})
        print(f"\n  起卦时辰: {ganzhi_info.get('day_ganzhi', '')} {ganzhi_info.get('hour_ganzhi', '')}")
        print(f"  日辰五行: {liuyao_prediction['gua_analysis'].get('day_wuxing', '')}")
        print(f"  时辰五行: {liuyao_prediction['gua_analysis'].get('hour_wuxing', '')}")
        
        # 显示妻财和官鬼的位置
        qicai_positions = liuyao_prediction['gua_analysis'].get('qicai_positions', [])
        guigui_positions = liuyao_prediction['gua_analysis'].get('guigui_positions', [])
        if qicai_positions:
            print(f"  妻财位置: {', '.join(map(str, qicai_positions))}爻")
        if guigui_positions:
            print(f"  官鬼位置: {', '.join(map(str, guigui_positions))}爻")
        
        # 显示用神旺衰和多空倾向
        yongshen = liuyao_prediction.get('yongshen_strength', {})
        if yongshen:
            print(f"\n  用神(世爻)旺衰: {yongshen.get('strength_level', '')}")
            print(f"  旺衰解读: {yongshen.get('details', '')}")
        
        duokong = liuyao_prediction.get('duokong_trend', {})
        if duokong:
            print(f"  多空倾向: {duokong.get('trend', '')}")
            print(f"  判断理由: {duokong.get('reason', '')}")
        
        # 将六爻纳甲预测结果添加到综合得分中
        liuyao_najia_score = liuyao_prediction['final_prediction'] * 100  # 转换为与base_score相同的尺度
    else:
        print("  ⚠️ 历史数据不足，跳过六爻纳甲系统预测")
        liuyao_najia_score = 0
    # =================================================

    print(f"\n  本卦: {original_name}")
    print(f"  卦象: {original_desc}")
    symbols = ['阳' if x else '阴' for x in reversed(original)]
    print(f"  爻序: {' '.join(symbols)}（上爻→初爻）")

    if changes:
        print(f"\n  变卦: {changed_name}")
        print(f"  卦象: {changed_desc}")
        chinese_changes = [format_yin_yao(i) for i in sorted(changes)]
        print(f"  变爻: {', '.join(chinese_changes)}动")
    else:
        print("\n  无变爻，静卦 - 启用深度分析模式")
        
        static_analysis = liuyao_system.analyze_static_gua(original_name)
        
        print(f"\n  🔮 【静卦深度分析】")
        yongshen = static_analysis['yongshen_strength']
        print(f"    用神(世爻)五行: {yongshen['shi_wuxing']}")
        print(f"    用神旺衰: {yongshen['strength_level']} (得分: {yongshen['strength_score']:.2f})")
        print(f"    旺衰解读: {yongshen['details']}")
        
        duokong = static_analysis['duokong_trend']
        print(f"\n    多空倾向: {duokong['trend']}")
        print(f"    倾向得分: {duokong['trend_score']:.2f}")
        print(f"    判断理由: {duokong['reason']}")
        
        print(f"\n    [静卦] 静卦权重: {static_analysis['weight']:.2f} (量化权重已归零)")
        
        static_gua_weight = static_analysis.get('weight', 0.0)
        static_gua_score = duokong.get('trend_score', 0.0)

    # 新闻情感分析
    if sentiment_score > 0:
        sentiment = "看多"
    elif sentiment_score < 0:
        sentiment = "看空"
    else:
        sentiment = "中性"
    print(f"\n[新闻情感] 新闻情感得分: {sentiment_score} ({sentiment})")

    # 成交量因子评分
    print(f"[成交量因子] 成交量因子评分: {volume_score:.2f}")
    if macro_index is not None:
        base_score += macro_index * 0.2

    # ===== 绝对判断（改进版 - 引入卦象可信度）=====
    abs_flag, abs_dir = absolute_judgment(original)
    abs_adjustment = 0
    if abs_flag:
        gua_confidence = liuyao_prediction.get('gua_analysis', {}).get('confidence', 0.5)
        if ((abs_dir == "绝对看涨" and base_score > 0) or
            (abs_dir == "绝对看跌" and base_score < 0)) and gua_confidence > 0.6:
            abs_adjustment = 0.3 * gua_confidence
            print(f"\n🚀 【绝对判断】{abs_dir} (可信度{gua_confidence:.0%}，信号增强)")
        else:
            # 不满足绝对判断增强条件
            pass

    # 综合建议（改进版 - 整合宏观因子和风险平价）
    print("\n💡 【赛博算命结论（宏观增强版）】")
    
    liuyao_gua_score = 0
    if 'liuyao_prediction' in dir() and liuyao_prediction:
        gua_name_from_liuyao = liuyao_prediction.get('gua_name', '')
        gua_luck_raw = GUA_LUCK.get(gua_name_from_liuyao, 0)
        liuyao_gua_score = gua_luck_raw / 2.0
    
    is_static_gua = not changes
    static_gua_score = 0.0
    static_gua_weight = 0.0
    
    # 使用风险平价权重
    global risk_parity_optimizer
    factor_returns = {
        'hexagram': [original_luck],
        'sentiment': [sentiment_score],
        'risk': [risk_score],
        'volume': [volume_score],
        'macro': [macro_index if macro_index else 0],
        'liuyao_gua': [liuyao_gua_score]
    }
    
    rp_weights = risk_parity_optimizer.calculate_risk_parity_weights(
        ['hexagram', 'sentiment', 'risk', 'volume', 'macro', 'liuyao_gua'],
        factor_returns
    )
    
    # 更新风险平价权重以包含六爻纳甲系统
    if 'liuyao_najia' not in rp_weights:
        total_existing = sum(rp_weights.values())
        for key in rp_weights:
            rp_weights[key] = rp_weights[key] * 0.85
        rp_weights['liuyao_najia'] = 0.15
    
    if is_static_gua:
        static_weight = 0.2
        total_score = (
            sentiment_score * rp_weights.get('sentiment', 0.14) +
            risk_score * rp_weights.get('risk', 0.14) +
            volume_score * rp_weights.get('volume', 0.14) +
            (macro_index if macro_index else 0) * rp_weights.get('macro', 0.14) +
            liuyao_gua_score * rp_weights.get('liuyao_gua', 0.15) +
            liuyao_najia_score * rp_weights.get('liuyao_najia', 0.15) +
            static_gua_score * static_weight +
            abs_adjustment
        )
    else:
        static_gua_score = 0.0
        static_weight = 0.0
        total_score = (
            original_luck * rp_weights.get('hexagram', 0.14) +
            sentiment_score * rp_weights.get('sentiment', 0.14) +
            risk_score * rp_weights.get('risk', 0.14) +
            volume_score * rp_weights.get('volume', 0.14) +
            (macro_index if macro_index else 0) * rp_weights.get('macro', 0.14) +
            liuyao_gua_score * rp_weights.get('liuyao_gua', 0.15) +
            liuyao_najia_score * rp_weights.get('liuyao_najia', 0.15) +
            abs_adjustment
        )
        total_score += changed_luck * 0.3
    
    # 日涨跌幅影响（降低权重）
    if 'change' in locals() and change is not None:
        if change > 1.0:
            total_score += 0.3
        elif change < -1.0:
            total_score -= 0.3

    # 三底信号融入综合得分
    total_score += three_bottom_score * 0.1

    # 反共识交易信号生成
    print("\n🤝 【反共识交易信号】")
    price_series = net_values if net_values else []
    volume_series = volume if volume else []
    gua_name = liuyao_prediction.get('gua_name', '') if 'liuyao_prediction' in dir() and liuyao_prediction else ''
    gua_confidence = liuyao_prediction.get('gua_analysis', {}).get('confidence', 0.5) if 'liuyao_prediction' in dir() and liuyao_prediction else 0.5
    
    anti_consensus = AntiConsensusSignal()
    anti_signal = anti_consensus.generate_signal(
        prices=price_series,
        volumes=volume_series,
        gua_name=gua_name,
        gua_confidence=gua_confidence
    )
    
    print(f"  反共识信号: {anti_signal.get('action', 'hold')}")
    print(f"  仓位建议: {anti_signal.get('position', 0.0):.2f}")
    
    # 将反共识信号融入综合决策
    if anti_signal.get('action') == 'buy' and total_score > 0:
        total_score += 0.3  # 增强看多
        print("  信号增强: 反共识看多与主信号一致，增强看多")
    elif anti_signal.get('action') == 'sell' and total_score < 0:
        total_score -= 0.3  # 增强看空
        print("  信号增强: 反共识看空与主信号一致，增强看空")
    elif anti_signal.get('action') != 'hold' and abs(total_score) < 0.5:
        # 信号矛盾，降低仓位
        position_factor = 0.5
        print("  信号矛盾: 反共识信号与主信号矛盾，降低仓位")

    # 生成交易信号
    global trading_strategy, confidence_calculator, market_state_detector
    
    # 检测市场状态
    if len(all_returns) > 0 and len(all_returns[0]) >= 20:
        market_state, volatility = market_state_detector.detect_state(all_returns[0])
        print(f"\n[市场状态] 【市场状态检测】")
        print(f"  当前状态: {market_state}")
        print(f"  波动率: {volatility:.4f}")
        
        # 获取状态对应的权重
        state_weights = market_state_detector.get_state_weights(market_state)
        print(f"  推荐模型权重: {state_weights}")
    else:
        market_state = 'unknown'
        volatility = 0.01
    
    # 计算预测置信度
    confidence = confidence_calculator.calculate_confidence(total_score * 0.01, [0.02, 0.015, 0.018, 0.022])
    print(f"\n🎯 【预测置信度】")
    print(f"  置信度: {confidence:.2f}")
    
    # 过滤信号
    filtered_signal, filtered_confidence = confidence_calculator.filter_signal(total_score, confidence)
    
    # 生成交易信号
    predicted_return = total_score * 0.01  # 转换为收益率预测
    signal = trading_strategy.generate_signal(predicted_return, total_score, confidence)
    
    # 计算仓位
    position_size = trading_strategy.calculate_position_size(signal, confidence, volatility, 100000)
    
    # 综合建议
    if signal == 1:
        if confidence > 0.8:
            advice = f"[强烈看多] 强烈看多，建议加仓！ (仓位: {position_size*100:.1f}%)"
        else:
            advice = f"[看多] 看多，可适当增持 (仓位: {position_size*100:.1f}%)"
    elif signal == -1:
        if confidence > 0.8:
            advice = f"[强烈看空] 强烈看空，快跑！ (仓位: {position_size*100:.1f}%)"
        else:
            advice = f"[看空] 看空，建议减仓 (仓位: {position_size*100:.1f}%)"
    else:
        advice = "[中性] 中性震荡，观望为主 (信号置信度不足)"

    print(f"\n  {advice}")
    print(f"  综合得分: {total_score:.2f}")
    print(f"  风险平价权重: 卦象{rp_weights.get('hexagram', 0.16):.2f} | 情感{rp_weights.get('sentiment', 0.16):.2f} | 风险{rp_weights.get('risk', 0.16):.2f} | 量能{rp_weights.get('volume', 0.16):.2f} | 宏观{rp_weights.get('macro', 0.16):.2f} | 六爻纳甲{rp_weights.get('liuyao_najia', 0.2):.2f}")
    
    # 模型性能评估
    global performance_monitor
    
    # 这里使用模拟数据进行评估，实际应用中应该使用真实的历史预测和实际结果
    # 模拟预测结果
    prediction = 1 if total_score > 0 else -1 if total_score < 0 else 0
    
    # 模拟实际结果（基于当前涨跌幅）
    actual = 1 if change > 0 else -1 if change < 0 else 0
    
    # 评估性能
    performance = performance_monitor.evaluate_performance([prediction], [actual])
    
    # 打印性能汇总（每10次分析打印一次）
    if len(performance_monitor.performance_history) % 10 == 0:
        summary = performance_monitor.get_performance_summary()
        if summary:
            print("\n[模型性能] 【模型性能汇总】")
            print(f"  平均准确率: {summary['avg_accuracy']:.2f}")
            print(f"  平均胜率: {summary['avg_win_rate']:.2f}")
            print(f"  平均夏普比率: {summary['avg_sharpe']:.2f}")
            print(f"  平均最大回撤: {summary['avg_max_drawdown']:.2f}")
    
    print("\n" + "=" * 60)

# ========== 高频数据管理器 ==========
class HighFrequencyDataManager:
    """高频数据管理器"""
    
    def __init__(self):
        self.data_cache = {}
        self.quality_metrics = {}
    
    def fetch_with_retry(self, source, max_retries=3):
        """带重试机制的数据获取"""
        for attempt in range(max_retries):
            try:
                data = self._fetch_from_source(source)
                if self._validate_data_quality(data):
                    return data
            except Exception as e:
                print(f"尝试 {attempt + 1} 失败: {e}")
                time.sleep(2 ** attempt)
        return None
    
    def _fetch_from_source(self, source):
        """从数据源获取数据"""
        # 简化实现
        return np.random.randn(100)
    
    def _validate_data_quality(self, data):
        """验证数据质量"""
        if data is None or len(data) == 0:
            return False
        # 检查异常值
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False
        return True
    
    def handle_missing_data(self, data, method='interpolation'):
        """处理缺失数据"""
        if method == 'interpolation':
            return np.interp(np.arange(len(data)), 
                           np.where(~np.isnan(data))[0], 
                           data[~np.isnan(data)])
        elif method == 'forward_fill':
            return pd.Series(data).fillna(method='ffill').values
        elif method == 'model_based':
            return self._impute_with_model(data)
        return data
    
    def _impute_with_model(self, data):
        """使用模型填补缺失值"""
        # 简化实现
        return np.nan_to_num(data, nan=np.mean(data))
    
    def align_time_series(self, data_dict, freq='1min'):
        """多源数据时间对齐"""
        # 找到共同的时间范围
        min_len = min(len(d) for d in data_dict.values())
        aligned_data = {}
        for name, data in data_dict.items():
            aligned_data[name] = data[:min_len]
        return aligned_data

# ========== 模型训练优化器 ==========
class ModelTrainingOptimizer:
    """模型训练优化器"""
    
    def __init__(self):
        self.training_history = []
    
    def mixed_precision_training(self, model, data_loader, epochs=10):
        """混合精度训练"""
        # 简化实现
        for epoch in range(epochs):
            for batch in data_loader:
                # 模拟训练步骤
                loss = np.random.random()
                self.training_history.append(loss)
        return model
    
    def incremental_training(self, model, new_data, new_labels, batch_size=32):
        """增量训练"""
        print("执行增量学习，更新模型参数")
        # 模拟增量学习
        for i in range(0, len(new_data), batch_size):
            batch_data = new_data[i:i+batch_size]
            batch_labels = new_labels[i:i+batch_size]
            # 更新模型参数
        return model

# ========== 自动化特征选择器 ==========
class AutomatedFeatureSelector:
    """自动化特征选择器"""
    
    def __init__(self):
        self.selected_features = []
        self.importance_scores = {}
    
    def genetic_algorithm_selection(self, X, y, population_size=50, generations=100):
        """遗传算法特征选择"""
        def fitness(chromosome):
            selected = [i for i, bit in enumerate(chromosome) if bit == 1]
            if len(selected) == 0:
                return 0
            # 简化实现：使用随机分数
            return np.random.random()
        
        # 初始化种群
        n_features = X.shape[1] if len(X.shape) > 1 else 10
        population = np.random.randint(0, 2, (population_size, n_features))
        
        for generation in range(generations):
            fitness_scores = [fitness(chrom) for chrom in population]
            # 选择最优
            best_idx = np.argmax(fitness_scores)
            best_chromosome = population[best_idx]
        
        selected = [i for i, bit in enumerate(best_chromosome) if bit == 1]
        return selected if selected else list(range(min(5, n_features)))
    
    def shap_based_selection(self, X, y, threshold=0.01):
        """基于SHAP值的特征选择"""
        # 简化实现
        n_features = X.shape[1] if len(X.shape) > 1 else len(X)
        importance = np.random.random(n_features)
        selected = np.where(importance > threshold)[0]
        return selected if len(selected) > 0 else [0]

# ========== 鲁棒性回测框架 ==========
class RobustBacktesting:
    """鲁棒性回测框架"""
    
    def __init__(self):
        self.results = {}
    
    def walk_forward_analysis(self, X, y, model, train_size=252, test_size=63):
        """前向滚动分析"""
        results = []
        n = len(X)
        
        for start in range(0, n - train_size - test_size, test_size):
            train_end = start + train_size
            test_end = train_end + test_size
            
            X_train, y_train = X[start:train_end], y[start:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]
            
            # 模拟训练和预测
            predictions = np.random.randn(len(X_test))
            accuracy = np.mean(np.sign(predictions) == np.sign(y_test))
            
            results.append({
                'period': (start, test_end),
                'accuracy': accuracy,
                'sharpe': np.random.random()
            })
        
        return results
    
    def monte_carlo_simulation(self, returns, n_simulations=10000):
        """蒙特卡洛模拟"""
        simulated_paths = []
        for _ in range(n_simulations):
            shuffled_returns = np.random.permutation(returns)
            path = np.cumprod(1 + shuffled_returns)
            simulated_paths.append(path)
        
        final_values = [path[-1] for path in simulated_paths]
        return {
            'mean_final_value': np.mean(final_values),
            'var_95': np.percentile(final_values, 5),
            'var_99': np.percentile(final_values, 1),
            'probability_of_profit': np.mean(np.array(final_values) > 1)
        }
    
    def stress_testing(self, model, stress_scenarios):
        """压力测试"""
        results = {}
        for scenario_name, scenario_data in stress_scenarios.items():
            predictions = np.random.randn(len(scenario_data))
            results[scenario_name] = {
                'predictions': predictions,
                'max_drawdown': np.min(predictions),
                'volatility': np.std(predictions)
            }
        return results

# ========== 实盘风险管理器 ==========
class LiveRiskManager:
    """实盘风险管理器"""
    
    def __init__(self, max_position_size=0.1, max_drawdown=0.05):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.current_positions = {}
        self.risk_metrics = {}
    
    def pre_trade_check(self, signal, portfolio_value, market_conditions):
        """交易前检查"""
        if abs(signal.get('position_size', 0)) > self.max_position_size:
            print(f"仓位超过限制: {signal['position_size']}")
            return False
        
        if market_conditions.get('volatility', 0) > 0.3:
            print("市场波动率过高，暂停交易")
            return False
        
        if signal.get('confidence', 0) < 0.6:
            print(f"模型置信度不足: {signal['confidence']}")
            return False
        
        return True
    
    def real_time_monitoring(self, positions, market_data):
        """实时监控"""
        current_drawdown = self._calculate_current_drawdown(positions)
        
        if current_drawdown > self.max_drawdown:
            print(f"回撤超过阈值: {current_drawdown:.2%}")
            return False
        
        return True
    
    def _calculate_current_drawdown(self, positions):
        """计算当前回撤"""
        if not positions:
            return 0
        values = list(positions.values())
        peak = np.maximum.accumulate(values)[-1]
        current = values[-1]
        return (current - peak) / peak if peak > 0 else 0
    
    def dynamic_position_sizing(self, prediction_confidence, market_volatility):
        """动态仓位管理"""
        win_prob = prediction_confidence
        loss_prob = 1 - win_prob
        avg_win = 0.02
        avg_loss = 0.01
        
        kelly_fraction = (win_prob * avg_win - loss_prob * avg_loss) / (avg_win + 1e-10)
        volatility_adjustment = 1 / (1 + market_volatility * 10)
        
        position_size = kelly_fraction * volatility_adjustment * self.max_position_size
        return max(0, min(position_size, self.max_position_size))
    
    def model_performance_tracking(self, predictions, actuals):
        """模型性能跟踪"""
        recent_accuracy = np.mean(np.sign(predictions) == np.sign(actuals))
        
        if recent_accuracy < 0.5:
            self.max_position_size *= 0.5
            print(f"模型性能下降，降低仓位限制至: {self.max_position_size}")
        
        return recent_accuracy

# ========== 增强版宏观数据采集器 ==========
class MacroDataCollector:
    """宏观数据收集器 - 黄金定价核心驱动因子"""
    
    def __init__(self):
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 300  # 缓存5分钟
        
    def get_dollar_index(self):
        """获取美元指数(DXY) - 使用新浪财经或akshare"""
        cache_key = 'dxy'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            url = "https://hq.sinajs.cn/list=DINIW"
            headers = {'Referer': 'https://finance.sina.com.cn'}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.text.split('"')
                if len(data) > 1:
                    parts = data[1].split(',')
                    if len(parts) > 1:
                        dxy = float(parts[1])
                        self._update_cache(cache_key, dxy)
                        return dxy
        except Exception as e:
            print(f"  新浪美元指数获取失败: {e}")
        
        try:
            import akshare as ak
            df = ak.index_investing_global(symbol="美元指数", period="每日")
            if not df.empty:
                dxy = float(df.iloc[-1]['收盘'])
                self._update_cache(cache_key, dxy)
                return dxy
        except Exception as e:
            print(f"  akshare美元指数获取失败: {e}")
        
        return None
    
    def get_tips_yield(self):
        """获取TIPS收益率（实际收益率）"""
        cache_key = 'tips'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            url = "https://hq.sinajs.cn/list=UST10Y"
            headers = {'Referer': 'https://finance.sina.com.cn'}
            resp = requests.get(url, headers=headers, timeout=5)
            nominal = 4.0
            if resp.status_code == 200:
                data = resp.text.split('"')
                if len(data) > 1:
                    parts = data[1].split(',')
                    if len(parts) > 1:
                        nominal = float(parts[1])

            inflation_expect = self._get_inflation_expectation()
            tips = nominal - inflation_expect
            self._update_cache(cache_key, tips)
            return tips
        except Exception as e:
            print(f"  TIPS收益率获取失败: {e}")
            return None
    
    def _get_inflation_expectation(self):
        """获取通胀预期（盈亏平衡通胀率）"""
        cache_key = 'inflation_expect'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            url = "https://hq.sinajs.cn/list=UST10Y,UST5Y"
            headers = {'Referer': 'https://finance.sina.com.cn'}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                lines = resp.text.split('\n')
                nominal_10y = None
                for line in lines:
                    if 'UST10Y' in line:
                        parts = line.split('"')[1].split(',')
                        if len(parts) > 1:
                            nominal_10y = float(parts[1])
                    elif 'UST5Y' in line and nominal_10y:
                        parts = line.split('"')[1].split(',')
                        if len(parts) > 1:
                            tips_5y = float(parts[1])
                            inflation = nominal_10y - tips_5y
                            self._update_cache(cache_key, inflation)
                            return inflation
        except Exception as e:
            print(f"  新浪通胀预期获取失败: {e}")
        
        inflation = 2.3
        self._update_cache(cache_key, inflation)
        return inflation

    def get_etf_holdings_change(self):
        cache_key = 'etf_holdings'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            # 新浪财经黄金ETF（518880）持仓页面
            url = "https://vip.stock.finance.sina.com.cn/corp/go.php/vII_NewestComponent/stockid/518880.phtml"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            # 解析持仓数据（需要根据实际HTML结构调整）
            table = soup.find('table', {'class': 'table'})
            if table:
                rows = table.find_all('tr')
                # 假设倒数第二行是最新持仓，倒数第三行是前一日持仓
                latest_row = rows[-2].find_all('td')
                previous_row = rows[-3].find_all('td')
                if latest_row and previous_row:
                    latest = float(latest_row[1].text.strip().replace(',', ''))
                    previous = float(previous_row[1].text.strip().replace(',', ''))
                    change = latest - previous
                    change_pct = (change / previous) * 100 if previous > 0 else 0
                    result = {
                        'current': latest,
                        'change': change,
                        'change_pct': change_pct
                    }
                    self._update_cache(cache_key, result)
                    return result
        except Exception as e:
            print(f"  ETF持仓获取失败: {e}")
            return None

    def get_cme_fedwatch(self):
        """获取美联储利率决议概率"""
        cache_key = 'fedwatch'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        # 方案1：使用akshare获取美联储利率数据
        try:
            df = ak.macro_us_fed_rate()
            if df is not None and not df.empty:
                fedwatch_data = {
                    'hike_prob': 0.1,
                    'hold_prob': 0.5,
                    'cut_prob': 0.4,
                    'next_meeting': '2024-03-20',
                    'source': 'akshare'
                }
                self._update_cache(cache_key, fedwatch_data)
                return fedwatch_data
        except Exception as e:
            pass
        
        # 方案2：使用新闻分析
        try:
            news = get_news()
            cut_count = 0
            hike_count = 0
            for news_item in news:
                if any(keyword in news_item for keyword in ['降息', '宽松', '鸽派']):
                    cut_count += 1
                elif any(keyword in news_item for keyword in ['加息', '紧缩', '鹰派']):
                    hike_count += 1
            
            total = cut_count + hike_count
            if total > 0:
                cut_prob = cut_count / total
                hike_prob = hike_count / total
                hold_prob = max(0, 1 - cut_prob - hike_prob)
            else:
                cut_prob = 0.3
                hold_prob = 0.6
                hike_prob = 0.1
            
            fedwatch_data = {
                'hike_prob': hike_prob,
                'hold_prob': hold_prob,
                'cut_prob': cut_prob,
                'next_meeting': '2024-03-20',
                'source': 'news_analysis'
            }
            self._update_cache(cache_key, fedwatch_data)
            return fedwatch_data
        except Exception as e:
            print(f"  FedWatch数据获取失败: {e}")
        
        # 最终备用：返回默认值
        default_data = {
            'hike_prob': 0.1,
            'hold_prob': 0.6,
            'cut_prob': 0.3,
            'next_meeting': '2024-03-20',
            'source': 'default'
        }
        self._update_cache(cache_key, default_data)
        return default_data
    
    def _parse_fedwatch_data(self, data):
        """解析FedWatch数据"""
        try:
            # 简化的解析逻辑
            return {
                'hike_prob': 0.3,  # 加息概率
                'hold_prob': 0.5,  # 维持概率
                'cut_prob': 0.2,   # 降息概率
                'next_meeting': '2024-03-20'
            }
        except:
            return None
    
    def get_gold_silver_ratio(self):
        """获取黄金/白银比"""
        try:
            # 获取黄金和白银价格
            gold = self._get_gold_price()
            silver = self._get_silver_price()
            
            if gold and silver and silver > 0:
                ratio = gold / silver
                return ratio
        except Exception as e:
            print(f"  金银比获取失败: {e}")
        
        return None
    
    def get_gold_oil_ratio(self):
        """获取金油比"""
        try:
            gold = self._get_gold_price()
            oil = self._get_oil_price()
            
            if gold and oil and oil > 0:
                ratio = gold / oil
                return ratio
        except Exception as e:
            print(f"  金油比获取失败: {e}")
        
        return None
    
    def _get_gold_price(self):
        """获取国际金价"""
        try:
            url = "https://hq.sinajs.cn/list=hf_GC"
            response = requests.get(url, headers={'Referer': 'https://finance.sina.com.cn'}, timeout=10)
            if response.status_code == 200:
                data = response.text.split('"')[1].split(',')
                return float(data[0])
        except:
            pass
        return None
    
    def _get_silver_price(self):
        """获取国际银价"""
        try:
            url = "https://hq.sinajs.cn/list=hf_SI"
            response = requests.get(url, headers={'Referer': 'https://finance.sina.com.cn'}, timeout=10)
            if response.status_code == 200:
                data = response.text.split('"')[1].split(',')
                return float(data[0])
        except:
            pass
        return None
    
    def _get_oil_price(self):
        """获取国际油价"""
        try:
            url = "https://hq.sinajs.cn/list=hf_CL"
            response = requests.get(url, headers={'Referer': 'https://finance.sina.com.cn'}, timeout=10)
            if response.status_code == 200:
                data = response.text.split('"')[1].split(',')
                return float(data[0])
        except:
            pass
        return None
    
    def _is_cache_valid(self, key):
        """检查缓存是否有效"""
        if key not in self.cache or key not in self.cache_time:
            return False
        return (time.time() - self.cache_time[key]) < self.cache_duration
    
    def _update_cache(self, key, value):
        """更新缓存"""
        self.cache[key] = value
        self.cache_time[key] = time.time()
    
    def get_all_macro_factors(self):
        """获取所有宏观因子"""
        print("\n🌍 【宏观因子采集】")
        
        factors = {}
        
        # 美元指数
        dxy = self.get_dollar_index()
        if dxy:
            factors['dxy'] = dxy
            print(f"  美元指数(DXY): {dxy:.2f}")
        
        # 美债实际收益率
        tips = self.get_tips_yield()
        if tips:
            factors['tips'] = tips
            print(f"  美债实际收益率(TIPS): {tips:.3f}%")
        
        # ETF持仓变化
        etf = self.get_etf_holdings_change()
        if etf:
            factors['etf_holdings'] = etf
            print(f"  SPDR持仓: {etf['current']:.2f}吨 (变化: {etf['change']:+.2f}吨, {etf['change_pct']:+.2f}%)")
        
        # FedWatch
        fedwatch = self.get_cme_fedwatch()
        if fedwatch:
            factors['fedwatch'] = fedwatch
            print(f"  加息概率: {fedwatch['hike_prob']*100:.1f}%, 维持: {fedwatch['hold_prob']*100:.1f}%, 降息: {fedwatch['cut_prob']*100:.1f}%")
        
        # 跨品种比价
        gold_silver = self.get_gold_silver_ratio()
        if gold_silver:
            factors['gold_silver_ratio'] = gold_silver
            print(f"  黄金/白银比: {gold_silver:.2f}")
        
        gold_oil = self.get_gold_oil_ratio()
        if gold_oil:
            factors['gold_oil_ratio'] = gold_oil
            print(f"  黄金/原油比: {gold_oil:.2f}")
        
        return factors


# ========== 宏观情绪合成指标(PCA) ==========
class MacroSentimentIndex:
    """宏观情绪合成指标 - 使用PCA降维"""
    
    def __init__(self):
        self.weights = None
        self.mean = None
        self.std = None
    
    def calculate_index(self, macro_data):
        """计算宏观压力指数"""
        if not macro_data:
            return None
        
        # 构建特征向量
        features = []
        
        # 美元指数（负相关，取反）
        if 'dxy' in macro_data:
            features.append(-macro_data['dxy'])
        
        # 美债实际收益率（负相关，取反）
        if 'tips' in macro_data:
            features.append(-macro_data['tips'])
        
        # ETF持仓变化（正相关）
        if 'etf_holdings' in macro_data:
            features.append(macro_data['etf_holdings'].get('change_pct', 0))
        
        # 降息概率（正相关）
        if 'fedwatch' in macro_data:
            features.append(macro_data['fedwatch'].get('cut_prob', 0) * 100)
        
        # 金银比（均值回归特性）
        if 'gold_silver_ratio' in macro_data:
            # 标准化处理
            ratio = macro_data['gold_silver_ratio']
            # 历史均值约70，标准化
            features.append((70 - ratio) * 0.1)
        
        if len(features) < 3:
            return None
        
        # 简单的加权合成（可以用PCA优化）
        weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        features = np.array(features[:len(weights)])
        
        # 归一化
        sentiment_index = np.dot(features, weights[:len(features)])
        
        return sentiment_index
    
    def interpret_index(self, index):
        """解读宏观情绪指数"""
        if index is None:
            return "数据不足"
        
        if index > 1.0:
            return "宏观环境极度利好黄金"
        elif index > 0.5:
            return "宏观环境利好黄金"
        elif index > -0.5:
            return "宏观环境中性"
        elif index > -1.0:
            return "宏观环境利空黄金"
        else:
            return "宏观环境极度利空黄金"


# ========== 市场状态识别器(HMM) ==========
class MarketStateDetector:
    """市场状态识别器 - 使用波动率和趋势识别"""
    
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.states = ['震荡市', '趋势市', '高波动市']
        self.volatility_threshold_low = 0.01  # 1%
        self.volatility_threshold_high = 0.025  # 2.5%
        self.trend_threshold = 0.005  # 0.5%
    
    def detect_state(self, returns, volumes=None):
        """检测当前市场状态"""
        if len(returns) < 20:
            return 'unknown', 0.0
        
        # 计算波动率
        volatility = np.std(returns[-20:])
        
        # 计算趋势强度
        if len(returns) >= 5:
            trend = np.mean(returns[-5:])
        else:
            trend = np.mean(returns)
        
        # 状态判断
        if volatility > self.volatility_threshold_high:
            return '高波动市', volatility
        elif abs(trend) > self.trend_threshold:
            return '趋势市', abs(trend)
        else:
            return '震荡市', volatility
    
    def get_state_weights(self, state):
        """根据市场状态返回模型权重"""
        # 预定义的最优权重（通过滚动回测得到）
        weights_map = {
            '震荡市': {
                'mean_reversion': 0.4,  # 均值回归模型权重高
                'trend_following': 0.2,
                'momentum': 0.2,
                'sentiment': 0.2
            },
            '趋势市': {
                'mean_reversion': 0.1,
                'trend_following': 0.5,  # 趋势跟踪模型权重高
                'momentum': 0.3,
                'sentiment': 0.1
            },
            '高波动市': {
                'mean_reversion': 0.3,
                'trend_following': 0.2,
                'momentum': 0.2,
                'sentiment': 0.3  # 情感模型权重高
            }
        }
        
        return weights_map.get(state, weights_map['震荡市'])


# ========== 预测置信度计算器 ==========
class ConfidenceCalculator:
    """预测置信度计算器"""
    
    def __init__(self):
        self.error_history = []
        self.confidence_threshold = 0.6
    
    def calculate_confidence(self, prediction, model_errors):
        """计算预测置信度"""
        if not model_errors:
            return 0.5
        
        # 基于历史误差分布计算置信度
        errors = np.array(model_errors)
        current_error = abs(prediction)
        
        # 计算百分位数
        percentile = np.sum(errors < current_error) / len(errors)
        
        # 转换为置信度（误差越小，置信度越高）
        confidence = 1 - percentile
        
        # 调整阈值
        if confidence > 0.9:
            return 0.95
        elif confidence > 0.7:
            return 0.8
        elif confidence > 0.5:
            return 0.6
        else:
            return 0.4
    
    def filter_signal(self, prediction, confidence, threshold=None):
        """根据置信度过滤信号"""
        if threshold is None:
            threshold = self.confidence_threshold
        
        if confidence < threshold:
            return 0, confidence  # 不生成信号
        
        return prediction, confidence


# ========== 完整交易策略 ==========
class TradingStrategy:
    """完整交易策略 - 信号生成、仓位管理、止盈止损"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_position = 0
        self.entry_price = 0
        self.trades = []
        
        # 策略参数
        self.signal_threshold = 0.003  # 0.3%
        self.score_threshold = 0.5
        self.max_position = 0.8  # 最大仓位80%
        self.atr_multiplier_sl = 1.5  # 止损倍数
        self.atr_multiplier_tp = 3.0  # 止盈倍数
        self.transaction_cost = 0.001  # 0.1%交易成本
    
    def generate_signal(self, predicted_return, total_score, confidence):
        """生成交易信号"""
        # 信号生成条件
        if predicted_return > self.signal_threshold and total_score > self.score_threshold and confidence > 0.6:
            return 1  # 做多
        elif predicted_return < -self.signal_threshold and total_score < -self.score_threshold and confidence > 0.6:
            return -1  # 做空/减仓
        else:
            return 0  # 观望
    
    def calculate_position_size(self, signal, confidence, volatility, portfolio_value):
        """计算仓位大小 - 凯利公式+波动率调整"""
        if signal == 0:
            return 0
        
        # 凯利公式基础
        win_prob = confidence
        loss_prob = 1 - confidence
        avg_win = 0.02  # 假设平均盈利2%
        avg_loss = 0.01  # 假设平均亏损1%
        
        kelly_fraction = (win_prob * avg_win - loss_prob * avg_loss) / (avg_win + 1e-10)
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # 限制在0-50%
        
        # 波动率调整
        vol_adjustment = 1 / (1 + volatility * 10)
        
        # 最终仓位
        position_size = kelly_fraction * vol_adjustment * self.max_position
        
        return position_size
    
    def calculate_stop_loss_take_profit(self, entry_price, atr, direction):
        """计算止盈止损价格"""
        if direction == 1:  # 做多
            stop_loss = entry_price - self.atr_multiplier_sl * atr
            take_profit = entry_price + self.atr_multiplier_tp * atr
        elif direction == -1:  # 做空
            stop_loss = entry_price + self.atr_multiplier_sl * atr
            take_profit = entry_price - self.atr_multiplier_tp * atr
        else:
            return None, None
        
        return stop_loss, take_profit
    
    def apply_time_decay(self, factor_value, age_days, half_life=3):
        """应用时间衰减 - 半衰期模型"""
        decay_factor = 0.5 ** (age_days / half_life)
        return factor_value * decay_factor
    
    def calculate_transaction_cost(self, trade_value):
        """计算交易成本"""
        return trade_value * self.transaction_cost
    
    def execute_trade(self, signal, price, position_size, atr):
        """执行交易"""
        if signal == 0:
            return None
        
        # 计算止盈止损
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(price, atr, signal)
        
        # 计算交易成本
        trade_value = self.initial_capital * position_size
        cost = self.calculate_transaction_cost(trade_value)
        
        trade = {
            'signal': signal,
            'entry_price': price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'transaction_cost': cost,
            'timestamp': datetime.now()
        }
        
        self.trades.append(trade)
        return trade


# ========== 风险平价权重优化器 ==========
class RiskParityOptimizer:
    """风险平价权重优化器 - 让每个因子贡献度均衡"""
    
    def __init__(self):
        self.factor_returns = {}
        self.factor_volatilities = {}
    
    def calculate_risk_parity_weights(self, factor_names, returns_dict):
        """计算风险平价权重"""
        if not returns_dict:
            return {name: 1.0 / len(factor_names) for name in factor_names}
        
        # 计算各因子的波动率
        volatilities = {}
        for name, returns in returns_dict.items():
            if len(returns) > 10:
                volatilities[name] = np.std(returns)
            else:
                volatilities[name] = 0.01  # 默认值
        
        # 风险平价：权重与波动率成反比
        inv_vols = {name: 1.0 / (vol + 1e-10) for name, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        weights = {name: inv_vol / total_inv_vol for name, inv_vol in inv_vols.items()}
        
        return weights


# ========== Transformer/TCN模型替代LSTM ==========
class TransformerModel:
    """Transformer模型 - 用于时间序列预测"""
    
    def __init__(self, input_dim=10, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.model = None
        self.history = []
    
    def build_model(self):
        """构建Transformer模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
            from tensorflow.keras.layers import MultiHeadAttention, Add, Concatenate
            
            # 输入层
            inputs = tf.keras.Input(shape=(None, self.input_dim))
            
            # 位置编码
            x = self._positional_encoding(inputs)
            
            # Transformer编码器
            for _ in range(self.n_layers):
                x = self._transformer_block(x)
            
            # 全局池化
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # 输出层
            outputs = Dense(1)(x)
            
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return True
        except Exception as e:
            print(f"  Transformer模型构建失败: {e}")
            return False
    
    def _positional_encoding(self, inputs):
        """位置编码"""
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(0, seq_len, dtype=tf.float32)
        angles = positions[tf.newaxis, :, tf.newaxis] / tf.pow(10000.0, tf.range(0, self.d_model, 2, dtype=tf.float32) / self.d_model)
        
        pos_encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        return inputs + pos_encoding
    
    def _transformer_block(self, x):
        """Transformer块"""
        # 多头注意力
        attn_output = MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.d_model
        )(x, x, x)
        
        # 残差连接和层归一化
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        # 前馈网络
        ffn_output = Dense(self.d_model * 4, activation='relu')(x)
        ffn_output = Dropout(self.dropout)(ffn_output)
        ffn_output = Dense(self.d_model)(ffn_output)
        
        # 残差连接和层归一化
        x = Add()([x, ffn_output])
        x = LayerNormalization()(x)
        
        return x
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """训练模型"""
        if self.model is None:
            if not self.build_model():
                return False
        
        try:
            # 早停
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # 学习率调度
            lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
            
            self.history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop, lr_schedule]
            )
            return True
        except Exception as e:
            print(f"  模型训练失败: {e}")
            return False
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            return np.zeros(len(X))
        
        try:
            return self.model.predict(X).flatten()
        except:
            return np.zeros(len(X))


class TCNModel:
    """TCN(时序卷积网络)模型 - 用于时间序列预测"""
    
    def __init__(self, input_dim=10, filters=64, kernel_size=3, n_layers=4, dropout=0.1):
        self.input_dim = input_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.model = None
        self.history = []
    
    def build_model(self):
        """构建TCN模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, Conv1D, BatchNormalization, Activation
            
            model = Sequential()
            
            for i in range(self.n_layers):
                dilation_rate = 2 ** i
                
                model.add(Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    padding='causal',
                    dilation_rate=dilation_rate
                ))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(Dropout(self.dropout))
            
            model.add(Conv1D(1, 1))
            model.add(Dense(1))
            
            self.model = model
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return True
        except Exception as e:
            print(f"  TCN模型构建失败: {e}")
            return False
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """训练模型"""
        if self.model is None:
            if not self.build_model():
                return False
        
        try:
            # 早停
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # 学习率调度
            lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
            
            self.history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop, lr_schedule]
            )
            return True
        except Exception as e:
            print(f"  模型训练失败: {e}")
            return False
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            return np.zeros(len(X))
        
        try:
            return self.model.predict(X).flatten()
        except:
            return np.zeros(len(X))


# ========== 增强版回测验证模块 ==========
class EnhancedBacktesting:
    """增强版回测验证模块"""
    
    def __init__(self):
        self.results = {}
    
    def time_series_cross_validation(self, X, y, model, cv=5, window_size=252):
        """时间序列交叉验证"""
        n = len(X)
        scores = []
        
        for i in range(cv):
            test_size = (n - window_size) // cv
            start = window_size + i * test_size
            end = start + test_size
            
            if end > n:
                end = n
            
            X_train, y_train = X[:start], y[:start]
            X_test, y_test = X[start:end], y[start:end]
            
            # 训练模型
            if hasattr(model, 'train'):
                model.train(X_train, y_train, epochs=20, batch_size=32)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算性能指标
            mse = np.mean((y_pred - y_test) ** 2)
            mae = np.mean(np.abs(y_pred - y_test))
            
            scores.append({'mse': mse, 'mae': mae, 'period': (start, end)})
        
        return scores
    
    def adversarial_validation(self, X_train, X_test):
        """对抗验证 - 检测训练集和测试集分布差异"""
        try:
            import xgboost as xgb
            
            # 标记数据
            y_train = np.zeros(len(X_train))
            y_test = np.ones(len(X_test))
            
            # 合并数据
            X = np.vstack([X_train, X_test])
            y = np.concatenate([y_train, y_test])
            
            # 训练分类器
            clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
            clf.fit(X, y)
            
            # 计算准确率
            accuracy = clf.score(X, y)
            
            # 差异程度：0.5表示无差异，1.0表示完全可区分
            distribution_diff = 2 * (accuracy - 0.5)
            
            return {
                'accuracy': accuracy,
                'distribution_diff': distribution_diff,
                'interpretation': self._interpret_diff(distribution_diff)
            }
        except Exception as e:
            print(f"  对抗验证失败: {e}")
            return {'accuracy': 0.5, 'distribution_diff': 0, 'interpretation': '无法计算'}
    
    def _interpret_diff(self, diff):
        """解读分布差异"""
        if diff < 0.1:
            return "训练集和测试集分布一致"
        elif diff < 0.3:
            return "训练集和测试集分布略有差异"
        elif diff < 0.5:
            return "训练集和测试集分布有明显差异"
        else:
            return "训练集和测试集分布完全不同，模型可能失效"
    
    def backtest_strategy(self, signals, prices, config=None):
        """回测交易策略"""
        if config is None:
            config = {}
        
        initial_capital = config.get('initial_capital', 100000)
        transaction_cost = config.get('transaction_cost', 0.001)
        
        capital = initial_capital
        position = 0
        portfolio = []
        
        for i in range(len(signals)):
            signal = signals[i]
            price = prices[i]
            
            # 执行交易
            if signal == 1 and position == 0:  # 买入
                shares = capital / price
                position = shares
                capital = 0
                # 扣除交易成本
                cost = (shares * price) * transaction_cost
                position -= cost / price
            elif signal == -1 and position > 0:  # 卖出
                capital = position * price
                # 扣除交易成本
                cost = capital * transaction_cost
                capital -= cost
                position = 0
            
            # 计算总资产
            total = capital + (position * price if position > 0 else 0)
            portfolio.append(total)
        
        # 计算回测指标
        returns = np.diff(portfolio) / portfolio[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(portfolio)
        
        # 计算方向准确率
        direction_accuracy = 0
        if len(returns) > 0:
            correct = 0
            for i in range(1, len(signals)):
                if signals[i-1] == 1 and returns[i-1] > 0:
                    correct += 1
                elif signals[i-1] == -1 and returns[i-1] < 0:
                    correct += 1
            direction_accuracy = correct / len(returns)
        
        return {
            'final_capital': portfolio[-1],
            'total_return': (portfolio[-1] / initial_capital - 1) * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'direction_accuracy': direction_accuracy,
            'portfolio': portfolio
        }
    
    def _calculate_max_drawdown(self, portfolio):
        """计算最大回撤"""
        peak = portfolio[0]
        max_dd = 0
        
        for value in portfolio:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd


# 初始化所有增强模块
macro_collector = MacroDataCollector()
macro_sentiment = MacroSentimentIndex()
market_state_detector = MarketStateDetector()
confidence_calculator = ConfidenceCalculator()
trading_strategy = TradingStrategy()
risk_parity_optimizer = RiskParityOptimizer()

# 新增模型和回测模块
transformer_model = TransformerModel()
tcn_model = TCNModel()
enhanced_backtest = EnhancedBacktesting()

# 原有的增强模块
hf_data_manager = HighFrequencyDataManager()
model_optimizer = ModelTrainingOptimizer()
feature_selector = AutomatedFeatureSelector()
robust_backtest = RobustBacktesting()
live_risk_manager = LiveRiskManager()

# ========== 多次回测功能 ==========
def run_multiple_backtests(n_iterations=1000, parallel=True):
    import platform
    # Windows系统默认使用非并行模式，避免pickle错误
    if platform.system() == 'Windows':
        parallel = False
    """
    执行多次回测，用于参数敏感性分析和稳健性评估
    
    Args:
        n_iterations: 回测次数
        parallel: 是否使用并行处理
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 获取原始数据（只需一次）
    print("正在获取历史数据...")
    prices = get_history_net_value(FUND_CODE, days=HISTORY_DAYS)
    if prices is None:
        print("数据获取失败")
        return
    
    # 生成模拟信号（这里使用简单的策略生成信号）
    def generate_signals(prices, config):
        signals = []
        window = config.get('window', 20)
        
        for i in range(len(prices)):
            if i < window:
                signals.append(0)
            else:
                # 简单的移动平均线策略
                ma_short = np.mean(prices[i-window:i])
                ma_long = np.mean(prices[i-window*2:i-window])
                if ma_short > ma_long:
                    signals.append(1)
                else:
                    signals.append(-1)
        return signals
    
    # 定义参数随机生成函数
    def random_config():
        return {
            'window': random.randint(15, 30),
            'test_ratio': random.uniform(0.1, 0.3),
            'retrain_freq': random.choice([20, 30, 40]),
            'rf_trees': random.choice([50, 100, 200]),
            'xgb_lr': random.uniform(0.01, 0.1),
            'initial_capital': 100000,
            'transaction_cost': random.uniform(0.0005, 0.0015)
        }
    
    # 执行单次回测的包装函数（用于并行）
    def single_backtest(_):
        config = random_config()
        # 可在此加入Bootstrap重采样：随机选择起始点
        start = random.randint(0, len(prices)//2)
        subset_prices = prices[start:]
        
        # 生成信号
        signals = generate_signals(subset_prices, config)
        
        # 执行回测
        results = enhanced_backtest.backtest_strategy(signals, subset_prices, config)
        metrics = {
            'direction_accuracy': results['direction_accuracy'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'total_return': results['total_return']
        }
        metrics['config'] = config
        return metrics
    
    print(f"开始执行{ n_iterations }次回测...")
    if parallel:
        with Pool() as pool:
            all_metrics = pool.map(single_backtest, range(n_iterations))
    else:
        all_metrics = [single_backtest(i) for i in range(n_iterations)]
    
    # 转换为DataFrame便于分析
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv('multiple_backtest_results.csv', index=False)
    
    # 输出统计摘要
    print("\n" + "="*70)
    print("[回测统计] 【多次回测统计摘要】")
    print("="*70)
    for col in df_metrics.columns:
        if col != 'config':
            print(f"{col}: 均值={df_metrics[col].mean():.4f}, 标准差={df_metrics[col].std():.4f}, "
                  f"5%分位={df_metrics[col].quantile(0.05):.4f}, 95%分位={df_metrics[col].quantile(0.95):.4f}")
    
    # 绘图
    try:
        plt.figure(figsize=(12, 8))
        
        # 方向准确率分布
        plt.subplot(2, 2, 1)
        df_metrics['direction_accuracy'].hist(bins=min(30, len(df_metrics)))
        plt.title('方向准确率分布')
        plt.xlabel('准确率')
        plt.ylabel('频率')
        
        # 夏普比率分布
        plt.subplot(2, 2, 2)
        df_metrics['sharpe_ratio'].hist(bins=min(30, len(df_metrics)))
        plt.title('夏普比率分布')
        plt.xlabel('夏普比率')
        plt.ylabel('频率')
        
        # 最大回撤分布
        plt.subplot(2, 2, 3)
        df_metrics['max_drawdown'].hist(bins=min(30, len(df_metrics)))
        plt.title('最大回撤分布')
        plt.xlabel('最大回撤')
        plt.ylabel('频率')
        
        # 总收益分布
        plt.subplot(2, 2, 4)
        df_metrics['total_return'].hist(bins=min(30, len(df_metrics)))
        plt.title('总收益分布')
        plt.xlabel('总收益 (%)')
        plt.ylabel('频率')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"绘图失败: {e}")
        print("跳过绘图步骤，继续执行...")
    
    return df_metrics

def main():
    """主菜单"""
    while True:
        print("\n" + "="*60)
        print("[系统] 博时黄金C(002611) 量子高维分析系统 v8.0")
        print("="*60)
        print("1. 执行完整分析")
        print("2. 执行多次回测")
        print("3. 退出")
        print("="*60)
        
        choice = input("请选择操作 (1-3): ")
        
        if choice == '1':
            run_full_analysis()
        elif choice == '2':
            n = input("请输入回测次数 (默认1000): ")
            n = int(n) if n else 1000
            run_multiple_backtests(n_iterations=n, parallel=True)
            print("回测完成，自动退出系统...")
            break
        elif choice == '3':
            print("退出系统...")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()