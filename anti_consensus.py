#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反共识交易模块 - Anti-Consensus Trading Module
================================================
将反共识交易逻辑融入量子观察系统，提供基于市场情绪和期权异动的反向交易信号。

核心逻辑：
- 散户极度看多(>0.8) + 看跌期权异动 → 反共识做空
- 散户极度看空(<0.2) + 看涨期权异动 → 反共识做多
- 结合六爻卦象可信度进行信号过滤
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# 配置常量
MAX_POSITION_RATIO = 0.05  # 单仓最大5%
STOP_LOSS_RATIO = 0.05     # 止损比例5%

# 情绪阈值
SENTIMENT_BULLISH_THRESHOLD = 0.8  # 极度看多阈值
SENTIMENT_BEARISH_THRESHOLD = 0.2  # 极度看空阈值
VOLUME_SURGE_THRESHOLD = 1.5       # 成交量激增阈值
OPTION_SURGE_THRESHOLD = 2.0      # 期权异动阈值


class AntiConsensusSignal:
    """反共识交易信号生成器"""
    
    def __init__(self, confidence_tracker=None):
        self.confidence_tracker = confidence_tracker
        self.signal_history = []
        self.history_file = 'anti_consensus_history.json'
        self.load_history()
    
    def load_history(self):
        """加载历史信号记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.signal_history = json.load(f)
            except Exception as e:
                print(f"加载反共识历史失败: {e}")
                self.signal_history = []
    
    def save_history(self):
        """保存历史信号记录"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.signal_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存反共识历史失败: {e}")
    
    def get_retail_sentiment(self, prices, volumes=None):
        """
        计算散户情绪指数
        基于价格动量、波动率和成交量综合计算
        返回: 0-1之间的情绪指数，>0.5表示看多，<0.5表示看空
        """
        if len(prices) < 20:
            return 0.5  # 默认中性
        
        prices = np.array(prices)
        
        # 1. 价格动量 (近5日)
        momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        # 2. 波动率 (近20日)
        returns = np.diff(prices[-20:]) / prices[-20:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.01
        
        # 3. 成交量趋势 (如果有)
        volume_factor = 1.0
        if volumes is not None and len(volumes) >= 20:
            volumes = np.array(volumes)
            vol_ma5 = np.mean(volumes[-5:])
            vol_ma20 = np.mean(volumes[-20:])
            volume_factor = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0
        
        # 综合情绪计算
        # 动量为正且波动大 → 散户可能过度乐观
        # 动量为负且波动大 → 散户可能过度悲观
        momentum_normalized = np.clip(momentum_5 / 0.1, -1, 1)  # 假设10%为显著动量
        volatility_normalized = np.clip(volatility * 10, 0, 1)  # 放大波动率影响
        
        # 情绪 = 0.5 + 动量因子 * 0.3 + 波动因子 * 0.2
        sentiment = 0.5 + momentum_normalized * 0.3 + volatility_normalized * 0.2
        sentiment = np.clip(sentiment, 0, 1)
        
        return float(sentiment)
    
    def get_volume_surge_rate(self, volumes):
        """
        计算成交量激增率
        返回: 相对于历史平均的倍数
        """
        if volumes is None or len(volumes) < 10:
            return 1.0
        
        volumes = np.array(volumes)
        
        # 近5日平均成交量
        avg_5 = np.mean(volumes[-5:])
        # 历史20日平均成交量
        avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else avg_5
        
        if avg_20 == 0:
            return 1.0
        
        surge_rate = avg_5 / avg_20
        return float(surge_rate)
    
    def get_option_surge_indicator(self, prices, volumes=None):
        """
        模拟期权异动指标
        实际应用中应替换为真实期权数据
        基于价格波动模式推断期权异动
        """
        if len(prices) < 20:
            return {'is_put_surge': False, 'is_call_surge': False, 'surge_rate': 1.0}
        
        prices = np.array(prices)
        
        # 计算近期的极端波动
        recent_prices = prices[-10:]
        if len(recent_prices) < 2:
            return {'is_put_surge': False, 'is_call_surge': False, 'surge_rate': 1.0}
        
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # 看跌期权异动特征：大幅下跌后波动加剧
        put_signal = np.any(returns < -0.02) and np.std(returns) > 0.02
        
        # 看涨期权异动特征：大幅上涨后波动加剧
        call_signal = np.any(returns > 0.02) and np.std(returns) > 0.02
        
        # 激增率基于波动
        surge_rate = 1.0 + np.std(returns) * 10 if len(returns) > 0 else 1.0
        
        return {
            'is_put_surge': bool(put_signal),
            'is_call_surge': bool(call_signal),
            'surge_rate': float(surge_rate)
        }
    
    def calculate_volatility_range(self, prices):
        """
        计算价格波动区间
        使用去头去尾的5%分位数方法
        """
        if len(prices) < 10:
            return {'lower': prices[-1] * 0.95, 'upper': prices[-1] * 1.05, 'mid': prices[-1]}
        
        prices = np.array(prices)
        sorted_prices = np.sort(prices[-20:])  # 取最近20日
        
        exclude_count = max(1, int(len(sorted_prices) * 0.05))
        valid_prices = sorted_prices[exclude_count:-exclude_count] if len(sorted_prices) > exclude_count * 2 else sorted_prices
        
        lower = valid_prices[0]
        upper = valid_prices[-1]
        mid = (lower + upper) / 2
        
        return {
            'lower': float(lower),
            'upper': float(upper),
            'mid': float(mid)
        }
    
    def generate_signal(self, prices, volumes=None, gua_name=None, gua_confidence=0.5):
        """
        生成反共识交易信号
        
        参数:
            prices: 价格序列
            volumes: 成交量序列 (可选)
            gua_name: 六爻卦象名称 (可选)
            gua_confidence: 卦象可信度 0-1 (可选)
        
        返回:
            dict: 包含信号类型、仓位、止损止盈等信息
        """
        # 1. 计算各项指标
        sentiment = self.get_retail_sentiment(prices, volumes)
        volume_surge = self.get_volume_surge_rate(volumes)
        option_indicator = self.get_option_surge_indicator(prices, volumes)
        volatility_range = self.calculate_volatility_range(prices)
        
        current_price = prices[-1]
        
        # 2. 基础信号判断
        signal = {
            'action': 'hold',
            'reason': '无有效信号',
            'position': 0.0,
            'price': current_price,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'sentiment': sentiment,
            'volume_surge': volume_surge,
            'option_indicator': option_indicator,
            'volatility_range': volatility_range,
            'gua_name': gua_name,
            'gua_confidence': gua_confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        # 3. 检查是否触发反共识条件
        
        # 条件A: 散户极度看多 + 看跌期权异动 → 做空
        if (sentiment >= SENTIMENT_BULLISH_THRESHOLD and 
            option_indicator['is_put_surge'] and 
            option_indicator['surge_rate'] >= OPTION_SURGE_THRESHOLD):
            
            position_ratio = MAX_POSITION_RATIO
            
            # 如果卦象可信度高，增加仓位
            if gua_confidence > 0.6:
                position_ratio = min(position_ratio * 1.5, MAX_POSITION_RATIO * 2)
            elif gua_confidence < 0.4:
                position_ratio = position_ratio * 0.5
            
            stop_loss = volatility_range['upper'] * (1 + STOP_LOSS_RATIO)
            take_profit = volatility_range['mid']
            
            signal = {
                'action': 'sell',
                'reason': f'反共识做空: 散户极度看多({sentiment:.2f}) + 看跌期权异动',
                'position': position_ratio,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sentiment': sentiment,
                'volume_surge': volume_surge,
                'option_indicator': option_indicator,
                'volatility_range': volatility_range,
                'gua_name': gua_name,
                'gua_confidence': gua_confidence,
                'timestamp': datetime.now().isoformat()
            }
        
        # 条件B: 散户极度看空 + 看涨期权异动 → 做多
        elif (sentiment <= SENTIMENT_BEARISH_THRESHOLD and 
              option_indicator['is_call_surge'] and 
              option_indicator['surge_rate'] >= OPTION_SURGE_THRESHOLD):
            
            position_ratio = MAX_POSITION_RATIO
            
            # 如果卦象可信度高，增加仓位
            if gua_confidence > 0.6:
                position_ratio = min(position_ratio * 1.5, MAX_POSITION_RATIO * 2)
            elif gua_confidence < 0.4:
                position_ratio = position_ratio * 0.5
            
            stop_loss = volatility_range['lower'] * (1 - STOP_LOSS_RATIO)
            take_profit = volatility_range['mid']
            
            signal = {
                'action': 'buy',
                'reason': f'反共识做多: 散户极度看空({sentiment:.2f}) + 看涨期权异动',
                'position': position_ratio,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sentiment': sentiment,
                'volume_surge': volume_surge,
                'option_indicator': option_indicator,
                'volatility_range': volatility_range,
                'gua_name': gua_name,
                'gua_confidence': gua_confidence,
                'timestamp': datetime.now().isoformat()
            }
        
        # 条件C: 成交量未激增，情绪信号无效
        elif volume_surge <= VOLUME_SURGE_THRESHOLD:
            signal['reason'] = f'成交量未激增({volume_surge:.2f})，情绪信号无效'
        
        # 条件D: 情绪未达极端
        elif SENTIMENT_BEARISH_THRESHOLD < sentiment < SENTIMENT_BULLISH_THRESHOLD:
            signal['reason'] = f'散户情绪未达极端({sentiment:.2f})，持仓观望'
        
        # 4. 记录信号
        self.signal_history.append(signal)
        self.save_history()
        
        return signal
    
    def get_signal_stats(self):
        """获取信号统计信息"""
        if not self.signal_history:
            return {'total_signals': 0, 'buy_signals': 0, 'sell_signals': 0, 'hold_signals': 0}
        
        total = len(self.signal_history)
        buy_count = sum(1 for s in self.signal_history if s['action'] == 'buy')
        sell_count = sum(1 for s in self.signal_history if s['action'] == 'sell')
        hold_count = sum(1 for s in self.signal_history if s['action'] == 'hold')
        
        return {
            'total_signals': total,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'buy_ratio': buy_count / total if total > 0 else 0,
            'sell_ratio': sell_count / total if total > 0 else 0
        }


class IntegratedDecisionEngine:
    """
    整合决策引擎
    整合: 六爻卦象 + 反共识信号 + 机器学习预测
    """
    
    def __init__(self, liuyao_system=None, confidence_tracker=None):
        self.liuyao_system = liuyao_system
        self.anti_consensus = AntiConsensusSignal(confidence_tracker)
        self.confidence_tracker = confidence_tracker
    
    def make_decision(self, prices, volumes=None, ml_prediction=None, dates=None):
        """
        综合决策
        
        参数:
            prices: 价格序列
            volumes: 成交量序列
            ml_prediction: 机器学习预测值 (可选)
            dates: 日期序列 (可选)
        
        返回:
            dict: 最终交易决策
        """
        # 1. 六爻卦象分析
        gua_signal = None
        gua_confidence = 0.5
        gua_name = None
        
        if self.liuyao_system is not None:
            gua_name = self.liuyao_system.get_gua_by_date()
            gua_analysis = self.liuyao_system.analyze_gua(gua_name)
            gua_confidence = gua_analysis['confidence']
            
            # 基于卦象可信度的预测
            if gua_confidence > 0.55:
                gua_direction = 'up'
                gua_prediction = (gua_confidence - 0.5) * 2  # 0-10%
            elif gua_confidence < 0.45:
                gua_direction = 'down'
                gua_prediction = (0.5 - gua_confidence) * -2  # -10-0%
            else:
                gua_direction = 'neutral'
                gua_prediction = 0.0
            
            gua_signal = {
                'name': gua_name,
                'direction': gua_direction,
                'prediction': gua_prediction,
                'confidence': gua_confidence
            }
        
        # 2. 反共识信号
        anti_signal = self.anti_consensus.generate_signal(
            prices, volumes, gua_name, gua_confidence
        )
        
        # 3. 机器学习预测
        ml_signal = None
        if ml_prediction is not None:
            ml_direction = 'up' if ml_prediction > 0 else ('down' if ml_prediction < 0 else 'neutral')
            ml_signal = {
                'prediction': ml_prediction,
                'direction': ml_direction
            }
        
        # 4. 综合决策
        # 权重分配
        w_anti = 0.4    # 反共识权重
        w_gua = 0.3     # 六爻权重
        w_ml = 0.3      # 机器学习权重
        
        # 计算综合评分
        score_buy = 0
        score_sell = 0
        
        # 反共识贡献
        if anti_signal['action'] == 'buy':
            score_buy += w_anti * anti_signal['position'] * 10
        elif anti_signal['action'] == 'sell':
            score_sell += w_anti * anti_signal['position'] * 10
        
        # 六爻贡献
        if gua_signal is not None:
            if gua_signal['direction'] == 'up':
                score_buy += w_gua * gua_confidence
            elif gua_signal['direction'] == 'down':
                score_sell += w_gua * gua_confidence
        
        # 机器学习贡献
        if ml_signal is not None:
            if ml_signal['direction'] == 'up':
                score_buy += w_ml * min(abs(ml_prediction) / 5, 1)
            elif ml_signal['direction'] == 'down':
                score_sell += w_ml * min(abs(ml_prediction) / 5, 1)
        
        # 5. 最终决策
        threshold = 0.02  # 决策阈值
        
        if score_buy - score_sell > threshold:
            final_action = 'buy'
            final_reason = f'综合信号: 反共识({anti_signal["action"]}) + 六爻({gua_signal["direction"] if gua_signal else "N/A"}) + ML({ml_signal["direction"] if ml_signal else "N/A"})'
            position = min(MAX_POSITION_RATIO * (score_buy - score_sell) * 5, MAX_POSITION_RATIO * 2)
        elif score_sell - score_buy > threshold:
            final_action = 'sell'
            final_reason = f'综合信号: 反共识({anti_signal["action"]}) + 六爻({gua_signal["direction"] if gua_signal else "N/A"}) + ML({ml_signal["direction"] if ml_signal else "N/A"})'
            position = min(MAX_POSITION_RATIO * (score_sell - score_buy) * 5, MAX_POSITION_RATIO * 2)
        else:
            final_action = 'hold'
            final_reason = '各信号相互抵消，观望为主'
            position = 0
        
        # 计算止损止盈
        volatility_range = anti_signal['volatility_range']
        current_price = prices[-1]
        
        if final_action == 'buy':
            stop_loss = volatility_range['lower'] * (1 - STOP_LOSS_RATIO)
            take_profit = volatility_range['mid']
        elif final_action == 'sell':
            stop_loss = volatility_range['upper'] * (1 + STOP_LOSS_RATIO)
            take_profit = volatility_range['mid']
        else:
            stop_loss = 0
            take_profit = 0
        
        # 构建最终决策
        decision = {
            'action': final_action,
            'reason': final_reason,
            'position': position,
            'price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'anti_consensus_signal': anti_signal,
            'gua_signal': gua_signal,
            'ml_signal': ml_signal,
            'scores': {
                'buy_score': score_buy,
                'sell_score': score_sell
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return decision
    
    def get_decision_report(self):
        """生成决策报告"""
        anti_stats = self.anti_consensus.get_signal_stats()
        
        report = {
            'anti_consensus_stats': anti_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return report


def test_anti_consensus():
    """测试反共识模块"""
    print("=" * 60)
    print("反共识交易模块测试")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    base_price = 2.0
    prices = []
    volumes = []
    
    for i in range(100):
        change = np.random.randn() * 0.01
        base_price *= (1 + change)
        prices.append(base_price)
        volumes.append(int(1000000 * (1 + np.random.randn() * 0.3)))
    
    prices = np.array(prices)
    volumes = np.array(volumes)
    
    # 测试反共识信号
    print("\n1. 测试反共识信号生成器")
    print("-" * 40)
    
    signal_generator = AntiConsensusSignal()
    
    sentiment = signal_generator.get_retail_sentiment(prices, volumes)
    print(f"散户情绪指数: {sentiment:.4f}")
    
    volume_surge = signal_generator.get_volume_surge_rate(volumes)
    print(f"成交量激增率: {volume_surge:.4f}")
    
    option_indicator = signal_generator.get_option_surge_indicator(prices, volumes)
    print(f"期权异动指标: {option_indicator}")
    
    volatility_range = signal_generator.calculate_volatility_range(prices)
    print(f"波动区间: 下界={volatility_range['lower']:.4f}, 上界={volatility_range['upper']:.4f}, 中位={volatility_range['mid']:.4f}")
    
    # 生成信号
    signal = signal_generator.generate_signal(prices, volumes)
    print(f"\n交易信号:")
    print(f"  操作: {signal['action']}")
    print(f"  理由: {signal['reason']}")
    print(f"  仓位: {signal['position']*100:.2f}%")
    print(f"  价格: {signal['price']:.4f}")
    print(f"  止损: {signal['stop_loss']:.4f}")
    print(f"  止盈: {signal['take_profit']:.4f}")
    
    # 测试整合决策引擎
    print("\n2. 测试整合决策引擎")
    print("-" * 40)
    
    # 模拟机器学习预测
    ml_prediction = 0.5  # 预测上涨0.5%
    
    engine = IntegratedDecisionEngine()
    decision = engine.make_decision(prices, volumes, ml_prediction)
    
    print(f"最终决策:")
    print(f"  操作: {decision['action']}")
    print(f"  理由: {decision['reason']}")
    print(f"  仓位: {decision['position']*100:.2f}%")
    print(f"  买入评分: {decision['scores']['buy_score']:.4f}")
    print(f"  卖出评分: {decision['scores']['sell_score']:.4f}")
    
    if decision['stop_loss'] > 0:
        print(f"  止损: {decision['stop_loss']:.4f}")
    if decision['take_profit'] > 0:
        print(f"  止盈: {decision['take_profit']:.4f}")
    
    # 显示信号统计
    print("\n3. 信号统计")
    print("-" * 40)
    stats = signal_generator.get_signal_stats()
    print(f"总信号数: {stats['total_signals']}")
    print(f"买入信号: {stats['buy_signals']}")
    print(f"卖出信号: {stats['sell_signals']}")
    print(f"观望信号: {stats['hold_signals']}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_anti_consensus()
