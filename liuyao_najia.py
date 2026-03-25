import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# 天干地支与五行对应关系
TIANGAN_WUXING = {
    '甲': '木', '乙': '木', '丙': '火', '丁': '火', '戊': '土',
    '己': '土', '庚': '金', '辛': '金', '壬': '水', '癸': '水'
}

DIZHI_WUXING = {
    '子': '水', '丑': '土', '寅': '木', '卯': '木', '辰': '土',
    '巳': '火', '午': '火', '未': '土', '申': '金', '酉': '金',
    '戌': '土', '亥': '水'
}

# 八经卦纳甲
NAJIA = {
    '乾': ['甲', '壬'],
    '坤': ['乙', '癸'],
    '震': ['庚'],
    '巽': ['辛'],
    '坎': ['戊'],
    '离': ['己'],
    '艮': ['丙'],
    '兑': ['丁']
}

# 纳支规则（根据卦宫和爻位）
# 乾宫：子寅辰午申戌
# 坎宫：寅辰午申戌子
# 艮宫：辰午申戌子寅
# 震宫：午申戌子寅辰
# 巽宫：丑亥酉未巳卯
# 离宫：卯丑亥酉未巳
# 坤宫：未巳卯丑亥酉
# 兑宫：酉未巳卯丑亥
NAZHI = {
    '乾': ['子', '寅', '辰', '午', '申', '戌'],
    '坎': ['寅', '辰', '午', '申', '戌', '子'],
    '艮': ['辰', '午', '申', '戌', '子', '寅'],
    '震': ['午', '申', '戌', '子', '寅', '辰'],
    '巽': ['丑', '亥', '酉', '未', '巳', '卯'],
    '离': ['卯', '丑', '亥', '酉', '未', '巳'],
    '坤': ['未', '巳', '卯', '丑', '亥', '酉'],
    '兑': ['酉', '未', '巳', '卯', '丑', '亥']
}

# 六亲关系定义
# 以用神五行为中心
LIUQIN = {
    '木': {'生我': '水', '我生': '火', '克我': '金', '我克': '土', '同我': '木'},
    '火': {'生我': '木', '我生': '土', '克我': '水', '我克': '金', '同我': '火'},
    '土': {'生我': '火', '我生': '金', '克我': '木', '我克': '水', '同我': '土'},
    '金': {'生我': '土', '我生': '水', '克我': '火', '我克': '木', '同我': '金'},
    '水': {'生我': '金', '我生': '木', '克我': '土', '我克': '火', '同我': '水'}
}

# 六亲名称
LIUQIN_NAMES = {
    '生我': '父母',
    '我生': '子孙',
    '克我': '官鬼',
    '我克': '妻财',
    '同我': '兄弟'
}

# 世应位置（根据卦在八宫中的序数）
SHIYING_POSITION = {
    1: (6, 3),  # 世爻在6位，应爻在3位
    2: (5, 2),
    3: (4, 1),
    4: (3, 6),
    5: (2, 5),
    6: (1, 4),
    7: (6, 3),  # 游魂卦
    8: (3, 6)   # 归魂卦
}

# 卦宫顺序
GUA_GONG = ['乾', '坎', '艮', '震', '巽', '离', '坤', '兑']

# 卦名与卦宫对应
GUA_TO_GONG = {
    # 乾宫
    '乾为天': '乾', '天风姤': '乾', '天山遁': '乾', '天地否': '乾',
    '风地观': '乾', '山地剥': '乾', '火地晋': '乾', '火天大有': '乾',
    # 坎宫
    '坎为水': '坎', '水泽节': '坎', '水雷屯': '坎', '水火既济': '坎',
    '泽火革': '坎', '雷火丰': '坎', '地火明夷': '坎', '地水师': '坎',
    # 艮宫
    '艮为山': '艮', '山火贲': '艮', '山天大畜': '艮', '山泽损': '艮',
    '火泽睽': '艮', '天泽履': '艮', '风泽中孚': '艮', '风山渐': '艮',
    # 震宫
    '震为雷': '震', '雷地豫': '震', '雷水解': '震', '雷风恒': '震',
    '地风升': '震', '水风井': '震', '泽风大过': '震', '泽雷随': '震',
    # 巽宫
    '巽为风': '巽', '风天小畜': '巽', '风火家人': '巽', '风雷益': '巽',
    '天雷无妄': '巽', '火雷噬嗑': '巽', '山雷颐': '巽', '山风蛊': '巽',
    # 离宫
    '离为火': '离', '火山旅': '离', '火风鼎': '离', '火水未济': '离',
    '山水蒙': '离', '风水涣': '离', '天水讼': '离', '天火同人': '离',
    # 坤宫
    '坤为地': '坤', '地雷复': '坤', '地泽临': '坤', '地天泰': '坤',
    '雷天大壮': '坤', '泽天夬': '坤', '水天需': '坤', '水地比': '坤',
    # 兑宫
    '兑为泽': '兑', '泽水困': '兑', '泽地萃': '兑', '泽山咸': '兑',
    '水山蹇': '兑', '地山谦': '兑', '雷山小过': '兑', '雷泽归妹': '兑'
}

# 保存卦象可信度的文件
GUA_CONFIDENCE_FILE = 'gua_confidence.json'
# 保存详细历史记录的文件
GUA_HISTORY_FILE = 'gua_history.json'

# 历史记录天数
HISTORY_DAYS = 3576

class GuaConfidenceTracker:
    """卦象统计可信度追踪器 - 基于历史数据统计每种卦象的实际表现"""
    
    def __init__(self):
        self.confidence_data = self.load_confidence_data()
        self.history_data = self.load_history_data()
    
    def load_confidence_data(self):
        """加载卦象可信度统计数据"""
        if os.path.exists(GUA_CONFIDENCE_FILE):
            try:
                with open(GUA_CONFIDENCE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载可信度数据失败: {e}")
        # 初始化所有64卦的统计数据
        return self._init_default_confidence()
    
    def load_history_data(self):
        """加载详细历史记录"""
        if os.path.exists(GUA_HISTORY_FILE):
            try:
                with open(GUA_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载历史数据失败: {e}")
        return []
    
    def _init_default_confidence(self):
        """初始化默认可信度数据"""
        default_data = {}
        for gua in GUA_TO_GONG.keys():
            default_data[gua] = {
                'total': 0,           # 总出现次数
                'success': 0,         # 预测成功次数
                'confidence': 0.5,    # 可信度（默认50%）
                'up_count': 0,        # 上涨次数
                'down_count': 0,      # 下跌次数
                'flat_count': 0,      # 平盘次数
                'avg_return': 0.0,    # 平均收益率
                'max_return': 0.0,    # 最大收益
                'min_return': 0.0,    # 最大亏损
                'sharpe_ratio': 0.0,  # 夏普比率（简化版）
                'last_updated': None, # 最后更新时间
                'consecutive_success': 0,  # 连续成功次数
                'consecutive_fail': 0      # 连续失败次数
            }
        return default_data
    
    def save_confidence_data(self):
        """保存卦象可信度数据"""
        try:
            with open(GUA_CONFIDENCE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.confidence_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存可信度数据失败: {e}")
    
    def save_history_data(self):
        """保存历史记录数据"""
        try:
            with open(GUA_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史数据失败: {e}")
    
    def record_prediction(self, gua_name, predicted_direction, actual_return, 
                         market_context=None, notes=None):
        """
        记录一次预测结果
        
        Args:
            gua_name: 卦象名称
            predicted_direction: 预测方向 ('up', 'down', 'neutral')
            actual_return: 实际收益率（百分比）
            market_context: 市场环境信息（可选）
            notes: 备注（可选）
        """
        timestamp = datetime.now().isoformat()
        
        # 判断预测是否成功
        actual_direction = 'up' if actual_return > 0.5 else ('down' if actual_return < -0.5 else 'neutral')
        success = (predicted_direction == actual_direction)
        
        # 记录历史
        history_record = {
            'timestamp': timestamp,
            'gua_name': gua_name,
            'predicted_direction': predicted_direction,
            'actual_direction': actual_direction,
            'actual_return': actual_return,
            'success': success,
            'market_context': market_context or {},
            'notes': notes
        }
        self.history_data.append(history_record)
        
        # 更新统计数据
        self._update_gua_stats(gua_name, success, actual_return)
        
        # 保存数据
        self.save_history_data()
        self.save_confidence_data()
        
        return success
    
    def _update_gua_stats(self, gua_name, success, actual_return):
        """更新卦象统计数据"""
        if gua_name not in self.confidence_data:
            self.confidence_data[gua_name] = self._init_default_confidence()[gua_name]
        
        stats = self.confidence_data[gua_name]
        
        # 添加时间衰减因子
        decay = 0.98  # 数值越接近1，衰减越慢；0.98表示每天遗忘2%的旧记忆
        
        # 更新基本计数（带衰减）
        stats['total'] = stats['total'] * decay + 1
        if success:
            stats['success'] = stats['success'] * decay + 1
            stats['consecutive_success'] += 1
            stats['consecutive_fail'] = 0
        else:
            stats['success'] = stats['success'] * decay
            stats['consecutive_fail'] += 1
            stats['consecutive_success'] = 0
        
        # 更新方向统计（带衰减）
        if actual_return > 0.5:
            stats['up_count'] = stats['up_count'] * decay + 1
        elif actual_return < -0.5:
            stats['down_count'] = stats['down_count'] * decay + 1
        else:
            stats['flat_count'] = stats['flat_count'] * decay + 1
        
        # 更新收益率统计（带衰减）
        n = stats['total']
        old_avg = stats['avg_return']
        stats['avg_return'] = old_avg * decay + actual_return / n
        stats['max_return'] = max(stats['max_return'], actual_return)
        stats['min_return'] = min(stats['min_return'], actual_return)
        
        # 计算可信度（使用衰减后的数据）
        stats['confidence'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0.5
        
        # 更新时间戳
        stats['last_updated'] = datetime.now().isoformat()
    
    def get_gua_confidence(self, gua_name):
        """
        获取指定卦象的可信度信息
        
        Returns:
            dict: 包含可信度和统计信息的字典
        """
        if gua_name not in self.confidence_data:
            return {
                'confidence': 0.5,
                'total': 0,
                'success': 0,
                'reliability': 'unknown'
            }
        
        stats = self.confidence_data[gua_name]
        
        # 判断可靠性等级
        total = stats['total']
        confidence = stats['confidence']
        
        if total < 5:
            reliability = 'insufficient_data'  # 数据不足
        elif total < 20:
            reliability = 'preliminary'  # 初步数据
        elif confidence >= 0.65:
            reliability = 'high'  # 高可信度
        elif confidence >= 0.55:
            reliability = 'medium'  # 中等可信度
        elif confidence >= 0.45:
            reliability = 'neutral'  # 中性
        else:
            reliability = 'low'  # 低可信度（反向指标）
        
        return {
            'confidence': confidence,
            'total': total,
            'success': stats['success'],
            'up_probability': stats['up_count'] / total if total > 0 else 0.33,
            'down_probability': stats['down_count'] / total if total > 0 else 0.33,
            'flat_probability': stats['flat_count'] / total if total > 0 else 0.34,
            'avg_return': stats['avg_return'],
            'max_return': stats['max_return'],
            'min_return': stats['min_return'],
            'consecutive_success': stats['consecutive_success'],
            'consecutive_fail': stats['consecutive_fail'],
            'reliability': reliability,
            'last_updated': stats['last_updated']
        }
    
    def get_top_gua(self, top_n=10, min_samples=5):
        """
        获取表现最好的卦象
        
        Args:
            top_n: 返回前N个
            min_samples: 最小样本数要求
        
        Returns:
            list: 排序后的卦象列表
        """
        qualified_gua = []
        for gua_name, stats in self.confidence_data.items():
            if stats['total'] >= min_samples:
                qualified_gua.append({
                    'gua_name': gua_name,
                    'confidence': stats['confidence'],
                    'total': stats['total'],
                    'success': stats['success'],
                    'avg_return': stats['avg_return']
                })
        
        # 按可信度排序
        qualified_gua.sort(key=lambda x: x['confidence'], reverse=True)
        return qualified_gua[:top_n]
    
    def get_worst_gua(self, top_n=10, min_samples=5):
        """获取表现最差的卦象（可作为反向指标）"""
        qualified_gua = []
        for gua_name, stats in self.confidence_data.items():
            if stats['total'] >= min_samples:
                qualified_gua.append({
                    'gua_name': gua_name,
                    'confidence': stats['confidence'],
                    'total': stats['total'],
                    'success': stats['success'],
                    'avg_return': stats['avg_return']
                })
        
        # 按可信度升序排序
        qualified_gua.sort(key=lambda x: x['confidence'])
        return qualified_gua[:top_n]
    
    def get_confidence_report(self):
        """生成可信度统计报告"""
        total_predictions = len(self.history_data)
        if total_predictions == 0:
            return "暂无预测历史数据"
        
        # 整体统计
        overall_success = sum(1 for h in self.history_data if h['success'])
        overall_accuracy = overall_success / total_predictions
        
        # 按卦象分组统计
        gua_stats = {}
        for record in self.history_data:
            gua = record['gua_name']
            if gua not in gua_stats:
                gua_stats[gua] = {'total': 0, 'success': 0}
            gua_stats[gua]['total'] += 1
            if record['success']:
                gua_stats[gua]['success'] += 1
        
        report = {
            'overall': {
                'total_predictions': total_predictions,
                'overall_accuracy': overall_accuracy,
                'success_count': overall_success
            },
            'gua_performance': gua_stats,
            'top_performers': self.get_top_gua(5),
            'worst_performers': self.get_worst_gua(5)
        }
        
        return report
    
    def export_to_csv(self, filename='gua_confidence_export.csv'):
        """导出统计数据到CSV"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['卦象名称', '总次数', '成功次数', '可信度', '上涨次数', 
                           '下跌次数', '平盘次数', '平均收益', '最大收益', '最大亏损'])
            
            for gua_name, stats in self.confidence_data.items():
                if stats['total'] > 0:
                    writer.writerow([
                        gua_name,
                        stats['total'],
                        stats['success'],
                        f"{stats['confidence']:.2%}",
                        stats['up_count'],
                        stats['down_count'],
                        stats['flat_count'],
                        f"{stats['avg_return']:.4f}%",
                        f"{stats['max_return']:.4f}%",
                        f"{stats['min_return']:.4f}%"
                    ])
        
        print(f"数据已导出到: {filename}")


class LiuYaoNaJia:
    def __init__(self):
        # 初始化卦象可信度追踪器
        self.confidence_tracker = GuaConfidenceTracker()
        # 保持向后兼容
        self.gua_confidence = self.confidence_tracker.confidence_data
    
    def load_gua_confidence(self):
        """加载卦象可信度数据（向后兼容）"""
        return self.confidence_tracker.confidence_data
    
    def save_gua_confidence(self):
        """保存卦象可信度数据（向后兼容）"""
        self.confidence_tracker.save_confidence_data()
    
    def update_gua_confidence(self, gua_name, success):
        """更新卦象可信度（向后兼容，建议使用 record_prediction）"""
        # 简化的更新，不记录详细历史
        if gua_name not in self.confidence_tracker.confidence_data:
            self.confidence_tracker.confidence_data[gua_name] = \
                self.confidence_tracker._init_default_confidence()[gua_name]
        
        stats = self.confidence_tracker.confidence_data[gua_name]
        stats['total'] += 1
        if success:
            stats['success'] += 1
        
        total = stats['total']
        success_count = stats['success']
        stats['confidence'] = success_count / total if total > 0 else 0.5
        stats['last_updated'] = datetime.now().isoformat()
        
        self.confidence_tracker.save_confidence_data()
    
    def record_prediction_result(self, gua_name, predicted_direction, actual_return, 
                                  market_context=None, notes=None):
        """
        记录预测结果并更新可信度（推荐使用的完整方法）
        
        Args:
            gua_name: 卦象名称
            predicted_direction: 预测方向 ('up', 'down', 'neutral')
            actual_return: 实际收益率（百分比）
            market_context: 市场环境信息（可选）
            notes: 备注（可选）
            
        Returns:
            bool: 预测是否成功
        """
        return self.confidence_tracker.record_prediction(
            gua_name, predicted_direction, actual_return, market_context, notes
        )
    
    def get_gua_by_date(self, date=None):
        """根据日期生成卦象"""
        if date is None:
            date = datetime.now()
        
        # 简单的日期到卦象映射（实际应用中可能需要更复杂的算法）
        # 这里使用日期的年、月、日之和取模64（64卦）
        year, month, day = date.year, date.month, date.day
        gua_index = (year + month + day) % 64
        
        # 64卦列表
        all_gua = list(GUA_TO_GONG.keys())
        gua_name = all_gua[gua_index % len(all_gua)]
        
        return gua_name
    
    def get_ganzhi_by_time(self, time=None):
        """获取指定时间的干支"""
        if time is None:
            time = datetime.now()
        
        # 简化的干支计算（实际应用中可能需要更精确的算法）
        # 这里使用时间戳取模的方法
        tiangan_list = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
        dizhi_list = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
        
        # 计算日干
        day_offset = (time - datetime(time.year, 1, 1)).days
        tiangan_day = tiangan_list[(day_offset + 4) % 10]  # 2020年1月1日是甲子日
        dizhi_day = dizhi_list[(day_offset + 4) % 12]
        
        # 计算时干
        hour = time.hour
        tiangan_hour = tiangan_list[((tiangan_list.index(tiangan_day) * 2 + hour // 2) % 10)]
        dizhi_hour = dizhi_list[hour // 2]
        
        return {
            'tiangan_day': tiangan_day,
            'dizhi_day': dizhi_day,
            'tiangan_hour': tiangan_hour,
            'dizhi_hour': dizhi_hour,
            'day_ganzhi': f'{tiangan_day}{dizhi_day}',
            'hour_ganzhi': f'{tiangan_hour}{dizhi_hour}'
        }
    
    def analyze_gua(self, gua_name, time=None):
        """分析卦象"""
        # 获取起卦时间的干支
        ganzhi_info = self.get_ganzhi_by_time(time)
        
        # 获取卦宫
        gong = GUA_TO_GONG.get(gua_name, '乾')
        
        # 确定世应位置
        # 简单实现：根据卦名在卦宫中的位置确定序数
        gong_gua_list = [k for k, v in GUA_TO_GONG.items() if v == gong]
        gua_index = gong_gua_list.index(gua_name) + 1
        shi_position, ying_position = SHIYING_POSITION.get(gua_index, (6, 3))
        
        # 纳甲纳支
        najia = NAJIA.get(gong, ['甲', '壬'])
        nazhi = NAZHI.get(gong, ['子', '寅', '辰', '午', '申', '戌'])
        
        # 定五行
        yao_info = []
        for i, (zhi, position) in enumerate(zip(nazhi, range(1, 7))):
            wuxing = DIZHI_WUXING.get(zhi, '土')
            yao_info.append({
                'position': position,
                'zhi': zhi,
                'wuxing': wuxing,
                'is_shi': position == shi_position,
                'is_ying': position == ying_position
            })
        
        # 确定用神（这里简化为以世爻为用神）
        shi_yao = next((yao for yao in yao_info if yao['is_shi']), yao_info[0])
        shen_wuxing = shi_yao['wuxing']
        
        # 定六亲
        for yao in yao_info:
            yao_wuxing = yao['wuxing']
            for relationship, wuxing in LIUQIN.get(shen_wuxing, {}).items():
                if yao_wuxing == wuxing:
                    yao['liuqin'] = LIUQIN_NAMES.get(relationship, '未知')
                    break
            else:
                yao['liuqin'] = '未知'
        
        # 分析日辰对爻的生克关系
        day_wuxing = TIANGAN_WUXING.get(ganzhi_info['tiangan_day'], '土')
        hour_wuxing = TIANGAN_WUXING.get(ganzhi_info['tiangan_hour'], '土')
        
        for yao in yao_info:
            # 日辰对爻的生克
            yao_wuxing = yao['wuxing']
            
            # 日辰生克关系
            if LIUQIN.get(day_wuxing, {}).get('我生') == yao_wuxing:
                yao['day_relationship'] = '日生'
                yao['day_impact'] = 0.2
            elif LIUQIN.get(day_wuxing, {}).get('克我') == yao_wuxing:
                yao['day_relationship'] = '日克'
                yao['day_impact'] = -0.2
            elif yao_wuxing == day_wuxing:
                yao['day_relationship'] = '日同'
                yao['day_impact'] = 0.1
            else:
                yao['day_relationship'] = '无'
                yao['day_impact'] = 0
            
            # 时辰生克关系
            if LIUQIN.get(hour_wuxing, {}).get('我生') == yao_wuxing:
                yao['hour_relationship'] = '时生'
                yao['hour_impact'] = 0.1
            elif LIUQIN.get(hour_wuxing, {}).get('克我') == yao_wuxing:
                yao['hour_relationship'] = '时克'
                yao['hour_impact'] = -0.1
            elif yao_wuxing == hour_wuxing:
                yao['hour_relationship'] = '时同'
                yao['hour_impact'] = 0.05
            else:
                yao['hour_relationship'] = '无'
                yao['hour_impact'] = 0
        
        # 分析六亲位置，特别是妻财和官鬼
        qicai_positions = [yao['position'] for yao in yao_info if yao['liuqin'] == '妻财']
        guigui_positions = [yao['position'] for yao in yao_info if yao['liuqin'] == '官鬼']
        
        # 获取可信度（使用新的追踪器）
        confidence_info = self.confidence_tracker.get_gua_confidence(gua_name)
        
        return {
            'gua_name': gua_name,
            'gong': gong,
            'shi_position': shi_position,
            'ying_position': ying_position,
            'najia': najia,
            'yao_info': yao_info,
            'shen_wuxing': shen_wuxing,
            'confidence': confidence_info['confidence'],
            'confidence_info': confidence_info,  # 包含详细的可信度信息
            'ganzhi_info': ganzhi_info,  # 起卦时辰的干支信息
            'qicai_positions': qicai_positions,  # 妻财位置
            'guigui_positions': guigui_positions,  # 官鬼位置
            'day_wuxing': day_wuxing,  # 日辰五行
            'hour_wuxing': hour_wuxing  # 时辰五行
        }
    
    def analyze_yongshen_strength(self, gua_analysis):
        """
        分析用神旺衰
        
        基于卦象中各爻的五行生克关系，判断用神（世爻）的强弱
        
        Returns:
            dict: {
                'shi_wuxing': str,      # 世爻五行
                'strength_level': str,   # 旺衰等级: '旺相'/'平相'/'衰相'
                'strength_score': float, # 旺衰得分 (-1到1)
                'details': str           # 详细分析
            }
        """
        yao_info = gua_analysis.get('yao_info', [])
        shen_wuxing = gua_analysis.get('shen_wuxing', '土')
        day_wuxing = gua_analysis.get('day_wuxing', '土')
        hour_wuxing = gua_analysis.get('hour_wuxing', '土')
        
        if not yao_info:
            return {
                'shi_wuxing': shen_wuxing,
                'strength_level': '平相',
                'strength_score': 0.0,
                'details': '卦象信息不完整'
            }
        
        WUXING_SHENG = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
        WUXING_KE = {'木': '土', '火': '金', '土': '水', '金': '木', '水': '火'}
        
        sheng_count = 0
        ke_count = 0
        same_count = 0
        total_impact = 0
        
        for yao in yao_info:
            if yao.get('is_shi'):
                # 计算日辰和时辰对世爻的影响
                total_impact += yao.get('day_impact', 0)
                total_impact += yao.get('hour_impact', 0)
                continue
            
            yao_wuxing = yao.get('wuxing', '土')
            
            if WUXING_SHENG.get(shen_wuxing) == yao_wuxing:
                sheng_count += 1
            elif WUXING_KE.get(shen_wuxing) == yao_wuxing:
                ke_count += 1
            elif yao_wuxing == shen_wuxing:
                same_count += 1
        
        # 基础得分
        if sheng_count >= 2:
            base_score = 0.8
            base_detail = f'世爻{shen_wuxing}得令，{sheng_count}个生化爻，格局旺盛'
        elif ke_count >= 2:
            base_score = -0.8
            base_detail = f'世爻{shen_wuxing}失令，{ke_count}个克泄爻，格局偏弱'
        elif sheng_count > ke_count:
            base_score = 0.3
            base_detail = f'世爻{shen_wuxing}生化稍多，格局中等'
        elif ke_count > sheng_count:
            base_score = -0.3
            base_detail = f'世爻{shen_wuxing}克泄稍多，格局中等'
        else:
            base_score = 0.0
            base_detail = f'世爻{shen_wuxing}生化平衡，格局平稳'
        
        # 加入日辰和时辰的影响
        strength_score = base_score + total_impact
        strength_score = max(-1.0, min(1.0, strength_score))
        
        # 重新判断旺衰等级
        if strength_score > 0.5:
            strength_level = '旺相'
        elif strength_score < -0.5:
            strength_level = '衰相'
        else:
            strength_level = '平相'
        
        # 详细分析
        time_impact_detail = ''
        if total_impact > 0.1:
            time_impact_detail = f'，得日辰时辰生助'
        elif total_impact < -0.1:
            time_impact_detail = f'，受日辰时辰克泄'
        
        details = base_detail + time_impact_detail
        
        return {
            'shi_wuxing': shen_wuxing,
            'strength_level': strength_level,
            'strength_score': strength_score,
            'details': details
        }
    
    def analyze_duokong_trend(self, gua_analysis, yongshen_strength):
        """
        分析多空倾向
        
        结合用神旺衰、卦象吉凶、六亲关系判断市场多空倾向
        
        Args:
            gua_analysis: 卦象分析结果
            yongshen_strength: 用神旺衰分析结果
            
        Returns:
            dict: {
                'trend': str,           # '偏多'/'偏空'/'僵局'/'观望'
                'trend_score': float,   # 倾向得分 (-1到1)
                'weight': float,        # 建议权重 (0到0.3)
                'reason': str           # 判断理由
            }
        """
        gua_name = gua_analysis.get('gua_name', '')
        qicai_positions = gua_analysis.get('qicai_positions', [])
        guigui_positions = gua_analysis.get('guigui_positions', [])
        
        GUA_DUOKONG = {
            '乾为天': 0.8, '天风姤': 0.6, '天山遁': -0.3, '天地否': -0.5,
            '风地观': -0.2, '山地剥': -0.7, '火地晋': 0.3, '火天大有': 0.9,
            '坎为水': -0.3, '水泽节': 0.2, '水雷屯': -0.4, '水火既济': 0.1,
            '泽火革': 0.4, '雷火丰': 0.3, '地火明夷': -0.6, '地水师': -0.2,
            '艮为山': -0.5, '山火贲': 0.1, '山天大畜': 0.4, '山泽损': -0.3,
            '火泽睽': 0.2, '天泽履': 0.5, '风泽中孚': 0.3, '风山渐': 0.1,
            '震为雷': 0.4, '雷地豫': 0.2, '雷水解': -0.1, '雷风恒': 0.5,
            '地风升': 0.3, '水风井': 0.1, '泽风大过': -0.6, '泽雷随': 0.4,
            '巽为风': 0.3, '风天小畜': 0.4, '风火家人': 0.5, '风雷益': 0.6,
            '天雷无妄': -0.2, '火雷噬嗑': 0.2, '山雷颐': -0.4, '山风蛊': -0.5,
            '离为火': 0.5, '火山旅': 0.1, '火风鼎': 0.6, '火水未济': -0.2,
            '山水蒙': -0.4, '风水涣': 0.2, '天水讼': -0.5, '天火同人': 0.5,
            '坤为地': -0.6, '地雷复': 0.3, '地泽临': 0.2, '地天泰': 0.6,
            '雷天大壮': 0.5, '泽天夬': 0.2, '水天需': 0.1, '水地比': -0.3,
            '兑为泽': 0.2, '泽水困': -0.4, '泽地萃': -0.1, '泽山咸': 0.4,
            '水山蹇': -0.5, '地山谦': 0.3, '雷山小过': -0.2, '雷泽归妹': -0.3
        }
        
        base_trend = GUA_DUOKONG.get(gua_name, 0.0)
        
        strength_score = yongshen_strength.get('strength_score', 0.0)
        
        # 分析妻财和官鬼的影响
        qicai_impact = 0
        guigui_impact = 0
        
        # 妻财位置分析（妻财直接对应价格）
        if qicai_positions:
            # 妻财在高位（上爻）为吉
            for pos in qicai_positions:
                if pos >= 5:
                    qicai_impact += 0.3
                elif pos >= 3:
                    qicai_impact += 0.1
        
        # 官鬼位置分析（官鬼为阻力）
        if guigui_positions:
            # 官鬼在高位为阻力
            for pos in guigui_positions:
                if pos >= 5:
                    guigui_impact -= 0.3
                elif pos >= 3:
                    guigui_impact -= 0.1
        
        # 综合得分
        combined_score = base_trend * 0.4 + strength_score * 0.3 + qicai_impact * 0.2 + guigui_impact * 0.1
        
        # 构建理由
        reason_parts = [f'卦象{gua_name}']
        
        if strength_score > 0.3:
            reason_parts.append('用神旺相')
        elif strength_score < -0.3:
            reason_parts.append('用神衰相')
        
        if qicai_impact > 0:
            reason_parts.append('妻财得位')
        
        if guigui_impact < 0:
            reason_parts.append('官鬼为阻')
        
        reason = '，'.join(reason_parts) + '，呈现'
        
        if combined_score > 0.4:
            trend = '偏多'
            trend_score = combined_score
            weight = 0.25
            reason += '明显多头格局'
        elif combined_score < -0.4:
            trend = '偏空'
            trend_score = combined_score
            weight = 0.25
            reason += '明显空头格局'
        elif abs(combined_score) <= 0.15:
            trend = '僵局'
            trend_score = combined_score
            weight = 0.0
            reason = f'卦象{gua_name}多空力量均衡，建议观望'
        else:
            trend = '观望'
            trend_score = combined_score
            weight = 0.1
            reason = f'卦象{gua_name}信号模糊，建议轻仓观望'
        
        return {
            'trend': trend,
            'trend_score': trend_score,
            'weight': weight,
            'reason': reason
        }
    
    def analyze_static_gua(self, gua_name):
        """
        静卦综合分析（无变爻时的深度分析）
        
        Args:
            gua_name: 卦名
            
        Returns:
            dict: 静卦分析结果
        """
        gua_analysis = self.analyze_gua(gua_name)
        
        yongshen_strength = self.analyze_yongshen_strength(gua_analysis)
        
        duokong_trend = self.analyze_duokong_trend(gua_analysis, yongshen_strength)
        
        return {
            'gua_name': gua_name,
            'is_static': True,
            'yongshen_strength': yongshen_strength,
            'duokong_trend': duokong_trend,
            'prediction': 0.0,
            'confidence': 0.3,
            'weight': duokong_trend['weight'],
            'message': duokong_trend['reason']
        }
    
    def calculate_market_state(self, prices):
        """计算市场状态"""
        if len(prices) < 20:
            return '混沌市'
        
        # 计算波动率
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:]) * np.sqrt(252) * 100
        
        # 计算趋势强度
        if len(prices) >= 60:
            ma20 = np.mean(prices[-20:])
            ma60 = np.mean(prices[-60:])
            trend_strength = abs((ma20 - ma60) / ma60) * 100
        else:
            trend_strength = 0
        
        # 判断市场状态
        if volatility < 10 and trend_strength > 1:
            return '趋势市'
        else:
            return '混沌市'
    
    def calculate_weights(self, market_state, recent_performance):
        """根据市场状态和近期表现计算权重"""
        # 基础权重
        base_ml_weight = 0.6
        base_liuyao_weight = 0.4
        
        # 根据市场状态调整
        if market_state == '趋势市':
            ml_weight = base_ml_weight + 0.2
            liuyao_weight = base_liuyao_weight - 0.2
        else:  # 混沌市
            ml_weight = base_ml_weight - 0.1
            liuyao_weight = base_liuyao_weight + 0.1
        
        # 根据近期表现调整
        if recent_performance < 0.5:
            # 机器学习表现不佳，增加六爻权重
            liuyao_weight += 0.1
            ml_weight -= 0.1
        
        # 确保权重在合理范围内
        ml_weight = max(0.3, min(0.8, ml_weight))
        liuyao_weight = max(0.2, min(0.7, liuyao_weight))
        
        # 归一化
        total = ml_weight + liuyao_weight
        ml_weight /= total
        liuyao_weight /= total
        
        return ml_weight, liuyao_weight
    
    def calculate_confidence_interval(self, prediction, volatility):
        """计算置信区间"""
        # 基于波动率计算置信区间
        interval = volatility * 0.02  # 简单的区间计算
        return max(0.05, interval)  # 确保最小区间
    
    def predict_with_liuyao(self, prices, volumes=None, ml_prediction=None, time=None):
        """结合六爻和机器学习进行预测"""
        if time is None:
            time = datetime.now()
        
        # 获取当前卦象
        gua_name = self.get_gua_by_date(time)
        gua_analysis = self.analyze_gua(gua_name, time)
        
        # 计算市场状态
        market_state = self.calculate_market_state(prices)
        
        # 计算近期表现（这里简化为使用最近5次预测的准确率）
        # 实际应用中应该从历史记录中获取
        recent_performance = 0.6  # 假设近期准确率为60%
        
        # 计算权重
        ml_weight, liuyao_weight = self.calculate_weights(market_state, recent_performance)
        
        # 分析用神旺衰
        yongshen_strength = self.analyze_yongshen_strength(gua_analysis)
        
        # 分析多空倾向
        duokong_trend = self.analyze_duokong_trend(gua_analysis, yongshen_strength)
        
        # 六爻预测（结合多空倾向和可信度）
        gua_confidence = gua_analysis['confidence']
        trend_score = duokong_trend['trend_score']
        
        # 综合计算六爻预测值
        base_prediction = trend_score * 0.6
        confidence_adjustment = (gua_confidence - 0.5) * 0.4
        liuyao_prediction = base_prediction + confidence_adjustment
        
        # 集成预测
        if ml_prediction is not None:
            final_prediction = ml_prediction * ml_weight + liuyao_prediction * liuyao_weight
        else:
            final_prediction = liuyao_prediction
        
        # 计算置信区间
        if len(prices) >= 20:
            returns = np.diff(prices[-20:]) / prices[-20:-1]
            volatility = np.std(returns) * np.sqrt(252) * 100
        else:
            volatility = 2.0  # 默认波动率
        
        confidence_interval = self.calculate_confidence_interval(final_prediction, volatility)
        
        return {
            'final_prediction': final_prediction,
            'ml_prediction': ml_prediction,
            'liuyao_prediction': liuyao_prediction,
            'ml_weight': ml_weight,
            'liuyao_weight': liuyao_weight,
            'gua_name': gua_name,
            'gua_analysis': gua_analysis,
            'yongshen_strength': yongshen_strength,
            'duokong_trend': duokong_trend,
            'market_state': market_state,
            'confidence_interval': confidence_interval,
            'time': time.isoformat()
        }

def backtest_gua_confidence(liuyao, historical_data, gua_generator_func):
    """
    回测卦象可信度系统
    
    Args:
        liuyao: LiuYaoNaJia实例
        historical_data: 历史数据列表，每项包含日期和实际收益率
        gua_generator_func: 根据日期生成卦象的函数
    
    Returns:
        dict: 回测结果统计
    """
    print("开始回测卦象可信度系统...")
    
    correct_predictions = 0
    total_predictions = 0
    returns_when_correct = []
    returns_when_wrong = []
    
    for i, data in enumerate(historical_data[:-1]):  # 最后一天没有次日数据
        date = data['date']
        current_price = data['price']
        next_price = historical_data[i + 1]['price']
        
        # 计算实际收益率
        actual_return = (next_price - current_price) / current_price * 100
        
        # 生成卦象
        gua_name = gua_generator_func(date)
        
        # 获取卦象分析
        analysis = liuyao.analyze_gua(gua_name)
        confidence = analysis['confidence']
        
        # 基于可信度生成预测
        if confidence > 0.55:
            predicted_direction = 'up'
        elif confidence < 0.45:
            predicted_direction = 'down'
        else:
            predicted_direction = 'neutral'
        
        # 记录预测结果
        success = liuyao.record_prediction_result(
            gua_name=gua_name,
            predicted_direction=predicted_direction,
            actual_return=actual_return,
            market_context={
                'date': date.strftime('%Y-%m-%d'),
                'price': current_price,
                'confidence': confidence
            }
        )
        
        total_predictions += 1
        if success:
            correct_predictions += 1
            returns_when_correct.append(actual_return)
        else:
            returns_when_wrong.append(actual_return)
        
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(historical_data)} 条数据...")
    
    # 计算回测统计
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_return_correct = np.mean(returns_when_correct) if returns_when_correct else 0
    avg_return_wrong = np.mean(returns_when_wrong) if returns_when_wrong else 0
    
    results = {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'avg_return_when_correct': avg_return_correct,
        'avg_return_when_wrong': avg_return_wrong,
        'return_difference': avg_return_correct - avg_return_wrong
    }
    
    print("\n回测结果:")
    print(f"总预测次数: {total_predictions}")
    print(f"正确次数: {correct_predictions}")
    print(f"准确率: {accuracy:.2%}")
    print(f"预测正确时的平均收益: {avg_return_correct:.4f}%")
    print(f"预测错误时的平均收益: {avg_return_wrong:.4f}%")
    print(f"收益差异: {results['return_difference']:.4f}%")
    
    return results


# 测试六爻纳甲系统
def test_liuyao_system():
    liuyao = LiuYaoNaJia()
    tracker = liuyao.confidence_tracker
    
    print("=" * 60)
    print("六爻纳甲系统测试")
    print("=" * 60)
    
    # 测试获取卦象
    gua_name = liuyao.get_gua_by_date()
    print(f"\n当前卦象: {gua_name}")
    
    # 测试分析卦象
    analysis = liuyao.analyze_gua(gua_name)
    print(f"卦宫: {analysis['gong']}")
    print(f"世爻位置: {analysis['shi_position']}")
    print(f"应爻位置: {analysis['ying_position']}")
    print(f"纳甲: {analysis['najia']}")
    
    # 显示详细的可信度信息
    conf_info = analysis['confidence_info']
    print(f"\n可信度信息:")
    print(f"  可信度: {conf_info['confidence']:.2%}")
    print(f"  历史出现次数: {conf_info['total']}")
    print(f"  成功次数: {conf_info['success']}")
    print(f"  可靠性等级: {conf_info['reliability']}")
    if conf_info['total'] > 0:
        print(f"  上涨概率: {conf_info['up_probability']:.2%}")
        print(f"  下跌概率: {conf_info['down_probability']:.2%}")
        print(f"  平盘概率: {conf_info['flat_probability']:.2%}")
        print(f"  平均收益: {conf_info['avg_return']:.4f}%")
    
    print("\n爻位信息:")
    for yao in analysis['yao_info']:
        print(f"  位置{yao['position']}: 地支={yao['zhi']}, 五行={yao['wuxing']}, "
              f"六亲={yao['liuqin']}, 世爻={yao['is_shi']}, 应爻={yao['is_ying']}")
    
    # 测试市场状态识别
    test_prices = np.random.randn(100) + 10
    market_state = liuyao.calculate_market_state(test_prices)
    print(f"\n市场状态: {market_state}")
    
    # 测试权重计算
    ml_weight, liuyao_weight = liuyao.calculate_weights(market_state, 0.6)
    print(f"机器学习权重: {ml_weight:.2f}, 六爻权重: {liuyao_weight:.2f}")
    
    # 测试预测
    prediction = liuyao.predict_with_liuyao(test_prices, ml_prediction=0.5)
    print(f"\n最终预测: {prediction['final_prediction']:.4f}% ± {prediction['confidence_interval']:.2f}%")
    print(f"机器学习预测: {prediction['ml_prediction']:.4f}%")
    print(f"六爻预测: {prediction['liuyao_prediction']:.4f}%")
    
    # 测试可信度追踪功能
    print("\n" + "=" * 60)
    print("卦象可信度追踪测试")
    print("=" * 60)
    
    # 模拟一些预测记录
    print("\n模拟记录预测结果...")
    test_gua_list = ['乾为天', '坤为地', '离为火', '坎为水', '震为雷']
    
    for i, gua in enumerate(test_gua_list):
        # 模拟不同结果
        predicted = 'up' if i % 2 == 0 else 'down'
        actual_return = np.random.uniform(-2, 3)  # 随机收益率
        
        success = liuyao.record_prediction_result(
            gua_name=gua,
            predicted_direction=predicted,
            actual_return=actual_return,
            market_context={'test': True, 'index': i},
            notes=f"测试记录 {i+1}"
        )
        print(f"  {gua}: 预测={predicted}, 实际收益={actual_return:.2f}%, 成功={success}")
    
    # 显示更新后的可信度
    print("\n更新后的可信度统计:")
    for gua in test_gua_list:
        conf = tracker.get_gua_confidence(gua)
        if conf['total'] > 0:
            print(f"  {gua}: 可信度={conf['confidence']:.2%}, "
                  f"次数={conf['total']}, 可靠性={conf['reliability']}")
    
    # 测试获取表现最好的卦象
    print("\n表现最好的卦象 (Top 3):")
    top_gua = tracker.get_top_gua(top_n=3, min_samples=1)
    for g in top_gua:
        print(f"  {g['gua_name']}: 可信度={g['confidence']:.2%}, "
              f"次数={g['total']}, 平均收益={g['avg_return']:.4f}%")
    
    # 生成可信度报告
    print("\n" + "=" * 60)
    print("可信度统计报告")
    print("=" * 60)
    report = tracker.get_confidence_report()
    if isinstance(report, dict):
        print(f"总预测次数: {report['overall']['total_predictions']}")
        print(f"整体准确率: {report['overall']['overall_accuracy']:.2%}")
    else:
        print(report)
    
    # 导出到CSV
    print("\n导出数据到CSV...")
    tracker.export_to_csv('test_gua_confidence.csv')
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_liuyao_system()