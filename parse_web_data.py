import json
import re

js_file = r'C:\Users\Administrator\Downloads\量子观察系统\web\博时黄金ETF联接C(002611)基金净值_估值_行情走势—天天基金网_files\002611.js.下载'

with open(js_file, 'r', encoding='utf-8') as f:
    content = f.read()

print("=" * 60)
print("解析 web 目录下的 002611.js 文件")
print("=" * 60)

match = re.search(r'var\s+Data_netWorthTrend\s*=\s*(\[[\s\S]*?\]);', content)
if match:
    try:
        data_str = match.group(1)
        data = json.loads(data_str)
        
        print(f"\n共有 {len(data)} 条净值记录")
        print("\n最近10条净值数据:")
        print("-" * 60)
        for item in data[-10:]:
            timestamp = item['x']
            net_value = item['y']
            equity_return = item.get('equityReturn', 0)
            
            from datetime import datetime
            date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
            print(f"日期: {date}, 净值: {net_value:.4f}, 涨跌幅: {equity_return}%")
        
        latest = data[-1]
        date = datetime.fromtimestamp(latest['x'] / 1000).strftime('%Y-%m-%d')
        print("\n" + "=" * 60)
        print(f"最新数据 (来自JS文件):")
        print(f"  日期: {date}")
        print(f"  净值: {latest['y']:.4f}")
        print(f"  涨跌幅: {latest.get('equityReturn', 0)}%")
        print("=" * 60)
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
else:
    print("未找到 Data_netWorthTrend 变量")

match_ac = re.search(r'var\s+Data_acWorthTrend\s*=\s*(\[[\s\S]*?\]);', content)
if match_ac:
    try:
        data_str = match_ac.group(1)
        data = json.loads(data_str)
        print(f"\n累计净值记录数: {len(data)}")
        if data:
            latest = data[-1]
            from datetime import datetime
            date = datetime.fromtimestamp(latest[0] / 1000).strftime('%Y-%m-%d')
            print(f"最新累计净值日期: {date}, 值: {latest[1]:.4f}")
    except:
        pass

match_grandTotal = re.search(r'var\s+Data_grandTotal\s*=\s*(\[[\s\S]*?\]);', content)
if match_grandTotal:
    try:
        data_str = match_grandTotal.group(1)
        data = json.loads(data_str)
        print(f"\nData_grandTotal: {data}")
    except:
        pass
