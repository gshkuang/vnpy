# 加载模块
import glob
import os
import re
from datetime import datetime

from vnpy.alpha import AlphaLab
from vnpy.trader.constant import Exchange

# 设置参数
task_name = "csi300"
index_symbol = "000300.SSE"
local_data_path = "/Users/guangshengkuang/Desktop/learning/fetch_market_data/stk_data/d"

start_date = "2007-01-01"
end_date = "2024-10-31"

# 创建投研实验室
lab = AlphaLab(f"./lab/{task_name}")  # 指定数据文件夹

# 获取所有CSV文件
csv_files = glob.glob(os.path.join(local_data_path, "*.csv"))


# 提取股票代码和交易所信息
def extract_symbol_exchange(filename):
    basename = os.path.basename(filename)
    match = re.match(r"(sh|sz)\.(\d+)\.csv", basename)
    if match:
        prefix, code = match.groups()
        if prefix == "sh":
            exchange = Exchange.SSE
            vt_symbol = f"{code}.SSE"
        elif prefix == "sz":
            exchange = Exchange.SZSE
            vt_symbol = f"{code}.SZSE"
        return vt_symbol, code, exchange
    return None, None, None


# 模拟指数成分股数据结构
def create_index_components():
    components = {}
    # 使用所有文件作为成分股，假设所有时间段都是相同的成分股
    component_symbols = []

    for csv_file in csv_files:
        vt_symbol, _, _ = extract_symbol_exchange(csv_file)
        if vt_symbol:
            component_symbols.append(vt_symbol)

    # 创建一个日期到成分股列表的映射
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # 为从start_date到end_date的每一天创建成分股映射
    from datetime import timedelta

    current_dt = start_dt
    while current_dt <= end_dt:
        current_date = current_dt.strftime("%Y-%m-%d")
        components[current_date] = component_symbols
        # 移动到下一天（使用timedelta避免月份溢出问题）
        current_dt = current_dt + timedelta(days=1)

    return components


# 保存成分股数据
index_components = create_index_components()

lab.save_component_data(index_symbol, index_components)

# 加载指数成分股代码
component_symbols = lab.load_component_symbols(index_symbol, start_date, end_date)
print(component_symbols)
# # 转换时间格式
# start = datetime.strptime(start_date, "%Y-%m-%d")
# start = start.replace(tzinfo=DB_TZ)

# end = datetime.strptime(end_date, "%Y-%m-%d")
# end = end.replace(tzinfo=DB_TZ)

# # 处理所有股票数据
# for csv_file in tqdm(csv_files):
#     vt_symbol, symbol, exchange = extract_symbol_exchange(csv_file)

#     if not vt_symbol:
#         continue

#     # 读取CSV数据
#     df = pd.read_csv(csv_file)

#     # 转换为Bar对象列表
#     bars = []
#     for _, row in df.iterrows():
#         # 解析日期
#         dt = datetime.strptime(str(row["date"]), "%Y-%m-%d")
#         dt = dt.replace(tzinfo=DB_TZ)

#         # 跳过不在时间范围内的数据
#         if dt < start or dt > end:
#             continue

#         # 创建Bar对象
#         bar = BarData(
#             symbol=symbol,
#             exchange=exchange,
#             datetime=dt,
#             interval=Interval.DAILY,
#             open_price=row["open"],
#             high_price=row["high"],
#             low_price=row["low"],
#             close_price=row["close"],
#             volume=row["volume"],
#             turnover=row["amount"],
#             gateway_name="CSV",
#         )
#         bars.append(bar)

#     # 保存到数据库
#     if bars:
#         lab.save_bar_data(bars)
#     else:
#         logger.warning(f"没有找到{vt_symbol}在指定时间范围内的数据")

# # 添加回测参数配置
# for vt_symbol in component_symbols:
#     lab.add_contract_setting(
#         vt_symbol,
#         long_rate=5 / 10000,
#         short_rate=10 / 10000,
#         size=1,
#         pricetick=0.0001,
#     )

# print(f"数据处理完成，共处理了{len(component_symbols)}个股票")
