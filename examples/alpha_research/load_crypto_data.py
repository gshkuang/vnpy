# 加载模块
import glob
import os
import re
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from vnpy.alpha import AlphaLab, logger
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ
from vnpy.trader.object import BarData

# 设置参数
# 为不同的时间周期设置不同的任务名称和日期范围
interval_settings = {
    # "15m": {
    #     "task_name": "crypto_15m",
    #     "index_name": "CRYPTO_INDEX_15M",
    #     "data_path": "/Users/guangshengkuang/Desktop/learning/fetch_market_data/data/15m",
    #     "start_date": "2020-01-01",  # 15分钟数据通常从较近的日期开始
    #     "end_date": "2025-09-04",
    # },
    # "1h": {
    #     "task_name": "crypto_1h",
    #     "index_name": "CRYPTO_INDEX_1H",
    #     "data_path": "/Users/guangshengkuang/Desktop/learning/fetch_market_data/data/1h",
    #     "start_date": "2020-01-01",  # 1小时数据可以从稍早开始
    #     "end_date": "2025-09-02",
    # },
    # "1d": {
    #     "task_name": "crypto_1d",
    #     "index_name": "CRYPTO_INDEX_1D",
    #     "data_path": "/Users/guangshengkuang/Desktop/learning/fetch_market_data/data/1d",
    #     "start_date": "2020-01-01",  # 日线数据可以从最早开始
    #     "end_date": "2025-09-03",
    # },
    "1m": {
        "task_name": "crypto_1m",
        "index_name": "CRYPTO_INDEX_1M",
        "data_path": "/Users/guangshengkuang/Desktop/learning/fetch_market_data/data/1m",
        "start_date": "2020-01-01",  # 1分钟数据可以从最早开始
        "end_date": "2025-09-04",
    },
}


# 解析文件名获取交易对和时间间隔信息
def extract_symbol_info(filename):
    basename = os.path.basename(filename)
    # 文件格式如A2ZUSDT_1h_2020-01-01_2025-09-02.csv
    match = re.match(
        r"(.+)_(15m|1h|1d|1m)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv", basename
    )
    if match:
        symbol, interval_str, start_date_str, end_date_str = match.groups()

        # 映射时间间隔
        interval_map = {
            "15m": Interval.MINUTE,
            "1h": Interval.HOUR,
            "1d": Interval.DAILY,
            "1m": Interval.MINUTE,
        }

        interval = interval_map.get(interval_str)
        if not interval:
            return None, None, None, None, None

        # 对于加密货币，使用BINANCE作为交易所
        exchange = Exchange.BINANCE
        vt_symbol = f"{symbol}.{exchange.value}"

        return vt_symbol, symbol, exchange, interval, (start_date_str, end_date_str)

    return None, None, None, None, None


# 创建加密货币组件数据结构
def create_crypto_components(data_path, start_date, end_date):
    components = {}
    component_symbols = []

    # 获取指定目录中的所有CSV文件
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))

    # 收集所有唯一的交易对
    for csv_file in csv_files:
        vt_symbol, _, _, _, _ = extract_symbol_info(csv_file)
        if vt_symbol and vt_symbol not in component_symbols:
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
        # 移动到下一天
        current_dt = current_dt + timedelta(days=1)

    return components, component_symbols


# 处理每个时间周期的数据
total_processed = 0

for interval_key, settings in interval_settings.items():
    print(f"\n处理 {interval_key} 时间周期的数据...")

    # 创建投研实验室
    task_name = settings["task_name"]
    index_name = settings["index_name"]
    data_path = settings["data_path"]
    start_date = settings["start_date"]
    end_date = settings["end_date"]

    lab = AlphaLab(f"./lab/{task_name}")  # 指定数据文件夹

    # 保存成分股数据
    print(f"创建 {interval_key} 加密货币组件数据...")
    crypto_components, component_symbols = create_crypto_components(
        data_path, start_date, end_date
    )
    lab.save_component_data(index_name, crypto_components)

    # 加载加密货币成分股代码
    component_symbols = lab.load_component_symbols(index_name, start_date, end_date)
    print(f"加载了 {len(component_symbols)} 个 {interval_key} 加密货币交易对")

    # 转换时间格式
    start = datetime.strptime(start_date, "%Y-%m-%d")
    start = start.replace(tzinfo=DB_TZ)

    end = datetime.strptime(end_date, "%Y-%m-%d")
    end = end.replace(tzinfo=DB_TZ)

    # 处理该时间周期的所有加密货币数据
    print(f"开始处理 {interval_key} 加密货币数据...")
    processed_count = 0

    csv_files = glob.glob(os.path.join(data_path, "*.csv"))

    for csv_file in tqdm(csv_files):
        vt_symbol, symbol, exchange, interval, _ = extract_symbol_info(csv_file)

        if not vt_symbol:
            continue

        # 读取CSV数据
        df = pd.read_csv(csv_file)

        # 转换为Bar对象列表
        bars = []
        for _, row in df.iterrows():
            # 解析日期，假设CSV中的时间列名为'datetime'或'date'
            date_column = "timestamp" if "timestamp" in df.columns else "date"

            try:
                # 尝试解析不同格式的日期时间
                if "time" in df.columns:  # 如果有单独的时间列
                    dt_str = f"{row[date_column]} {row['time']}"
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                else:
                    # 尝试不同的日期格式
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            dt = datetime.strptime(str(row[date_column]), fmt)
                            break
                        except ValueError:
                            continue

                dt = dt.replace(tzinfo=DB_TZ)

                # 跳过不在时间范围内的数据
                if dt < start or dt > end:
                    continue

                # 检查OHLC数据是否为NaN
                if (
                    pd.isna(row["open"])
                    or pd.isna(row["high"])
                    or pd.isna(row["low"])
                    or pd.isna(row["close"])
                ):
                    logger.warning(f"跳过{vt_symbol}的NaN数据: {row}")
                    continue

                # 创建Bar对象
                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=dt,
                    interval=interval,
                    open_price=float(row["open"]),
                    high_price=float(row["high"]),
                    low_price=float(row["low"]),
                    close_price=float(row["close"]),
                    volume=float(row["volume"]) if "volume" in row else 0,
                    turnover=float(row["amount"]) if "amount" in row else 0,
                    gateway_name="CSV",
                )
                bars.append(bar)
            except (ValueError, KeyError) as e:
                logger.warning(f"处理{vt_symbol}数据时出错: {e}, 行数据: {row}")
                continue

        # 保存到数据库
        if bars:
            lab.save_bar_data(bars)
            processed_count += 1
        else:
            logger.warning(f"没有找到{vt_symbol}在指定时间范围内的数据")

    # 添加回测参数配置
    for vt_symbol in component_symbols:
        lab.add_contract_setting(
            vt_symbol,
            long_rate=1 / 1000,  # 加密货币的费率通常比股票高
            short_rate=1 / 1000,
            size=1,
            pricetick=0.0001,
        )

    print(
        f"{interval_key} 数据处理完成，共处理了 {processed_count} 个加密货币交易对的数据文件"
    )
    total_processed += processed_count

print(f"\n所有时间周期数据处理完成，总共处理了 {total_processed} 个数据文件")
