import polars as pl

from vnpy.alpha import AlphaDataset
from vnpy.alpha.dataset.utility import FeatProxy
from vnpy.alpha.dataset.ts_function import (
    ts_delay, ts_min, ts_max,
    ts_argmax, ts_argmin,
    ts_rank, ts_sum,
    ts_mean, ts_std,
    ts_slope, ts_quantile,
    ts_rsquare, ts_resi,
    ts_corr,
    ts_less, ts_greater,
    ts_log, ts_abs
)


class Alpha158(AlphaDataset):
    """158 basic factors from Qlib"""

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        period: str = "m",
        ) -> None:
            """Constructor"""
            super().__init__(
                df=df,
                train_period=train_period,
                valid_period=valid_period,
                test_period=test_period,
            )


            o = FeatProxy.col2proxy(self.df, "open")
            h = FeatProxy.col2proxy(self.df, "high")
            l = FeatProxy.col2proxy(self.df, "low")
            c = FeatProxy.col2proxy(self.df, "close")
            vwap = FeatProxy.col2proxy(self.df, "vwap")
            v = FeatProxy.col2proxy(self.df, "volume")

            # Candlestick pattern features
            self.add_feature("kmid", (c - o) / o)
            self.add_feature("klen", (h - l) / o)
            self.add_feature("kmid_2", (c - o) / (h - l + 1e-12))
            self.add_feature("kup", (h - ts_greater(o, c)) / o)
            self.add_feature("kup_2", (h - ts_greater(o, c)) / (h - l + 1e-12))
            self.add_feature("klow", (ts_less(o, c) - l) / o)
            self.add_feature("klow_2", (ts_less(o, c) - l) / (h - l + 1e-12))
            self.add_feature("ksft", (c * 2 - h - l) / o)
            self.add_feature("ksft_2", (c * 2 - h - l) / (h - l + 1e-12))

            # Price change features
            for field_name, dp_field in [("open", o), ("high", h), ("low", l), ("vwap", vwap)]:
                self.add_feature(f"{field_name}_0", dp_field / c)

            # Time series features
            windows: list[int] = [5, 10, 20, 30, 60]

            for w in windows:
                self.add_feature(f"roc_{w}", ts_delay(c, w) / c)

            for w in windows:
                self.add_feature(f"ma_{w}", ts_mean(c, w) / c)

            for w in windows:
                self.add_feature(f"std_{w}", ts_std(c, w) / c)

            for w in windows:
                self.add_feature(f"beta_{w}", ts_slope(c, w) / c)

            for w in windows:
                self.add_feature(f"rsqr_{w}", ts_rsquare(c, w))

            for w in windows:
                self.add_feature(f"resi_{w}", ts_resi(c, w) / c)

            for w in windows:
                self.add_feature(f"max_{w}", ts_max(h, w) / c)

            for w in windows:
                self.add_feature(f"min_{w}", ts_min(l, w) / c)

            for w in windows:
                self.add_feature(f"qtlu_{w}", ts_quantile(c, w, 0.8) / c)

            for w in windows:
                self.add_feature(f"qtld_{w}", ts_quantile(c, w, 0.2) / c)

            for w in windows:
                self.add_feature(f"rank_{w}", ts_rank(c, w, period))

            for w in windows:
                self.add_feature(
                    f"rsv_{w}",
                    (c - ts_min(l, w)) / (ts_max(h, w) - ts_min(l, w) + 1e-12)
                )

            for w in windows:
                self.add_feature(f"imax_{w}", ts_argmax(h, w) / w)

            for w in windows:
                self.add_feature(f"imin_{w}", ts_argmin(l, w) / w)

            for w in windows:
                self.add_feature(f"imxd_{w}", (ts_argmax(h, w) - ts_argmin(l, w)) / w)

            for w in windows:
                self.add_feature(f"corr_{w}", ts_corr(c, ts_log(v + 1), w))

            for w in windows:
                self.add_feature(
                    f"cord_{w}",
                    ts_corr(c / ts_delay(c, 1), ts_log(v / ts_delay(v, 1) + 1), w)
                )

            for w in windows:
                self.add_feature(f"cntp_{w}", ts_mean(c > ts_delay(c, 1), w))

            for w in windows:
                self.add_feature(f"cntn_{w}", ts_mean(c < ts_delay(c, 1), w))

            for w in windows:
                self.add_feature(f"cntd_{w}", ts_mean(c > ts_delay(c, 1), w) - ts_mean(c < ts_delay(c, 1), w))

            for w in windows:
                self.add_feature(
                    f"sump_{w}",
                    ts_sum(ts_greater(c - ts_delay(c, 1), 0), w) / (ts_sum(ts_abs(c - ts_delay(c, 1)), w) + 1e-12)
                )

            for w in windows:
                self.add_feature(
                    f"sumn_{w}",
                    ts_sum(ts_greater(ts_delay(c, 1) - c, 0), w) / (ts_sum(ts_abs(c - ts_delay(c, 1)), w) + 1e-12)
                )

            for w in windows:
                self.add_feature(
                    f"sumd_{w}",
                    (ts_sum(ts_greater(c - ts_delay(c, 1), 0), w) - ts_sum(ts_greater(ts_delay(c, 1) - c, 0), w)) / (ts_sum(ts_abs(c - ts_delay(c, 1)), w) + 1e-12)
                )

            for w in windows:
                self.add_feature(f"vma_{w}", ts_mean(v, w) / (v + 1e-12))

            for w in windows:
                self.add_feature(f"vstd_{w}", ts_std(v, w) / (v + 1e-12))

            for w in windows:
                self.add_feature(
                    f"wvma_{w}",
                    ts_std(ts_abs(c / ts_delay(c, 1) - 1) * v, w) / (ts_mean(ts_abs(c / ts_delay(c, 1) - 1) * v, w) + 1e-12)
                )

            for w in windows:
                self.add_feature(
                    f"vsump_{w}",
                    ts_sum(ts_greater(v - ts_delay(v, 1), 0), w) / (ts_sum(ts_abs(v - ts_delay(v, 1)), w) + 1e-12)
                )

            for w in windows:
                self.add_feature(
                    f"vsumn_{w}",
                    ts_sum(ts_greater(ts_delay(v, 1) - v, 0), w) / (ts_sum(ts_abs(v - ts_delay(v, 1)), w) + 1e-12)
                )

            for w in windows:
                self.add_feature(
                    f"vsumd_{w}",
                    (ts_sum(ts_greater(v - ts_delay(v, 1), 0), w) - ts_sum(ts_greater(ts_delay(v, 1) - v, 0), w)) / (ts_sum(ts_abs(v - ts_delay(v, 1)), w) + 1e-12)
                )

            # Set label
            label = ts_delay(c, -3) / ts_delay(c, -1) - 1
            self.set_label(label)
