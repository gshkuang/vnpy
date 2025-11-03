import time
from typing import Tuple

import polars as pl

from vnpy.alpha.dataset.ts_function import ts_resi as ts_resi_old
from vnpy.alpha.dataset.ts_function import ts_rsquare as ts_rsquare_old
from vnpy.alpha.dataset.ts_function import ts_slope as ts_slope_old
from vnpy.alpha.dataset.utility import FeatProxy


def make_synthetic_df(
    num_symbols: int = 5, rows_per_symbol: int = 10000
) -> pl.DataFrame:
    """Generate a typical multi-symbol OHLC-like dataset with trends and noise."""
    dfs = []
    for i in range(num_symbols):
        sym = f"SYM{i:02d}"
        # datetime as integer index for speed; real flow uses actual timestamps
        dt = pl.arange(0, rows_per_symbol, eager=True)
        # base price with slight trend and sinusoidal component + noise
        idx = pl.Series("idx", list(range(rows_per_symbol)))
        base = idx.cast(pl.Float64) * (0.001 + 0.0002 * i)  # symbol-specific trend
        sinus = (idx.cast(pl.Float64) / 50.0).sin() * 0.5
        noise = pl.Series("noise", pl.arange(0, rows_per_symbol, eager=True)).cast(
            pl.Float64
        )
        # create random-like noise via hash
        noise = (noise * 9301 + 49297) % 233280
        noise = (noise / 233280.0 - 0.5) * 0.2

        close = (base + sinus + noise).alias("close")
        df_sym = pl.DataFrame(
            {
                "datetime": dt,
                "vt_symbol": pl.Series([sym] * rows_per_symbol, dtype=pl.Utf8),
                "close": close,
            }
        )
        dfs.append(df_sym)

    df = pl.concat(dfs)
    # ensure sorted by symbol/time
    df = df.sort(["vt_symbol", "datetime"])
    return df


def ts_slope_vec(feature: FeatProxy, window: int) -> FeatProxy:
    """Vectorized slope via rolling sums and closed-form formula."""
    # Use datetime as sequential x (synthetic data ensures step=1)
    lf = (
        feature.df.with_columns(pl.col("datetime").cast(pl.Float64).alias("x"))
        .with_columns(
            (pl.col("x"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sx"),
            (pl.col("data"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sy"),
            (pl.col("x") * pl.col("x"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sxx"),
            (pl.col("x") * pl.col("data"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sxy"),
            # n = min(window, index within group + 1). Compute index by datetime offset from group min
            (
                pl.min_horizontal(
                    pl.lit(window),
                    (
                        pl.col("datetime") - pl.col("datetime").min().over("vt_symbol")
                    ).cast(pl.Float64)
                    + 1,
                )
            ).alias("n"),
        )
        .with_columns(
            (
                (pl.col("Sxy") - pl.col("Sx") * pl.col("Sy") / pl.col("n"))
                / (pl.col("Sxx") - pl.col("Sx") * pl.col("Sx") / pl.col("n"))
            ).alias("slope_raw")
        )
        .select(
            pl.col("datetime"), pl.col("vt_symbol"), pl.col("slope_raw").alias("data")
        )
        .with_columns(
            pl.when(pl.col("data").is_infinite() | pl.col("data").is_nan())
            .then(None)
            .otherwise(pl.col("data"))
            .alias("data")
        )
    )
    return FeatProxy(lf)


def ts_rsquare_vec(feature: FeatProxy, window: int) -> FeatProxy:
    """Vectorized R^2 using population variance convention, matching original intent."""
    lf = (
        feature.df.with_columns(pl.col("datetime").cast(pl.Float64).alias("x"))
        .with_columns(
            (pl.col("x"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sx"),
            (pl.col("data"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sy"),
            (pl.col("x") * pl.col("x"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sxx"),
            (pl.col("data") * pl.col("data"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Syy"),
            (pl.col("x") * pl.col("data"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sxy"),
            (
                pl.min_horizontal(
                    pl.lit(window),
                    (
                        pl.col("datetime") - pl.col("datetime").min().over("vt_symbol")
                    ).cast(pl.Float64)
                    + 1,
                )
            ).alias("n"),
        )
        .with_columns(
            (
                (pl.col("Sxy") - pl.col("Sx") * pl.col("Sy") / pl.col("n"))
                / pl.col("n")
            ).alias("cov"),
            (
                (pl.col("Sxx") - pl.col("Sx") * pl.col("Sx") / pl.col("n"))
                / pl.col("n")
            ).alias("var_x"),
            (
                (pl.col("Syy") - pl.col("Sy") * pl.col("Sy") / pl.col("n"))
                / pl.col("n")
            ).alias("var_y"),
        )
        .select(
            pl.col("datetime"),
            pl.col("vt_symbol"),
            pl.when((pl.col("var_x") <= 0) | (pl.col("var_y") <= 0))
            .then(None)
            .otherwise(
                ((pl.col("cov") * pl.col("cov")) / (pl.col("var_x") * pl.col("var_y")))
            )
            .alias("data"),
        )
    )
    return FeatProxy(lf)


def ts_resi_vec(feature: FeatProxy, window: int) -> FeatProxy:
    """Vectorized residual at last point y_last - (m*x_last + b)."""
    lf = (
        feature.df.with_columns(pl.col("datetime").cast(pl.Float64).alias("x"))
        .with_columns(
            (pl.col("x"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sx"),
            (pl.col("data"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sy"),
            (pl.col("x") * pl.col("x"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sxx"),
            (pl.col("x") * pl.col("data"))
            .rolling_sum(window, min_samples=1)
            .over("vt_symbol")
            .alias("Sxy"),
            (
                pl.min_horizontal(
                    pl.lit(window),
                    (
                        pl.col("datetime") - pl.col("datetime").min().over("vt_symbol")
                    ).cast(pl.Float64)
                    + 1,
                )
            ).alias("n"),
        )
        .with_columns(
            (
                (
                    (pl.col("Sxy") - pl.col("Sx") * pl.col("Sy") / pl.col("n"))
                    / (pl.col("Sxx") - pl.col("Sx") * pl.col("Sx") / pl.col("n"))
                )
            ).alias("m"),
            (pl.col("Sx") / pl.col("n")).alias("mx"),
            (pl.col("Sy") / pl.col("n")).alias("my"),
        )
        .select(
            pl.col("datetime"),
            pl.col("vt_symbol"),
            (
                pl.col("data")
                - (
                    pl.col("m") * pl.col("x")
                    + (pl.col("my") - pl.col("m") * pl.col("mx"))
                )
            ).alias("data"),
        )
        .with_columns(
            pl.when(pl.col("data").is_infinite() | pl.col("data").is_nan())
            .then(None)
            .otherwise(pl.col("data"))
            .alias("data")
        )
    )
    return FeatProxy(lf)


def bench_one(
    name: str, old_fn, new_fn, feature: FeatProxy, window: int
) -> Tuple[float, float, pl.DataFrame]:
    """Run timing for old/new and return results and joined DataFrame for accuracy check."""

    # old
    t0 = time.perf_counter()
    old_df = old_fn(feature, window).df.collect()
    t1 = time.perf_counter()
    old_time = t1 - t0

    # new
    t0 = time.perf_counter()
    new_df = new_fn(feature, window).df.collect()
    t1 = time.perf_counter()
    new_time = t1 - t0

    joined = old_df.rename({"data": f"{name}_old"}).join(
        new_df.rename({"data": f"{name}_new"}),
        on=["datetime", "vt_symbol"],
        how="inner",
    )
    return old_time, new_time, joined


def accuracy_stats(df: pl.DataFrame, name: str) -> None:
    old_col = f"{name}_old"
    new_col = f"{name}_new"
    diff_col = f"{name}_diff"
    df2 = df.with_columns(
        pl.when(pl.any_horizontal(pl.col(old_col).is_null(), pl.col(new_col).is_null()))
        .then(None)
        .otherwise((pl.col(old_col) - pl.col(new_col)).abs())
        .alias(diff_col)
    )
    # compute stats ignoring nulls
    stats = df2.select(
        pl.len().alias("rows"),
        pl.col(diff_col).mean().alias("mean_abs_diff"),
        pl.col(diff_col).max().alias("max_abs_diff"),
        (pl.col(diff_col) > 1e-9).sum().alias("mismatch_gt_1e-9"),
    )
    print(f"[ACCURACY] {name}:\n{stats}")

    # show a few mismatches if any
    mism = df2.filter(pl.col(diff_col) > 1e-6).select(
        ["datetime", "vt_symbol", old_col, new_col, diff_col]
    )
    if mism.height > 0:
        print(f"[SAMPLES >1e-6] {name}:")
        print(mism.head(5))


def main() -> None:
    window = 20
    df = make_synthetic_df(num_symbols=6, rows_per_symbol=15000)
    feature = FeatProxy.col2proxy(df, "close")

    print("Dataset rows:", df.height)
    print("Symbols:", df.select(pl.col("vt_symbol").n_unique()))
    print("Window:", window)

    # slope
    s_old, s_new, s_join = bench_one(
        "slope", ts_slope_old, ts_slope_vec, feature, window
    )
    print(
        f"[TIME] slope old={s_old:.4f}s new={s_new:.4f}s speedup={s_old/s_new if s_new>0 else float('inf'):.2f}x"
    )
    accuracy_stats(s_join, "slope")

    # rsquare
    r_old, r_new, r_join = bench_one(
        "rsquare", ts_rsquare_old, ts_rsquare_vec, feature, window
    )
    print(
        f"[TIME] rsquare old={r_old:.4f}s new={r_new:.4f}s speedup={r_old/r_new if r_new>0 else float('inf'):.2f}x"
    )
    accuracy_stats(r_join, "rsquare")

    # residual
    e_old, e_new, e_join = bench_one("resi", ts_resi_old, ts_resi_vec, feature, window)
    print(
        f"[TIME] resi old={e_old:.4f}s new={e_new:.4f}s speedup={e_old/e_new if e_new>0 else float('inf'):.2f}x"
    )
    accuracy_stats(e_join, "resi")


if __name__ == "__main__":
    main()
