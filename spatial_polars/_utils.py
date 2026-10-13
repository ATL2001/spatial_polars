from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from polars import selectors as cs

try:
    from matplotlib.colors import Colormap

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False

try:
    from palettable.palette import Palette

    HAS_PALETTABLE = True
except ModuleNotFoundError:
    HAS_PALETTABLE = False


def validate_col_exists_in_df(df: pl.DataFrame, col: str) -> None:
    if col not in df.columns:
        # verify column exists in df
        msg = (
            f"Input `{col=}` does not exist in frame.  Please provide the name of a",
            " column from the dataframe ({df.columns}).",
        )
        raise ValueError(msg)


def validate_col_is_numeric(df: pl.DataFrame, col: str) -> None:
    if df[col].dtype.is_numeric() is False:
        # verify cmap_col is a numeric dtype.
        msg = (
            f"The '{col}' column is not a numeric column.  Please provide a numeric",
            " column.",
        )
        raise ValueError(msg)


def _validate_cmap_dict(
    df: pl.DataFrame,
    cmap_col: str,
    cmap: Palette | Colormap | dict,
) -> None:
    # verify values in dataframe column all exist in cmap dictionary keys
    try:
        cmap_keys_s = pl.Series("cmap_keys", list(cmap.keys()))
    except TypeError:
        msg = (
            "`cmap_type` is `categorical` and `cmap` is a dictionary, but ",
            " there was an error converting the dictionary keys to a polars ",
            " series. Please provide a dictionary with keys that are all the ",
            "appropriate datatype for the data in the column.",
        )
        raise ValueError(msg) from None

    col_vals_and_keys_df = (
        df[cmap_col]
        .unique()
        .to_frame()
        .join(
            cmap_keys_s.to_frame().with_columns(
                pl.col("cmap_keys").cast(df[cmap_col].dtype),
                pl.lit(value=True).alias("exists_in_keys"),
            ),
            how="left",
            left_on=cmap_col,
            right_on="cmap_keys",
        )
    )
    if col_vals_and_keys_df["exists_in_keys"].is_null().any():
        missing_values = col_vals_and_keys_df.filter(
            pl.col("exists_in_keys").is_null(),
        )[cmap_col].to_list()
        msg = (
            "`cmap` dictionary keys do not include all values in the data.",
            " Please provide colors for the following values",
            f" {missing_values}.",
        )
        raise ValueError(msg)

    # verify values have three or four items
    try:
        cmap_vals_df = pl.DataFrame(list(cmap.values()), orient="row")
    except TypeError:
        msg = (
            "`cmap` dictionary values could not be converted to a dataframe",
            " with three or four integer columns for color codes.  Please",
            " provide RGB[A] color codes for all values.",
        )
        raise ValueError(msg) from None

    col_count = cmap_vals_df.shape[1]
    if col_count not in [3, 4]:
        msg = (
            "`cmap` dictionary values could not be converted to a dataframe",
            " with three or four integer columns for color codes.  Please",
            " provide RGB[A] color codes for all values.",
        )
        raise ValueError(msg)

    if col_count == 3:  # NOQA:PLR2004
        cmap_vals_df.columns = ["R", "G", "B"]
    elif col_count == 4:  # NOQA:PLR2004
        cmap_vals_df.columns = ["R", "G", "B", "A"]

    # verify all RGB[A] values are between 0-255
    oob_rgba_df = cmap_vals_df.select(cmap_keys_s, pl.all()).filter(
        pl.any_horizontal(
            ~cs.by_name("R", "G", "B", "A", require_all=False).is_between(
                0,
                255,
                closed="both",
            ),
        ),
    )

    if len(oob_rgba_df) > 0:
        msg = (
            "`cmap` dictionary values for color codes were not all between 0",
            " and 255.  Please provide number between 0-255 for all RGB[A]",
            f" color codes. Bad values were: {oob_rgba_df}",
        )
        raise ValueError(msg)


def validate_cmap_input(
    df: pl.DataFrame,
    cmap_col: str,
    cmap_type: Literal["categorical", "continuous"],
    cmap: Palette | Colormap | dict,
) -> None:
    """Validate the input combinations for color mapping options."""
    if cmap_col is None:
        # no column provided is always good, use lonboard default color
        return

    validate_col_exists_in_df(df, cmap_col)

    if cmap_type not in ["categorical", "continuous"]:
        # verify cmap type is expected
        msg = (
            f"`cmap_col` was provided, but {cmap_type=}.  `c_map` must be",
            " `categorical` or `continuous` when `cmap_col` is provided.",
        )
        raise ValueError(msg)

    if cmap_type == "continuous":
        if (HAS_MATPLOTLIB and isinstance(cmap, Colormap) is False) and (
            HAS_PALETTABLE and isinstance(cmap, Palette) is False
        ):
            # verify continuous cmaps get a cmap or appropriate type.
            msg = (
                f"`{cmap_type=}` is `continuous`, but {cmap=}.  Please provide a",
                " palettable.palette.Palette or matplotlib.colors.Colormap when using",
                " `cmap_type`=`continuous`.",
            )
            raise ValueError(msg)
        validate_col_is_numeric(df, cmap_col)

    if cmap_type == "categorical":
        if isinstance(cmap, dict | None) is False:
            # verify categorical cmaps get a cmap or appropriate type.
            msg = (
                "`cmap_type` is `categorical`, but `cmap` is not a dictionary or",
                " None.  Please provide a dictionary or None when using",
                "`cmap_type`=`categorical`.",
            )
            raise ValueError(msg)

        if isinstance(cmap, dict):
            _validate_cmap_dict(df, cmap_col, cmap)


def validate_width_and_radius_input(
    df: pl.DataFrame,
    width: float | np.ndarray | str,
) -> None:
    if width is None:
        # None is always good
        return

    if isinstance(width, float | int):
        # numbers are always good
        return
    if isinstance(width, np.ndarray):
        if np.issubdtype(width.dtype, np.number) is False:
            # validate numpy array dtype is numeric
            msg = (
                "Input `width` is a numpy array that is not numeric.  If using a numpy",
                " array for the width, the datatype of the array must be numeric.",
            )
            raise ValueError(msg)

        width_len = width.shape[0]
        df_len = len(df)
        if width_len != df_len:
            # validate numpy array has the same number of items as the df
            msg = (
                "Input `width` is a numpy array with a different number of elements",
                f" than the frame (array: {width_len:,} frame: {df_len:,}).  If using",
                " a numpy array for the width, the the array must have the same number",
                " of elements as the frame has rows.",
            )
            raise ValueError(msg)

    elif isinstance(width, str):
        validate_col_exists_in_df(df, width)
        validate_col_is_numeric(df, width)

    else:
        msg = (
            f"Input `width` '{width}' is a '{type(width)}', which is not allowed.",
            " Please provide: None, a number, a column name, or a numpy array",
            " for the width.",
        )
        raise TypeError(msg)
