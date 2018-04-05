from typing import (
    NamedTuple,
    Tuple,
)


class PlotSpec(NamedTuple):
    name: str
    title: str
    outdir: str
    ylabel: str
    figsize: Tuple[int] = (14, 14)
    grid: bool = True
    line_width: int = 4
    save_format: str = "png"
    label_loc: str = "best"
