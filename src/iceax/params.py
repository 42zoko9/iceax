from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass
class FigureParams:
    figsize: Tuple[int, int]
    title: str
    xlabel: str
    ylabel: str
    xlim: Optional[Tuple[Union[float, int, str], Union[float, int, str]]] = None
    ylim: Optional[Tuple[Union[float, int, str], Union[float, int, str]]] = None
    legend: Optional[str] = None
