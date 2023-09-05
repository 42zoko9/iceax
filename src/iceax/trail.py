from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .params import FigureParams


class Trail:
    def __init__(self, series: pd.Series) -> None:
        self.series = series

    def stats(self) -> pd.DataFrame:
        '''対象の統計量を出力

        Returns:
            pd.DataFrame: 統計量をまとめたデータフレーム
        '''
        pass

    def __quantitative_stats(self) -> pd.DataFrame:
        '''量的変数及び時系列の統計量(最小・四分位点・最頻値・最大)を出力

        Returns:
            pd.DataFrame: 量的変数の統計量をまとめたデータフレーム
        '''
        pass

    def plot(self, kind: str, params: FigureParams) -> Figure:
        '''対象の可視化

        Args:
            kind (str): グラフの種別
            params (FigureParams): グラフの体裁を調整するパラメータ群

        Returns:
            Figure: 可視化結果
        '''
        pass

    def __barplot(self, top: Optional[int] = None) -> Figure:
        '''質的変数の値あたりの出現頻度を棒グラフとして出力

        Args:
            top (Optional[int], optional): 可視化する頻度の下限値

        Returns:
            Figure: 棒グラフ
        '''
        pass

    def __freq_cnt(self) -> pd.DataFrame:
        '''質的変数の値あたりの出現頻度を計算

        Returns:
            pd.DataFrame: 出現頻度で降順でソートしたデータフレーム
        '''
        pass

    def __boxplot(self, outlier: Optional[bool] = True) -> Figure:
        '''量的変数の分布を箱ひげ図とjitter plotで出力

        Args:
            outlier (bool, optional): _description_. Defaults to True.

        Returns:
            Figure: 箱ひげ図
        '''
        pass

    def __quantile(self) -> List[float]:
        '''箱ひげ図出力用に対象の量的変数の四分位点を出力

        Returns:
            List[float]: 四分位点の値が格納されたリスト
        '''
        pass

    def __histgram(self, width: Optional[Union[float, int]] = None) -> Figure:
        '''量的変数の値あたりの出現頻度をヒストグラムとして出力

        Args:
            width (Optional[Union[float, int]], optional): ビンの幅

        Returns:
            Figure: ヒストグラム
        '''
        pass
