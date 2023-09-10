from typing import List, Optional, Tuple, Union

import pandas as pd
from matplotlib.figure import Figure


class Trails:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def corr(self) -> None:
        '''相関行列を出力
        '''
        pass

    def __summarise(self, key: List[str], func: str) -> pd.DataFrame:
        '''可視化用の集計関数

        Args:
            key (List[str]): 集計キーとなるカラム名一覧
            func (str): 集計関数, とりうる値はavg, count, max, min, sum

        Returns:
            pd.DataFrame: 集計結果のデータフレーム
        '''
        pass

    def lineplot(self, x: str, y: str, color: Optional[str] = None, figsize: Tuple[Union[float, int], Union[float, int]] = (16., 9.)) -> Figure:
        '''折れ線グラフを出力

        Args:
            x (str): x軸用の時系列変数のカラム名
            y (str): y軸用の量的変数のカラム名
            color (str, optional): 配色用の質的変数のカラム名. デフォルト値はNone.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ.デフォルト値は(16., 9.).

        Returns:
            Figure: 折れ線グラフ
        '''
        pass

    def scatterplot(self, x: str, y: str, color: Optional[str] = None, figsize: Tuple[Union[float, int], Union[float, int]] = (16., 9.)) -> Figure:
        '''散布図を出力

        Args:
            x (str): x軸用の量的変数のカラム名
            y (str): y軸用の量的変数のカラム名
            color (str, optional): 配色用の質的変数のカラム名. デフォルト値はNone.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ.デフォルト値は(16., 9.).

        Returns:
            Figure: 散布図
        '''
        pass

    def barplot(self, x: str, y: str, color: Optional[str] = None, rate_txt: bool = True, divline: bool = True, figsize: Tuple[Union[float, int], Union[float, int]] = (16., 9.)) -> Figure:
        '''棒グラフを出力

        Args:
            x (str): x軸用の質的変数のカラム名
            y (str): y軸用の量的変数のカラム名
            color (str, optional): 配色用の質的変数のカラム名. デフォルト値はNone.
            rate_txt (bool, optional): 比率の数値をプロットする. デフォルト値はTrue.
            divline (bool, optional): 区分線をプロットする. デフォルト値はTrue.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: 棒グラフ
        '''
        pass

    def boxplot(self, x: str, y: str, outlier: bool = True, figsize: Tuple[Union[float, int], Union[float, int]] = (16., 9.)) -> Figure:
        '''箱ひげ図を出力

        Args:
            x (str): x軸用の質的変数のカラム名
            y (str): y軸用の量的変数のカラム名
            outlier (bool, optional): jitter-plotの描画判定. デフォルト値はTrue.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: 箱ひげ図
        '''
        pass

    def heatmap(self, x: str, y: str, rate_txt: bool = True, figsize: Tuple[Union[float, int], Union[float, int]] = (16., 9.)) -> Figure:
        '''出現頻度のヒートマップを出力

        Args:
            x (str): x軸用の質的変数のカラム名
            y (str): y軸用の質的変数のカラム名
            rate_txt (bool, optional): 比率の数値をプロットする. デフォルト値はTrue.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: ヒートマップ
        '''
        pass
