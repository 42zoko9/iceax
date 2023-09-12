from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


class Trail:
    def __init__(self, df: pd.DataFrame, name: str) -> None:
        self.name = name
        try:
            self.series: pd.Series = df[name]
        except KeyError:
            raise KeyError('"{}" is not found.'.format(name))

    def stats(self) -> pd.DataFrame:
        '''対象の統計量を出力

        Returns:
            pd.DataFrame: 統計量をまとめたデータフレーム
        '''
        pass

    # NOTE: pandasのdescribeとほぼ同一
    def __quantitative_stats(self) -> pd.DataFrame:
        '''量的変数及び時系列の統計量(最小・四分位点・最頻値・最大)を出力

        Returns:
            pd.DataFrame: 量的変数の統計量をまとめたデータフレーム
        '''
        if self.series.dtype not in ('int', 'float'):
            raise TypeError('self.series is not quantitative.')
        s_quant = self.series.quantile([0.25, 0.5, 0.75])
        s_quant.index = ['q1', 'q2', 'q3']
        s_min = self.series.min()
        s_max = self.series.max()
        s_range = pd.Series({'min': s_min, 'max': s_max})
        stats = pd.concat([s_range, s_quant])
        df = pd.DataFrame(stats)
        df = df.loc[['min', 'q1', 'q2', 'q3', 'max'], :]
        df.reset_index(inplace=True)
        df.columns = ['key', 'value']
        return df

    def barplot(self, top: Optional[int] = None, figsize: Tuple[Union[float, int], Union[float, int]] = (16., 9.)) -> Figure:
        '''質的変数の値あたりの出現頻度を棒グラフとして出力

        Args:
            top (Optional[int], optional): 可視化する頻度の下限値, デフォルト値はNone
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: 棒グラフ
        '''
        df = self.__freq_cnt(top)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        x = df['key']
        y = df['value']
        ax.bar(x, y)
        ax.set_xlabel('variable')
        ax.set_ylabel('frequency')
        ax.yaxis.set_major_formatter('{x:,.0f}')
        ax.set_title(self.name)
        return fig

    def __freq_cnt(self, top: Optional[Union[float, int]] = None) -> pd.DataFrame:
        '''質的変数の値あたりの出現頻度を計算

        Args:
            top (Union[float, int], optional): 可視化する頻度の下限値, デフォルト値はNone

        Returns:
            pd.DataFrame: 出現頻度で降順でソートしたデータフレーム
        '''
        if self.series.dtype != 'object':
            raise TypeError('self.series is not qualitative.')
        if (top is not None) and (type(top) not in (float, int)):
            raise TypeError('"top" must be float, int or None.')
        cnt = self.series.value_counts(dropna=False)
        df = pd.DataFrame(cnt)
        df.sort_values('count', ascending=False, inplace=True)
        df.reset_index(inplace=True)
        df.columns = ['key', 'value']
        if top is not None:
            df = df.iloc[:top, :]
        return df

    def boxplot(self, outlier: bool = True, figsize: Tuple[Union[float, int], Union[float, int]] = (16., 9.)) -> Figure:
        '''量的変数の分布を箱ひげ図とjitter plotで出力

        Args:
            outlier (bool, optional): _description_. Defaults to True.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: 箱ひげ図
        '''
        if type(outlier) != bool:
            raise TypeError('"outlier" must be boolean.')
        x = self.__quantile(self.series)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        sns.boxplot(x=x, color='steelblue', showfliers=False, ax=ax)
        if outlier:
            sns.stripplot(x=self.series, dodge=True, jitter=True, color='black', ax=ax)
        ax.set_xlabel('value')
        ax.set_ylabel('variable')
        ax.xaxis.set_major_formatter('{x:,.0f}')
        ax.set_title(self.name)
        return fig

    @staticmethod
    def __quantile(series: pd.Series) -> List[float]:
        '''箱ひげ図出力用に対象の量的変数の統計量を出力

        Returns:
            List[float]: 四分位点の値が格納されたリスト
        '''
        if series.dtype not in ('int', 'float'):
            raise TypeError('series is not quantitative.')
        q = [0.25, 0.5, 0.75]
        quant = series.quantile(q)
        s_min = series.min()
        s_max = series.max()
        s_range = abs(quant[0.75] - quant[0.25]) * 1.5
        min_whis = quant[0.25] - s_range
        min_whis = min_whis if min_whis > s_min else s_min
        max_whis = quant[0.75] + s_range
        max_whis = max_whis if max_whis < s_max else s_max
        result = [min_whis] + quant.tolist() + [max_whis]
        return result

    def histgram(self, bins: Optional[List[Union[float, int]]] = None, figsize: Tuple[Union[float, int], Union[float, int]] = (16., 9.)) -> Figure:
        '''量的変数の値あたりの出現頻度をヒストグラムとして出力

        Args:
            bins (List[Union[float, int]], optional): ビンの幅, デフォルト値はNone.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: ヒストグラム
        '''
        if self.series.dtype not in ('int', 'float'):
            raise TypeError('self.series is not quantitative.')
        if type(bins) is not list:
            raise TypeError('"bins" must be list of real number.')
        invalid_length = len([b for b in bins if b not in (float, int)])
        if invalid_length > 0:
            raise ValueError('"bins" is contained invalid values.')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if bins is None:
            ax.hist(self.series, color='steelblue')
        else:
            ax.hist(self.series, bins=bins, color='steelblue')
        ax.set_xlabel('value')
        ax.set_ylabel('frequency')
        ax.xaxis.set_major_formatter('{x:,.0f}')
        ax.yaxis.set_major_formatter('{x:,.0f}')
        ax.set_title(self.name)
