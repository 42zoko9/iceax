from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .trail import Trail


# TODO: 質的変数・時系列変数を含めた相関行列の出力方法について考える
class Trails:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def corr(self) -> None:
        """相関行列を出力"""
        pass

    def __summarise(
        self, keys: List[str], value: Optional[str] = None, func: str = "sum"
    ) -> pd.DataFrame:
        """可視化用の集計関数

        Args:
            keys (List[str]): 集計キーとなるカラム名一覧
            value (str, optional): 集計対象となるカラム名 デフォルト値はsum
            func (str, optional): 集計関数, とりうる値はcount, max, mean, min, quant, sum. デフォルト値はsum

        Returns:
            pd.DataFrame: 集計結果のデータフレーム
        """
        if type(keys) != list:
            raise TypeError('"keys" must be list.')
        invalid_key_cnt = len([k for k in keys if k not in self.df.columns])
        if invalid_key_cnt > 0:
            raise ValueError('"keys" is contained invalid name.')
        not_qualitive_key_cnt = len(
            [k for k in keys if self.df[k].dtype not in ("category", "datetime64[ns]")]
        )
        if not_qualitive_key_cnt > 0:
            raise ValueError('"keys" is contained invalid dtype.')

        if not (type(value) == str or value is None):
            raise TypeError('"value" must be str or None.')
        elif type(value) == str:
            if value not in self.df.columns:
                raise ValueError('"value" is invalid name.')
            elif self.df[value].dtype not in ("float", "int"):
                raise ValueError('"value" is invalid dtype.')

        func_lst = ["count", "max", "mean", "min", "quant", "sum"]
        if type(func) != str:
            raise TypeError('"func" must be str.')
        elif func not in func_lst:
            raise ValueError(
                '"func" must be "count", "max", "mean", "min", "quant" or "sum".'
            )

        if func != "count" and value is None:
            raise ValueError(
                'If "func" is not "count" then "value" must be column name.'
            )

        if func == "count":
            summarised_df = pd.DataFrame(self.df.groupby(keys).size())
        elif func == "quant":
            summarised_df = self.df.groupby(keys).agg({value: Trail._Trail__quantile})
        else:
            summarised_df = self.df.groupby(keys).agg({value: func})
        summarised_df.reset_index(inplace=True)
        if func == "count":
            summarised_df.rename(columns={0: "cnt"}, inplace=True)
            summarised_df["rate"] = np.round(
                (summarised_df["cnt"] / summarised_df["cnt"].sum()) * 100, 2
            )
        elif func == "quant":
            summarised_df = summarised_df.explode(value)
            summarised_df[value] = summarised_df[value].astype(float)
            summarised_df.reset_index(drop=True, inplace=True)
        return summarised_df

    def lineplot(
        self,
        x: str,
        y: str,
        func: str,
        color: Optional[str] = None,
        figsize: Tuple[Union[float, int], Union[float, int]] = (16.0, 9.0),
    ) -> Figure:
        """折れ線グラフを出力

        Args:
            x (str): x軸用の時系列変数のカラム名
            y (str): y軸用の量的変数のカラム名
            func (str): 集計用関数名. count, max, mean, min, quant, sum
            color (str, optional): 配色用の質的変数のカラム名. デフォルト値はNone.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ.デフォルト値は(16., 9.).

        Returns:
            Figure: 折れ線グラフ
        """
        if type(x) != str:
            raise TypeError('"x" must be str.')
        elif x not in self.df.columns:
            raise ValueError('"x" is invalid name.')
        elif self.df[x].dtype != "datetime64[ns]":
            raise ValueError('"x"s dtype must be datetime.')

        if type(y) != str:
            raise TypeError('"y" must be str.')
        elif y not in self.df.columns:
            raise ValueError('"y" is invalid name.')
        elif self.df[y].dtype not in ("float", "int"):
            raise ValueError('"y" is invalid dtype.')

        if not (type(color) == str or color is None):
            raise TypeError('"color" must be str or None.')
        elif type(color) == str:
            if color not in self.df.columns:
                raise ValueError('"color" is invalid name.')
            elif self.df[color].dtype != "category":
                raise ValueError('"color" is invalid dtype.')

        keys = [x] if color is None else [x, color]
        plot_df = self.__summarise(keys, y, func)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if color is None:
            sns.lineplot(data=plot_df, x=x, y=y, markers=True, ax=ax)
        else:
            sns.lineplot(
                data=plot_df, x=x, y=y, markers=True, hue=color, style=color, ax=ax
            )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.yaxis.set_major_formatter("{x:,.0f}")
        return fig

    def scatterplot(
        self,
        x: str,
        y: str,
        color: Optional[str] = None,
        figsize: Tuple[Union[float, int], Union[float, int]] = (16.0, 9.0),
    ) -> Figure:
        """散布図を出力

        Args:
            x (str): x軸用の量的変数のカラム名
            y (str): y軸用の量的変数のカラム名
            color (str, optional): 配色用の質的変数のカラム名. デフォルト値はNone.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ.デフォルト値は(16., 9.).

        Returns:
            Figure: 散布図
        """
        if type(x) != str:
            raise TypeError('"x" must be str.')
        elif x not in self.df.columns:
            raise ValueError('"x" is invalid name.')
        elif self.df[x].dtype not in ("float", "int"):
            raise ValueError('"x" is invalid dtype.')

        if type(y) != str:
            raise TypeError('"y" must be str.')
        elif y not in self.df.columns:
            raise ValueError('"y" is invalid name.')
        elif self.df[y].dtype not in ("float", "int"):
            raise ValueError('"y" is invalid dtype.')

        if type(color) not in (str, None):
            raise TypeError('"color" must be str or None.')
        elif color not in self.df.columns:
            raise ValueError('"color" is invalid name.')
        elif self.df[color].dtype != "category":
            raise ValueError('"color" is invalid dtype.')

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if color is None:
            sns.scatterplot(data=self.df, x=x, y=y, ax=ax)
        else:
            sns.scatterplot(data=self.df, x=x, y=y, hue=color, ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.xaxis.set_major_formatter("{x:,.0f}")
        ax.yaxis.set_major_formatter("{x:,.0f}")
        return fig

    def barplot(
        self,
        x: str,
        y: str,
        func: str = "sum",
        figsize: Tuple[Union[float, int], Union[float, int]] = (16.0, 9.0),
    ) -> Figure:
        """棒グラフを出力

        Args:
            x (str): x軸用の質的変数のカラム名
            y (str): y軸用の量的変数のカラム名
            func (str): 集計用の関数名. count, max, mean, min, quant, sumから選択. デフォルト値はsum.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: 棒グラフ
        """
        if type(x) != str:
            raise TypeError('"x" must be str.')
        elif x not in self.df.columns:
            raise ValueError('"x" is invalid name.')
        elif self.df[x].dtype != "category":
            raise ValueError('"x" is invalid dtype.')

        if type(y) != str:
            raise TypeError('"y" must be str.')
        elif y not in self.df.columns:
            raise ValueError('"y" is invalid name.')
        elif self.df[y].dtype not in ("float", "int"):
            raise ValueError('"y" is invalid dtype.')

        plot_df = self.__summarise(keys=[x], value=y, func=func)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        sns.barplot(data=plot_df, x=x, y=y, ax=ax)
        ax.yaxis.set_major_formatter("{x:,.0f}")
        return fig

    def stacked_barplot(
        self,
        x: str,
        y: str,
        color: str,
        func: str = "sum",
        rate_txt: bool = True,
        divline: bool = True,
        figsize: Tuple[Union[float, int], Union[float, int]] = (16.0, 9.0),
    ) -> Figure:
        """棒グラフを出力

        Args:
            x (str): x軸用の質的変数のカラム名
            y (str): y軸用の量的変数のカラム名
            color (str): 配色用の質的変数のカラム名
            func (str): 集計用の関数名. count, max, mean, min, quant, sumから選択. デフォルト値はsum.
            rate_txt (bool, optional): 比率の数値をプロットする. デフォルト値はTrue.
            divline (bool, optional): 区分線をプロットする. デフォルト値はTrue.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: 棒グラフ
        """
        if type(x) != str:
            raise TypeError('"x" must be str.')
        elif x not in self.df.columns:
            raise ValueError('"x" is invalid name.')
        elif self.df[x].dtype != "category":
            raise ValueError('"x" is invalid dtype.')

        if type(y) != str:
            raise TypeError('"y" must be str.')
        elif y not in self.df.columns:
            raise ValueError('"y" is invalid name.')
        elif self.df[y].dtype not in ("float", "int"):
            raise ValueError('"y" is invalid dtype.')

        if not (type(color) == str or color is None):
            raise TypeError('"color" must be str or None.')
        elif type(color) == str:
            if color not in self.df.columns:
                raise ValueError('"color" is invalid name.')
            elif self.df[color].dtype != "category":
                raise ValueError('"color" is invalid dtype.')

        if type(rate_txt) != bool:
            raise TypeError('"rate_txt" must be boolean.')

        if type(divline) != bool:
            raise TypeError('"divline" must be boolean.')

        plot_df = self.__summarise(keys=[x, color], value=y, func=func)
        summarise_df = plot_df.groupby([x]).agg({y: "sum"}).reset_index()
        summarise_df.rename(columns={y: "total"}, inplace=True)
        plot_df = pd.merge(plot_df, summarise_df, how="left", on=[x])
        plot_df.reset_index(drop=True, inplace=True)
        plot_df["rate"] = np.round((plot_df[y] / plot_df["total"]) * 100, 2)

        stack_values = list(plot_df[color].cat.categories)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for i, j in enumerate(stack_values):
            x_lst = plot_df.loc[plot_df[color] == j, x].tolist()
            y_lst = plot_df.loc[plot_df[color] == j, y].tolist()
            if i == 0:
                y_cumsum = [0.0] * len(y_lst)
            ax.bar(x_lst, y_lst, label=j, bottom=y_cumsum)
            if rate_txt is True:
                txt_lst = plot_df.loc[plot_df[color] == j, "rate"].tolist()
                txt_y_lst = [
                    a + b for (a, b) in zip(y_cumsum, [yi * 0.5 for yi in y_lst])
                ]
                for k, l in enumerate(txt_lst):
                    txt_x = x_lst[k]
                    txt_y = txt_y_lst[k]
                    numer = plot_df.loc[
                        (plot_df[x] == txt_x) & (plot_df[color] == j), y
                    ].values[0]
                    denom = plot_df.loc[plot_df[x] == txt_x, y].sum()
                    rate = numer / denom
                    if rate >= 0.1:
                        ax.text(txt_x, txt_y, l, ha="center", va="center")
            y_cumsum = [a + b for (a, b) in zip(y_cumsum, y_lst)]
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.yaxis.set_major_formatter("{x:,.0f}")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        return fig

    # TODO: 外れ値の可視化方法についての実装
    def boxplot(
        self,
        x: str,
        y: str,
        outlier: bool = True,
        figsize: Tuple[Union[float, int], Union[float, int]] = (16.0, 9.0),
    ) -> Figure:
        """箱ひげ図を出力

        Args:
            x (str): x軸用の質的変数のカラム名
            y (str): y軸用の量的変数のカラム名
            outlier (bool, optional): jitter-plotの描画判定. デフォルト値はTrue.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: 箱ひげ図
        """
        if type(x) != str:
            raise TypeError('"x" must be str.')
        elif x not in self.df.columns:
            raise ValueError('"x" is invalid name.')
        elif self.df[x].dtype != "category":
            raise ValueError('"x" is invalid dtype.')

        if type(y) != str:
            raise TypeError('"y" must be str.')
        elif y not in self.df.columns:
            raise ValueError('"y" is invalid name.')
        elif self.df[y].dtype not in ("float", "int"):
            raise ValueError('"y" is invalid dtype.')

        if type(outlier) != bool:
            raise TypeError('"outlier" must be boolean.')

        plot_df = self.__summarise([x], y, func="quant")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        sns.boxplot(data=plot_df, x=x, y=y, ax=ax)
        ax.yaxis.set_major_formatter("{x:,.0f}")
        return fig

    def heatmap(
        self,
        x: str,
        y: str,
        rate_txt: bool = True,
        figsize: Tuple[Union[float, int], Union[float, int]] = (16.0, 9.0),
    ) -> Figure:
        """出現頻度のヒートマップを出力

        Args:
            x (str): x軸用の質的変数のカラム名
            y (str): y軸用の質的変数のカラム名
            rate_txt (bool, optional): 比率の数値をプロットする. デフォルト値はTrue.
            figsize (Tuple[Union[float, int], Union[float, int]], optional): 画像サイズ，デフォルト値は(16., 9.)

        Returns:
            Figure: ヒートマップ
        """
        if type(x) != str:
            raise TypeError('"x" must be str.')
        elif x not in self.df.columns:
            raise ValueError('"x" is invalid name.')
        elif self.df[x].dtype != "category":
            raise ValueError('"x" is invalid dtype.')

        if type(y) != str:
            raise TypeError('"y" must be str.')
        elif y not in self.df.columns:
            raise ValueError('"y" is invalid name.')
        elif self.df[y].dtype != "category":
            raise ValueError('"y" is invalid dtype.')

        if type(rate_txt) != bool:
            raise TypeError('"rate_txt" must be boolean.')

        plot_df = self.__summarise(keys=[x, y], func="count")
        plot_df = plot_df.pivot(index="x", columns="y", values="rate")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        sns.heatmap(
            data=plot_df,
            vmin=0,
            vmax=100,
            cmap="BuPu",
            center=False,
            annot=rate_txt,
            ax=ax,
        )
        return fig
