import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib.testing.compare import compare_images

from iceax.trails import Trails


@pytest.fixture
def valid_input_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "x": ["a", "a", "a", "b", "b"],
            "y": ["i", "i", "j", "j", "k"],
            "z": [0.0, 3.0, 9.0, 10.0, 18.0],
            "w": [-1.5, 2.0, -21.0, np.NaN, 12.0],
            "t": ["2019-11-07", "2021-04-20", "2021-04-20", "2020-09-01", "2022-10-30"],
        }
    )
    df["x"] = pd.Categorical(df["x"], categories=["a", "b"], ordered=True)
    df["y"] = pd.Categorical(df["y"], categories=["i", "j", "k"], ordered=True)
    df["t"] = pd.to_datetime(df["t"])
    return df


@pytest.fixture
def valid_output_lineplot() -> str:
    valid_fn = "valid.png"
    df = pd.DataFrame(
        {
            "t": ["2019-11-07", "2021-04-20", "2020-09-01", "2022-10-30"],
            "x": ["a", "a", "b", "b"],
            "z": [0.0, 12.0, 10.0, 18.0],
        }
    )
    df["x"] = pd.Categorical(df["x"], categories=["a", "b"], ordered=True)
    df["t"] = pd.to_datetime(df["t"])
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    sns.lineplot(data=df, x="t", y="z", hue="x", style="x", markers=True, ax=ax)
    ax.set_xlabel("t")
    ax.set_ylabel("z")
    ax.yaxis.set_major_formatter("{x:,.0f}")
    fig.savefig(valid_fn)
    yield valid_fn
    for fn in glob.glob("*.png"):
        os.remove(fn)


@pytest.fixture
def valid_output_barplot() -> str:
    valid_fn = "valid.png"
    df = pd.DataFrame({"x": ["a", "b"], "w": [-20.5, 12.0]})
    df["x"] = pd.Categorical(df["x"], categories=["a", "b"], ordered=True)
    fig = plt.figure(figsize=(16.0, 9.0))
    ax = fig.add_subplot(111)
    sns.barplot(data=df, x="x", y="w", ax=ax)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    fig.savefig(valid_fn)
    yield valid_fn
    for fn in glob.glob("*.png"):
        os.remove(fn)


@pytest.fixture
def valid_output_stacked_barplot() -> str:
    valid_fn = "valid.png"
    df = pd.DataFrame(
        {
            "x": ["a"] * 3 + ["b"] * 3,
            "y": ["i", "j", "k"] * 2,
            "z": [3.0, 9.0, 0.0, 0.0, 10.0, 18.0],
            "rate": [25.0, 75.0, 0.0, 0.0, 35.71, 64.29],
        }
    )
    df["x"] = pd.Categorical(df["x"], categories=["a", "b"], ordered=True)
    df["y"] = pd.Categorical(df["y"], categories=["i", "j", "k"], ordered=True)
    fig = plt.figure(figsize=(16.0, 9.0))
    ax = fig.add_subplot(111)
    x_lst = list(df["x"].cat.categories)
    stack_lst = list(df["y"].cat.categories)
    for i, j in enumerate(stack_lst):
        if i == 0:
            y_cumsum = [0.0] * len(x_lst)
        y_lst = df.loc[df["y"] == j, "z"].tolist()
        ax.bar(x_lst, y_lst, label=j, bottom=y_cumsum)
        txt_lst = df.loc[df["y"] == j, "rate"].tolist()
        txt_y_lst = [a + b for (a, b) in zip(y_cumsum, [yi * 0.5 for yi in y_lst])]
        for k, l in enumerate(txt_lst):
            txt_x = x_lst[k]
            txt_y = txt_y_lst[k]
            numer = df.loc[(df["x"] == txt_x) & (df["y"] == j), "z"].values[0]
            denom = df.loc[df["x"] == txt_x, "z"].sum()
            rate = numer / denom
            if rate >= 0.1:
                ax.text(txt_x, txt_y, l, ha="center", va="center")
        y_cumsum = [a + b for (a, b) in zip(y_cumsum, y_lst)]
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.yaxis.set_major_formatter("{x:,.0f}")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    fig.savefig(valid_fn)
    yield valid_fn
    for fn in glob.glob("*.png"):
        os.remove(fn)


@pytest.fixture
def valid_output_boxplot() -> str:
    valid_fn = "valid.png"
    df = pd.DataFrame(
        {"x": (["a"] * 5) + (["b"] * 5), "z": [0, 1.5, 3, 6, 9, 10, 12, 14, 16, 18]}
    )
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    sns.boxplot(data=df, x="x", y="z", ax=ax)
    ax.yaxis.set_major_formatter("{x:,.0f}")
    fig.savefig(valid_fn)
    yield valid_fn
    for fn in glob.glob("*.png"):
        os.remove(fn)


@pytest.fixture
def valid_output_heatmap() -> str:
    valid_fn = "valid.png"
    df = pd.DataFrame({"i": [40, 0], "j": [20, 20], "k": [0, 20]}, index=["a", "b"])

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    sns.heatmap(data=df, vmin=0, vmax=100, cmap="BuPu", center=False, annot=True, ax=ax)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    fig.savefig(valid_fn)
    yield valid_fn
    for fn in glob.glob("*.png"):
        os.remove(fn)


# TODO: 質的変数・時系列変数を含めた相関行列の出力方法について考える
class TestCorr:
    pass


class TestSummarise:
    def test_invalid_keys_not_list(self, valid_input_df) -> None:
        """異常系: keysがlist以外の型となっている"""
        invalid_keys = "x"
        valid_value = "z"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t._Trails__summarise(invalid_keys, valid_value, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"keys" must be list.'

    def test_invalid_keys(self, valid_input_df) -> None:
        """異常系: keys内の値がself.dfの持つカラム名以外の値を持つ"""
        invalid_keys = ["xxx"]
        valid_value = "z"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t._Trails__summarise(invalid_keys, valid_value, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"keys" is contained invalid name.'

    def test_invalid_keys_not_qualitative(self, valid_input_df) -> None:
        """異常系: keysの値が質的変数ではない"""
        invalid_keys = ["x", "z"]
        valid_value = "w"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t._Trails__summarise(invalid_keys, valid_value, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"keys" is contained invalid dtype.'

    def test_invalie_value_not_str(self, valid_input_df) -> None:
        """異常系: valueがstr以外の型となっている"""
        valid_keys = ["x", "y"]
        invalid_value = ["z"]
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t._Trails__summarise(valid_keys, invalid_value, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"value" must be str or None.'

    def test_invalid_value(self, valid_input_df) -> None:
        """異常系: valueの値がself.dfの持つカラム名以外の値を持つ"""
        valid_keys = ["x", "y"]
        invalid_value = "zzz"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t._Trails__summarise(valid_keys, invalid_value, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"value" is invalid name.'

    def test_invalid_value_not_quantitative(self, valid_input_df) -> None:
        """異常系: valueの値が量的変数ではない"""
        valid_keys = ["x"]
        invalid_value = "y"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t._Trails__summarise(valid_keys, invalid_value, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"value" is invalid dtype.'

    def test_invalid_func_not_str(self, valid_input_df) -> None:
        """異常系: funcがstr以外の型となっている"""
        valid_keys = ["x"]
        valid_value = "z"
        invalid_func = ["sum"]
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t._Trails__summarise(valid_keys, valid_value, invalid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"func" must be str.'

    def test_invalid_func(self, valid_input_df) -> None:
        """異常系: funcが指定した値以外の値を取る"""
        valid_keys = ["x"]
        valid_value = "z"
        invalid_func = "avg"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t._Trails__summarise(valid_keys, valid_value, invalid_func)
        err_msg = e.value.args[0]
        assert (
            err_msg == '"func" must be "count", "max", "mean", "min", "quant" or "sum".'
        )

    def test_valid_count(self, valid_input_df) -> None:
        """正常系: func="count"のが想定通りに出力される"""
        valid_keys = ["x", "y"]
        valid_func = "count"
        t = Trails(valid_input_df)
        summarise_df = t._Trails__summarise(valid_keys, func=valid_func)
        valid_df = pd.DataFrame(
            {
                "x": ["a", "a", "a", "b", "b", "b"],
                "y": ["i", "j", "k", "i", "j", "k"],
                "cnt": [2, 1, 0, 0, 1, 1],
                "rate": [40.0, 20.0, 0.0, 0.0, 20.0, 20.0],
            }
        )
        valid_df["x"] = pd.Categorical(
            valid_df["x"], categories=["a", "b"], ordered=True
        )
        valid_df["y"] = pd.Categorical(
            valid_df["y"], categories=["i", "j", "k"], ordered=True
        )
        pd.testing.assert_frame_equal(valid_df, summarise_df)

    def test_valid_mean(self, valid_input_df) -> None:
        """正常系: func="mean"のが想定通りに出力される"""
        valid_keys = ["x"]
        valid_value = "z"
        valid_func = "mean"
        t = Trails(valid_input_df)
        summarise_df = t._Trails__summarise(valid_keys, valid_value, valid_func)
        valid_df = pd.DataFrame(
            {
                "x": ["a", "b"],
                "z": [4.0, 14.0],
            }
        )
        valid_df["x"] = pd.Categorical(
            valid_df["x"], categories=["a", "b"], ordered=True
        )
        pd.testing.assert_frame_equal(valid_df, summarise_df)

    def test_valid_quant(self, valid_input_df) -> None:
        """正常系: func="quant"のが想定通りに出力される"""
        valid_keys = ["x"]
        valid_value = "z"
        valid_func = "quant"
        t = Trails(valid_input_df)
        summarise_df = t._Trails__summarise(valid_keys, valid_value, valid_func)
        valid_df = pd.DataFrame(
            {"x": (["a"] * 5) + (["b"] * 5), "z": [0, 1.5, 3, 6, 9, 10, 12, 14, 16, 18]}
        )
        valid_df["x"] = pd.Categorical(
            valid_df["x"], categories=["a", "b"], ordered=True
        )
        pd.testing.assert_frame_equal(valid_df, summarise_df)

    def test_valid_sum(self, valid_input_df) -> None:
        """正常系: func="sum"のが想定通りに出力される"""
        valid_keys = ["x", "y"]
        valid_value = "z"
        valid_func = "sum"
        t = Trails(valid_input_df)
        summarise_df = t._Trails__summarise(valid_keys, valid_value, valid_func)
        valid_df = pd.DataFrame(
            {
                "x": ["a", "a", "a", "b", "b", "b"],
                "y": ["i", "j", "k", "i", "j", "k"],
                "z": [3, 9, np.NaN, np.NaN, 10, 18],
            }
        )
        valid_df["x"] = pd.Categorical(
            valid_df["x"], categories=["a", "b"], ordered=True
        )
        valid_df["y"] = pd.Categorical(
            valid_df["y"], categories=["i", "j", "k"], ordered=True
        )
        pd.testing.assert_frame_equal(valid_df, summarise_df)

    @pytest.mark.skip("取り敢えずスキップ")
    def test_valid_others(self, valid_input_df) -> None:
        """正常系: sum, mean, quant以外の処理にて想定通りに出力される"""
        pass


class TestLineplot:
    def test_invalie_x_not_str(self, valid_input_df) -> None:
        """異常系: xがstr以外の型となっている"""
        invalid_x = ["t"]
        valid_y = "w"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.lineplot(invalid_x, valid_y, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"x" must be str.'

    def test_invalid_x(self, valid_input_df) -> None:
        """異常系: xがself.dfの持つカラム名以外の値を取る"""
        invalid_x = "ttt"
        valid_y = "w"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.lineplot(invalid_x, valid_y, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid name.'

    def test_invalid_x_not_quantitative(self, valid_input_df) -> None:
        """異常系: xが時系列変数ではない"""
        invalid_x = "z"
        valid_y = "w"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.lineplot(invalid_x, valid_y, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"x"s dtype must be datetime.'

    def test_invalie_y_not_str(self, valid_input_df) -> None:
        """異常系: yがstr以外の型となっている"""
        valid_x = "t"
        invalid_y = True
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.lineplot(valid_x, invalid_y, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"y" must be str.'

    def test_invalid_y(self, valid_input_df) -> None:
        """異常系: yがself.dfの持つカラム名以外の値を取る"""
        valid_x = "t"
        invalid_y = "www"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.lineplot(valid_x, invalid_y, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid name.'

    def test_invalid_y_not_quantitative(self, valid_input_df) -> None:
        """異常系: yが量的変数ではない"""
        valid_x = "t"
        invalid_y = "x"
        valid_func = "sum"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.lineplot(valid_x, invalid_y, valid_func)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid dtype.'

    def test_invalie_color_not_str(self, valid_input_df) -> None:
        """異常系: colorがstr以外の型となっている"""
        valid_x = "t"
        valid_y = "w"
        valid_func = "sum"
        invalid_color = ["x"]
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.lineplot(valid_x, valid_y, valid_func, invalid_color)
        err_msg = e.value.args[0]
        assert err_msg == '"color" must be str or None.'

    def test_invalid_color(self, valid_input_df) -> None:
        """異常系: colorがself.dfの持つカラム名以外の値を取る"""
        valid_x = "t"
        valid_y = "w"
        valid_func = "sum"
        invalid_color = "xxx"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.lineplot(valid_x, valid_y, valid_func, invalid_color)
        err_msg = e.value.args[0]
        assert err_msg == '"color" is invalid name.'

    def test_invalid_color_not_qualitative(self, valid_input_df) -> None:
        """異常系: colorが質的変数ではない"""
        valid_x = "t"
        valid_y = "w"
        valid_func = "sum"
        invalid_color = "z"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.lineplot(valid_x, valid_y, valid_func, invalid_color)
        err_msg = e.value.args[0]
        assert err_msg == '"color" is invalid dtype.'

    @pytest.mark.skip("TestSummariseで同様の検証を実施しているためスキップ")
    def test_invalid_func(self) -> None:
        """異常系: funcが想定外の値を取る"""
        pass

    def test_valid(self, valid_input_df, valid_output_lineplot) -> None:
        """正常系: 想定通りに出力される"""
        result_fn = "result.png"
        valid_x = "t"
        valid_y = "z"
        valid_func = "sum"
        valid_color = "x"
        t = Trails(valid_input_df)
        fig = t.lineplot(valid_x, valid_y, valid_func, valid_color)
        fig.savefig(result_fn)
        result = compare_images(valid_output_lineplot, result_fn, 0.001)
        assert result is None


class TestScatterplot:
    def test_invalie_x_not_str(self, valid_input_df) -> None:
        """異常系: xがstr以外の型となっている"""
        invalid_x = ["z"]
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.scatterplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" must be str.'

    def test_invalid_x(self, valid_input_df) -> None:
        """異常系: xがself.dfの持つカラム名以外の値を取る"""
        invalid_x = "zzz"
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.scatterplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid name.'

    def test_invalid_x_not_quantitative(self, valid_input_df) -> None:
        """異常系: xが量的変数ではない"""
        invalid_x = "x"
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.scatterplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid dtype.'

    def test_invalie_y_not_str(self, valid_input_df) -> None:
        """異常系: yがstr以外の型となっている"""
        valid_x = "z"
        invalid_y = True
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.scatterplot(valid_x, invalid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" must be str.'

    def test_invalid_y(self, valid_input_df) -> None:
        """異常系: yがself.dfの持つカラム名以外の値を取る"""
        valid_x = "z"
        invalid_y = "www"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.scatterplot(valid_x, invalid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid name.'

    def test_invalid_y_not_quantitative(self, valid_input_df) -> None:
        """異常系: yが量的変数ではない"""
        valid_x = "z"
        invalid_y = "x"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.scatterplot(valid_x, invalid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid dtype.'

    def test_invalie_color_not_str(self, valid_input_df) -> None:
        """異常系: colorがstr以外の型となっている"""
        valid_x = "z"
        valid_y = "w"
        invalid_color = ["x"]
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.scatterplot(valid_x, valid_y, invalid_color)
        err_msg = e.value.args[0]
        assert err_msg == '"color" must be str or None.'

    def test_invalid_color(self, valid_input_df) -> None:
        """異常系: colorがself.dfの持つカラム名以外の値を取る"""
        valid_x = "z"
        valid_y = "w"
        invalid_color = "xxx"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.scatterplot(valid_x, valid_y, invalid_color)
        err_msg = e.value.args[0]
        assert err_msg == '"color" is invalid name.'

    def test_invalid_color_not_qualitative(self, valid_input_df) -> None:
        """異常系: colorが質的変数ではない"""
        valid_x = "z"
        valid_y = "w"
        invalid_color = "z"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.scatterplot(valid_x, valid_y, invalid_color)
        err_msg = e.value.args[0]
        assert err_msg == '"color" is invalid dtype.'

    @pytest.mark.skip("検証用結果の出力手順が実装済みとほぼ同じであるため")
    def test_valid(self) -> None:
        """正常系: 想定通りに出力される"""
        pass


class TestBarplot:
    def test_invalie_x_not_str(self, valid_input_df) -> None:
        """異常系: xがstr以外の型となっている"""
        invalid_x = ["x"]
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" must be str.'

    def test_invalid_x(self, valid_input_df) -> None:
        """異常系: xがself.dfの持つカラム名以外の値を取る"""
        invalid_x = "xxx"
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid name.'

    def test_invalid_x_not_qualitative(self, valid_input_df) -> None:
        """異常系: xが質的変数ではない"""
        invalid_x = "z"
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid dtype.'

    def test_invalie_y_not_str(self, valid_input_df) -> None:
        """異常系: yがstr以外の型となっている"""
        invalid_x = "x"
        valid_y = ["w"]
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" must be str.'

    def test_invalid_y(self, valid_input_df) -> None:
        """異常系: yがself.dfの持つカラム名以外の値を取る"""
        invalid_x = "x"
        valid_y = "www"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid name.'

    def test_invalid_y_not_quantitative(self, valid_input_df) -> None:
        """異常系: yが量的変数ではない"""
        invalid_x = "x"
        valid_y = "y"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid dtype.'

    @pytest.mark.skip("TestSummariseで同様の検証を実施しているためスキップ")
    def test_invalid_func(self) -> None:
        """異常系: funcが不正な値を取る"""
        pass

    def test_valid(self, valid_input_df, valid_output_barplot) -> None:
        """正常系: 想定通りに出力される"""
        result_fn = "result.png"
        valid_x = "x"
        valid_y = "w"
        valid_func = "sum"
        t = Trails(valid_input_df)
        fig = t.barplot(valid_x, valid_y, valid_func)
        fig.savefig(result_fn)
        result = compare_images(valid_output_barplot, result_fn, 0.001)
        assert result is None


class TestStackedBarplot:
    def test_invalie_x_not_str(self, valid_input_df) -> None:
        """異常系: xがstr以外の型となっている"""
        invalid_x = ["x"]
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" must be str.'

    def test_invalid_x(self, valid_input_df) -> None:
        """異常系: xがself.dfの持つカラム名以外の値を取る"""
        invalid_x = "xxx"
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid name.'

    def test_invalid_x_not_qualitative(self, valid_input_df) -> None:
        """異常系: xが質的変数ではない"""
        invalid_x = "z"
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid dtype.'

    def test_invalie_y_not_str(self, valid_input_df) -> None:
        """異常系: yがstr以外の型となっている"""
        invalid_x = "x"
        valid_y = ["w"]
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" must be str.'

    def test_invalid_y(self, valid_input_df) -> None:
        """異常系: yがself.dfの持つカラム名以外の値を取る"""
        invalid_x = "x"
        valid_y = "www"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid name.'

    def test_invalid_y_not_quantitative(self, valid_input_df) -> None:
        """異常系: yが量的変数ではない"""
        invalid_x = "x"
        valid_y = "y"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid dtype.'

    def test_invalie_color_not_str(self, valid_input_df) -> None:
        """異常系: yがstr以外の型となっている"""
        valid_x = "x"
        valid_y = "w"
        invalid_color = ["y"]
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.barplot(valid_x, valid_y, invalid_color)
        err_msg = e.value.args[0]
        assert err_msg == '"color" must be str or None.'

    def test_invalid_color(self, valid_input_df) -> None:
        """異常系: colorがself.dfの持つカラム名以外の値を取る"""
        valid_x = "x"
        valid_y = "w"
        invalid_color = "yyy"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(valid_x, valid_y, invalid_color)
        err_msg = e.value.args[0]
        assert err_msg == '"color" is invalid name.'

    def test_invalid_color_not_qualitative(self, valid_input_df) -> None:
        """異常系: colorが質的変数ではない"""
        valid_x = "x"
        valid_y = "w"
        invalid_color = "z"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.barplot(valid_x, valid_y, invalid_color)
        err_msg = e.value.args[0]
        assert err_msg == '"color" is invalid dtype.'

    def test_invalid_rate_txt(self, valid_input_df) -> None:
        """異常系: rate_txtがbool以外の値を取る"""
        valid_x = "x"
        valid_y = "w"
        valid_color = "y"
        invalid_rate_txt = "True"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.barplot(valid_x, valid_y, valid_color, rate_txt=invalid_rate_txt)
        err_msg = e.value.args[0]
        assert err_msg == '"rate_txt" must be boolean.'

    def test_invalid_divline(self, valid_input_df) -> None:
        """異常系: divlineがbool以外の値を取る"""
        valid_x = "x"
        valid_y = "w"
        valid_color = "y"
        invalid_divline = "True"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.barplot(valid_x, valid_y, valid_color, divline=invalid_divline)
        err_msg = e.value.args[0]
        assert err_msg == '"divline" must be boolean.'

    @pytest.mark.skip("TestSummariseで同様の検証を実施しているためスキップ")
    def test_invalid_func(self) -> None:
        """異常系: funcが不正な値を取る"""
        pass

    def test_valid(self, valid_input_df, valid_output_stacked_barplot) -> None:
        """正常系: 想定通りに出力される"""
        result_fn = "result.png"
        valid_x = "x"
        valid_y = "z"
        valid_func = "sum"
        valid_color = "y"
        valid_rate_txt = True
        t = Trails(valid_input_df)
        fig = t.stacked_barplot(
            valid_x, valid_y, valid_color, valid_func, valid_rate_txt
        )
        fig.savefig(result_fn)
        result = compare_images(valid_output_stacked_barplot, result_fn, 0.001)
        assert result is None


class TestBoxplot:
    def test_invalie_x_not_str(self, valid_input_df) -> None:
        """異常系: xがstr以外の型となっている"""
        invalid_x = ["x"]
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.boxplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" must be str.'

    def test_invalid_x(self, valid_input_df) -> None:
        """異常系: xがself.dfの持つカラム名以外の値を取る"""
        invalid_x = "xxx"
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.boxplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid name.'

    def test_invalid_x_not_qualitative(self, valid_input_df) -> None:
        """異常系: xが質的変数ではない"""
        invalid_x = "z"
        valid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.boxplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid dtype.'

    def test_invalie_y_not_str(self, valid_input_df) -> None:
        """異常系: yがstr以外の型となっている"""
        invalid_x = "x"
        valid_y = ["w"]
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.boxplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" must be str.'

    def test_invalid_y(self, valid_input_df) -> None:
        """異常系: yがself.dfの持つカラム名以外の値を取る"""
        invalid_x = "x"
        valid_y = "www"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.boxplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid name.'

    def test_invalid_y_not_quantitative(self, valid_input_df) -> None:
        """異常系: yが量的変数ではない"""
        invalid_x = "x"
        valid_y = "y"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.boxplot(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid dtype.'

    def test_invalid_outlier(self, valid_input_df) -> None:
        """異常系: outlierがbool以外の値を取る"""
        valid_x = "x"
        valid_y = "z"
        invalid_outlier = "y"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.boxplot(valid_x, valid_y, outlier=invalid_outlier)
        err_msg = e.value.args[0]
        assert err_msg == '"outlier" must be boolean.'

    def test_valid(self, valid_input_df, valid_output_boxplot) -> None:
        """正常系: 想定通りに出力される"""
        result_fn = "result.png"
        valid_x = "x"
        valid_y = "z"
        t = Trails(valid_input_df)
        fig = t.boxplot(x=valid_x, y=valid_y)
        fig.savefig(result_fn)
        result = compare_images(valid_output_boxplot, result_fn, 0.001)
        assert result is None


class TestHeatmap:
    def test_invalie_x_not_str(self, valid_input_df) -> None:
        """異常系: xがstr以外の型となっている"""
        invalid_x = ["x"]
        valid_y = "y"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.heatmap(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" must be str.'

    def test_invalid_x(self, valid_input_df) -> None:
        """異常系: xがself.dfの持つカラム名以外の値を取る"""
        invalid_x = "xxx"
        valid_y = "y"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.heatmap(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid name.'

    def test_invalid_x_not_qualitative(self, valid_input_df) -> None:
        """異常系: xが質的変数ではない"""
        invalid_x = "z"
        valid_y = "y"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.heatmap(invalid_x, valid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"x" is invalid dtype.'

    def test_invalie_y_not_str(self, valid_input_df) -> None:
        """異常系: yがstr以外の型となっている"""
        valid_x = "x"
        invalid_y = ["y"]
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.heatmap(valid_x, invalid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" must be str.'

    def test_invalid_y(self, valid_input_df) -> None:
        """異常系: yがself.dfの持つカラム名以外の値を取る"""
        valid_x = "x"
        invalid_y = "yyy"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.heatmap(valid_x, invalid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid name.'

    def test_invalid_y_not_qualitative(self, valid_input_df) -> None:
        """異常系: yが質的変数ではない"""
        valid_x = "x"
        invalid_y = "w"
        t = Trails(valid_input_df)
        with pytest.raises(ValueError) as e:
            t.heatmap(valid_x, invalid_y)
        err_msg = e.value.args[0]
        assert err_msg == '"y" is invalid dtype.'

    def test_invalid_rate_txt(self, valid_input_df) -> None:
        """異常系: rate_txtがbool以外の値を取る"""
        valid_x = "x"
        valid_y = "y"
        invalid_rate_txt = "True"
        t = Trails(valid_input_df)
        with pytest.raises(TypeError) as e:
            t.heatmap(valid_x, valid_y, rate_txt=invalid_rate_txt)
        err_msg = e.value.args[0]
        assert err_msg == '"rate_txt" must be boolean.'

    def test_valid(self, valid_input_df, valid_output_heatmap) -> None:
        """正常系: 想定通りに出力される"""
        result_fn = "result.png"
        valid_x = "x"
        valid_y = "y"
        valid_rate_txt = True
        t = Trails(valid_input_df)
        fig = t.heatmap(valid_x, valid_y, valid_rate_txt)
        fig.savefig(result_fn)
        result = compare_images(valid_output_heatmap, result_fn, 0.001)
        assert result is None
