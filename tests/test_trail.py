import glob
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pytest
import seaborn as sns
from matplotlib.testing.compare import compare_images

from iceax.trail import Trail


@pytest.fixture
def valid_input_barplot() -> pd.DataFrame:
    l: List = ['a', 'a', 'a', 'b', 'b', 'c', 'e', 'e', np.NaN]
    df = pd.DataFrame({'tmp': l})
    return df


@pytest.fixture
def valid_output_barplot() -> str:
    valid_fn = 'valid.png'
    x = ['a', 'b', 'e']
    y = [3, 2, 2]
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.bar(x, y)
    ax.set_xlabel('variable')
    ax.set_ylabel('frequency')
    ax.yaxis.set_major_formatter('{x:,.0f}')
    ax.set_title('tmp')
    fig.savefig(valid_fn)
    yield valid_fn
    for fn in glob.glob('*.png'):
        os.remove(fn)


@pytest.fixture
def valid_input_boxplot() -> pd.DataFrame:
    l: List = [0, 0, 0, 2, 3, 7, 7, 8, 11]
    df = pd.DataFrame({'tmp': l})
    return df


@pytest.fixture
def valid_output_boxplot() -> str:
    valid_fn = 'valid.png'
    box_x = [0., 0., 3., 7., 11.]
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    sns.boxplot(x=box_x, color='steelblue', showfliers=False, ax=ax)
    ax.set_xlabel('value')
    ax.set_ylabel('variable')
    ax.xaxis.set_major_formatter('{x:,.0f}')
    ax.set_title('tmp')
    fig.savefig(valid_fn)
    yield valid_fn
    for fn in glob.glob('*.png'):
        os.remove(fn)


class TestQuantitativeStats:
    def test_invalid_not_quantitative(self) -> None:
        '''異常系: 指定したカラムが量的変数ではない
        '''
        invalid_df = pd.DataFrame({'tmp': ['a', 'b']})
        t = Trail(invalid_df, 'tmp')
        with pytest.raises(TypeError) as e:
            t._Trail__quantitative_stats()
        err_msg = e.value.args[0]
        assert err_msg == 'self.series is not quantitative.'

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        valid_result = pd.DataFrame({
            'key': ['min', 'q1', 'q2', 'q3', 'max'],
            'value': [0., 0., 3., 7., 11.]
        })
        valid_df = pd.DataFrame({'tmp': [0, 0, 0, 2, 3, 7, 7, 8, 11]})
        t = Trail(valid_df, 'tmp')
        result = t._Trail__quantitative_stats()
        pd.testing.assert_frame_equal(valid_result, result)


class TestFreqCnt:
    def test_invalid_not_qualitative(self) -> None:
        '''異常系: 指定したカラムが質的変数ではない
        '''
        invalid_df = pd.DataFrame({'tmp': [0, 1, 3]})
        t = Trail(invalid_df, 'tmp')
        with pytest.raises(TypeError) as e:
            t._Trail__freq_cnt()
        err_msg = e.value.args[0]
        assert err_msg == 'self.series is not qualitative.'

    def test_invalid_top(self) -> None:
        '''異常系: 引数topがNaN及び自然数以外の値を取る
        '''
        invalid_top = 'x'
        input_df = pd.DataFrame({'tmp': ['a', 'a', 'b']})
        t = Trail(input_df, 'tmp')
        with pytest.raises(TypeError) as e:
            t._Trail__freq_cnt(invalid_top)
        err_msg = e.value.args[0]
        assert err_msg == '"top" must be float, int or None.'

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        valid_top = 3
        valid_result = pd.DataFrame({
            'key': ['a', 'b', 'e'],
            'value': [3, 2, 2]
        })
        input_df = pd.DataFrame({'tmp': ['a', 'a', 'a', 'b', 'b', 'c', 'e', 'e', np.NaN]})
        t = Trail(input_df, 'tmp')
        result = t._Trail__freq_cnt(valid_top)
        pd.testing.assert_frame_equal(valid_result, result)


class TestBarplot:
    @pytest.mark.skip('TestFreqCnt::test_invalid_not_qualitativeと重複しているためスキップ')
    def test_invalid_not_qualitative(self) -> None:
        '''異常系: 指定したカラムが質的変数ではない
        '''
        pass

    @pytest.mark.skip('TestFreqCnt::test_invalid_topと重複しているためスキップ')
    def test_invalid_top(self) -> None:
        '''異常系: 引数topがNaN及び自然数以外の値を取る
        '''
        pass

    def test_valid(self, valid_input_barplot, valid_output_barplot) -> None:
        '''正常系: 想定通りに出力される
        '''
        result_fn = 'result.png'
        t = Trail(valid_input_barplot, 'tmp')
        fig = t.barplot(top=3)
        fig.savefig(result_fn)
        result = compare_images(valid_output_barplot, result_fn, 0.001)
        assert result is None


class TestQuantile:
    def test_invalid_not_quantitative(self) -> None:
        '''異常系: 指定したカラムが量的変数ではない
        '''
        invalid_df = pd.DataFrame({'tmp': [True, True, False]})
        t = Trail(invalid_df, 'tmp')
        with pytest.raises(TypeError) as e:
            t._Trail__quantile(t.series)
        err_msg = e.value.args[0]
        assert err_msg == 'series is not quantitative.'

    def test_valid(self, valid_input_boxplot) -> None:
        '''正常系: 想定通りに出力される
        '''
        valid_result = [0., 0., 3., 7., 11.]
        t = Trail(valid_input_boxplot, 'tmp')
        result = t._Trail__quantile(t.series)
        assert result == valid_result


class TestBoxplot:
    @pytest.mark.skip('TestQuantile::test_invalid_not_qualitativeと重複しているためスキップ')
    def test_invalid_not_quantitative(self) -> None:
        '''異常系: 指定したカラムが量的変数ではない
        '''
        pass

    def test_invalid_outlier(self) -> None:
        '''異常系: 引数outlierにbool値以外の値を取る
        '''
        invalid_outlier = 10
        input_df = pd.DataFrame({'tmp': [1, 1, 3, 3, 3, 2]})
        t = Trail(input_df, 'tmp')
        with pytest.raises(TypeError) as e:
            t.boxplot(invalid_outlier)
        err_msg = e.value.args[0]
        assert err_msg == '"outlier" must be boolean.'

    # NOTE: jitter-plotは乱数によって都度プロットされる点の位置が異なるためテスト対象から省略
    def test_valid(self, valid_input_boxplot, valid_output_boxplot) -> None:
        '''正常系: 想定通りに出力される
        '''
        result_fn = 'result.png'
        valid_outlier = False
        t = Trail(valid_input_boxplot, 'tmp')
        fig = t.boxplot(outlier=valid_outlier)
        fig.savefig(result_fn)
        result = compare_images(valid_output_boxplot, result_fn, 0.001)
        assert result is None


class TestHistgram:
    def test_invalid_not_quantitative(self) -> None:
        '''異常系: 指定したカラムが量的変数ではない
        '''
        invalid_df = pd.DataFrame({'tmp': ['a', 'b']})
        t = Trail(invalid_df, 'tmp')
        with pytest.raises(TypeError) as e:
            t.histgram()
        err_msg = e.value.args[0]
        assert err_msg == 'self.series is not quantitative.'

    def test_invalid_bins_type(self) -> None:
        '''正常系: binsがNaNまたはリスト以外が与えられている
        '''
        input_df = pd.DataFrame({'tmp': [1, 4, 7, 10]})
        invalid_bins = True
        t = Trail(input_df, 'tmp')
        with pytest.raises(TypeError) as e:
            t.histgram(bins=invalid_bins)
        err_msg = e.value.args[0]
        assert err_msg == '"bins" must be list of real number.'

    def test_invalid_bins_values(self) -> None:
        '''異常系: bins内に実数以外の値が含まれている
        '''
        input_df = pd.DataFrame({'tmp': [1, 4, 7, 10]})
        invalid_bins = [1, 3, 9, True]
        t = Trail(input_df, 'tmp')
        with pytest.raises(ValueError) as e:
            t.histgram(bins=invalid_bins)
        err_msg = e.value.args[0]
        assert err_msg == '"bins" is contained invalid values.'

    @pytest.mark.skip('テストの回答データを作成する手順が対象メソッドと同じになるためテストにならない')
    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass
