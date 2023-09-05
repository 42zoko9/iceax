import pandas as pd
import pytest

from iceax.trail import Trail


class TestQuantitativeStats:
    def test_invalid_not_quantitative(self) -> None:
        '''異常系: 指定したカラムが量的変数ではない
        '''
        pass

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass


class TestFreqCnt:
    def test_invalid_not_qualitative(self) -> None:
        '''異常系: 指定したカラムが質的変数ではない
        '''
        pass

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass


class TestBarplot:
    def test_invalid_not_qualitative(self) -> None:
        '''異常系: 指定したカラムが質的変数ではない
        '''
        pass

    def test_invalid_top(self) -> None:
        '''異常系: 引数topがNaN及び自然数以外の値を取る
        '''
        pass


class TestQuantile:
    def test_invalid_not_quantitative(self) -> None:
        '''異常系: 指定したカラムが量的変数ではない
        '''
        pass

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass


class TestBoxplot:
    def test_invalid_not_quantitative(self) -> None:
        '''異常系: 指定したカラムが量的変数ではない
        '''
        pass

    def test_invalid_outlier(self) -> None:
        '''異常系: 引数outlierにbool値以外の値を取る
        '''
        pass


class TestHistgram:
    def test_invalid_not_quantitative(self) -> None:
        '''異常系: 指定したカラムが量的変数ではない
        '''
        pass

    def test_invalid_width(self) -> None:
        '''正常系: widthがNaNまたは0より大きな実数以外の値を取る
        '''
        pass
