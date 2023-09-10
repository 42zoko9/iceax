from iceax.trails import Trails


class TestCorr:
    pass


class TestSummarise:
    def test_invalid_keys_not_list(self) -> None:
        '''異常系: keysがリスト以外の型をとる
        '''
        pass

    def test_invalid_keys(self) -> None:
        '''異常系: keys内の値がself.dfの持つカラム名以外の値を持つ
        '''
        pass

    def test_invalid_func(self) -> None:
        '''異常系: funcが指定した値以外の値を取る
        '''
        pass

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass


class TestLineplot:
    def test_invalid_x(self) -> None:
        '''異常系: xがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_x_not_datetime(self) -> None:
        '''異常系: xが時系列変数ではない
        '''
        pass

    def test_invalid_y(self) -> None:
        '''異常系: yがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_y_not_quantitative(self) -> None:
        '''異常系: yが量的変数ではない
        '''
        pass

    def test_invalid_color(self) -> None:
        '''異常系: colorがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_color_not_qualitative(self) -> None:
        '''異常系: colorが質的変数ではない
        '''
        pass

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass


class TestScatterplot:
    def test_invalid_x(self) -> None:
        '''異常系: xがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_x_not_quantitative(self) -> None:
        '''異常系: xが量的変数ではない
        '''
        pass

    def test_invalid_y(self) -> None:
        '''異常系: yがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_y_not_quantitative(self) -> None:
        '''異常系: yが量的変数ではない
        '''
        pass

    def test_invalid_color(self) -> None:
        '''異常系: colorがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_color_not_qualitative(self) -> None:
        '''異常系: colorが質的変数ではない
        '''
        pass

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass


class TestBarplot:
    def test_invalid_x(self) -> None:
        '''異常系: xがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_x_not_qualitative(self) -> None:
        '''異常系: xが質的変数ではない
        '''
        pass

    def test_invalid_y(self) -> None:
        '''異常系: yがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_y_not_quantitative(self) -> None:
        '''異常系: yが量的変数ではない
        '''
        pass

    def test_invalid_color(self) -> None:
        '''異常系: colorがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_color_not_qualitative(self) -> None:
        '''異常系: colorが質的変数ではない
        '''
        pass

    def test_invalid_rate_txt(self) -> None:
        '''異常系: rate_txtがbool以外の値を取る
        '''
        pass

    def test_invalid_divline(self) -> None:
        '''異常系: divlineがbool以外の値を取る
        '''
        pass

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass


class TestBoxplot:
    def test_invalid_x(self) -> None:
        '''異常系: xがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_x_not_qualitative(self) -> None:
        '''異常系: xが質的変数ではない
        '''
        pass

    def test_invalid_y(self) -> None:
        '''異常系: yがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_y_not_quantitative(self) -> None:
        '''異常系: yが量的変数ではない
        '''
        pass

    def test_invalid_outlier(self) -> None:
        '''異常系: outlierがbool以外の値を取る
        '''
        pass

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass


class TestHeatmap:
    def test_invalid_x(self) -> None:
        '''異常系: xがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_x_not_qualitative(self) -> None:
        '''異常系: xが質的変数ではない
        '''
        pass

    def test_invalid_y(self) -> None:
        '''異常系: yがself.dfの持つカラム名以外の値を取る
        '''
        pass

    def test_invalid_y_not_qualitative(self) -> None:
        '''異常系: yが量的変数ではない
        '''
        pass

    def test_invalid_rate_txt(self) -> None:
        '''異常系: rate_txtがbool以外の値を取る
        '''
        pass

    def test_valid(self) -> None:
        '''正常系: 想定通りに出力される
        '''
        pass
