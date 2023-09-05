import pandas as pd

from .trail import Trail


class Pile:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def overview(self) -> pd.DataFrame:
        '''対象データフレームの列あたりの型・欠損数・欠損率・ユニーク数を出力

        Returns:
            pd.DataFrame: 対象データフレームの全体像
        '''
        pass

    def trail(self, col: str) -> None:
        '''引数colで指定したカラムのデータをTrailに格納

        Args:
            col (str): カラム名
        '''
        s = self.df[col]
        self.trail = Trail(s)
