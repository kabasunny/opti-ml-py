import pandas as pd
from selectores.SupervisedFeatureSelectorABC import SupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.DataManager import DataManager


class SelectorPipeline:
    @ArgsChecker((None, DataManager, DataManager, DataManager, list), None)
    def __init__(
        self,
        target_data_manager: DataManager,
        normalized_f_d_manager: DataManager,
        selected_f_d_manager: DataManager,
        selectors: list,
    ):
        self.target_data_manager = target_data_manager
        self.normalized_f_d_manager = normalized_f_d_manager
        self.selected_f_d_manager = selected_f_d_manager
        self.selectors = selectors

    @ArgsChecker((), None)
    def run(self) -> None:
        """
        特徴量選択を一連の流れで実行するメソッド

        Returns:
            pd.DataFrame: 最終的に選択された特徴量のみを含むデータフレーム
        """
        print("Run Selector pipeline")
        # 正規化済みデータをロード
        df_normalized = self.normalized_f_d_manager.load_data()

        # date カラムを Timestamp 型に変換
        if "date" in df_normalized.columns:
            df_normalized["date"] = pd.to_datetime(df_normalized["date"])

        # ラベルデータを読み込んでマージ
        target_df = self.target_data_manager.load_data()
        if "date" in target_df.columns:
            target_df["date"] = pd.to_datetime(target_df["date"])
        df_with_label = df_normalized.merge(
            target_df[["date", "label"]],  # ラベルデータをマージ
            on=["date"],
            how="left",
        )

        # symbol を一行目から取得して保存
        symbol_value = (
            df_normalized["symbol"].iloc[0]
            if "symbol" in df_normalized.columns
            else None
        )

        # 不要なカラムをドロップ
        columns_to_drop = [
            "Unnamed: 0",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        df_with_label = df_with_label.drop(
            columns=[col for col in columns_to_drop if col in df_with_label.columns]
        )

        # 初期データをコピーして保持
        df_selected = pd.DataFrame()
        selected_columns = set()  # 選択された特徴量を保持するためのセット
        df_pre = df_with_label.drop(columns=["date"])  # date列を削除
        c = 0
        # 各セレクターに初期データを渡して実行
        for selector in self.selectors:
            df_initial = df_pre.copy()
            # if c == 0:
            #     print(
            #         f"dataframe before feature selection:{selector.__class__.__name__}\n {df_initial.head()}"
            #     )
            if isinstance(selector, SupervisedFeatureSelectorABC):
                # 特徴量選択（ラベル付き）
                df_temp = selector.select_features(df_initial, "label")
            else:
                # 特徴量選択（ラベルなし）
                df_temp = selector.select_features(df_initial.drop(columns=["label"]))
            # 重複する特徴量を追加しないように選択
            for col in df_temp.columns:
                if col not in selected_columns:
                    df_selected[col] = df_temp[col]
                    selected_columns.add(col)

            # c += 1
            # print(
            #     f"{c} dataframe after feature selection:{selector.__class__.__name__}\n {df_selected.head()}"
            # )

        # 除外されたカラムを表示
        excluded_columns = (
            set(df_pre.drop(columns=["label"]).columns) - selected_columns
        )
        print(f"Excluded feture columns: {excluded_columns}")

        # dateを追加
        df_selected["date"] = df_with_label["date"]

        # symbolを一律で追加
        if symbol_value is not None:
            df_selected["symbol"] = symbol_value

        # カラムの順序を指定して保存
        columns_order = ["date", "symbol"] + [
            col for col in df_selected.columns if col not in ["date", "symbol"]
        ]
        df_selected = df_selected[columns_order]

        # 結果を保存
        self.selected_f_d_manager.save_data(df_selected)

        print("Selector pipeline completed successfully")
