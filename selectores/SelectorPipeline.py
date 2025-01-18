import pandas as pd
from selectores.SupervisedFeatureSelectorABC import SupervisedFeatureSelectorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.DataManager import DataManager

class SelectorPipeline:
    @ArgsChecker((None, DataManager, DataManager, DataManager, DataManager, list), None)
    def __init__(
        self,
        label_data_manager: DataManager,
        normalized_f_d_manager: DataManager,
        selected_f_d_manager: DataManager,
        selected_f_w_l_d_manager: DataManager,
        selectors: list,
    ):
        self.label_data_manager = label_data_manager
        self.normalized_f_d_manager = normalized_f_d_manager
        self.selected_f_d_manager = selected_f_d_manager
        self.selected_f_w_l_d_manager = selected_f_w_l_d_manager
        self.selectors = selectors

    @ArgsChecker((None, str), None)
    def run(self, symbol) -> None:
        """
        特徴量選択を一連の流れで実行するメソッド
        
        Args:
            symbol (str): 対象のシンボル名
        """
        # 正規化済みデータをロード
        df_normalized = self.normalized_f_d_manager.load_data(symbol)

        # date カラムを Timestamp 型に変換
        if "date" in df_normalized.columns:
            df_normalized["date"] = pd.to_datetime(df_normalized["date"])

        # ラベルデータを読み込んでマージ
        target_df = self.label_data_manager.load_data(symbol)
        if "date" in target_df.columns:
            target_df["date"] = pd.to_datetime(target_df["date"])

        df_with_label = df_normalized.merge(
            target_df[["date", "label"]],  # ラベルデータをマージ
            on=["date"],
            how="left",
        )

        # symbol を一行目から取得して保存
        symbol_value = (
            df_normalized["symbol"].iloc[0] if "symbol" in df_normalized.columns else None
        )

        # 不要なカラムをドロップ
        columns_to_drop = [
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

        # 各セレクターに初期データを渡して実行
        for selector in self.selectors:
            df_initial = df_pre.copy()
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

        # 除外されたカラムを表示
        excluded_columns = (
            set(df_pre.drop(columns=["label"]).columns) - selected_columns
        )
        print(f"selected feature columns({len(selected_columns)}): {selected_columns}")
        print(f"Excluded feature columns({len(excluded_columns)}): {excluded_columns}")

        # dateを追加
        df_selected["date"] = df_with_label["date"]

        # symbolを一律で追加
        if symbol_value is not None:
            df_selected["symbol"] = symbol_value

        # ラベル付きデータを準備
        df_selected_with_label = df_selected.copy()
        if "label" in df_with_label.columns:
            df_selected_with_label["label"] = df_with_label["label"]

        # カラムの順序を指定
        columns_order = ["date", "symbol"] + [
            col for col in df_selected.columns if col not in ["date", "symbol"]
        ]
        df_selected = df_selected[columns_order]

        columns_order_with_label = columns_order + ["label"]
        df_selected_with_label = df_selected_with_label[columns_order_with_label]

        # ラベル無しデータを先に保存
        self.selected_f_d_manager.save_data(df_selected, symbol)

        # ラベル付きデータを後に保存
        self.selected_f_w_l_d_manager.save_data(df_selected_with_label, symbol)

        print("Selector pipeline completed successfully")
