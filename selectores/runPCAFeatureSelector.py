import pandas as pd
from sklearn.decomposition import PCA

# データを準備する
data_path = "data/processed/demo_processed_stock_data.csv"
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])  # 日付をdatetime型に変換

# 'date' 列を除外して数値データのみを抽出
features = df.drop(columns=["date"])

# PCAのインスタンスを作成
pca = PCA(n_components=3)

# PCAを適用して特徴量を選択
selected_features = pca.fit_transform(features)

# 結果をデータフレームに変換して表示
df_selected_features = pd.DataFrame(selected_features, columns=["PC1", "PC2", "PC3"])
print(df_selected_features.head(10))
print(df_selected_features.tail(10))
