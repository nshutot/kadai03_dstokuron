import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------- ファイル読み込み ----------
df = pd.read_csv(r"C:\Users\Owner\Pictures\Camera Roll\院１\火曜四限 データ・サイエンス特論\hitter_score_2024.csv")


# ---------- 特徴量選択 ----------
features = [
    '打率',     # 打率
    'IsoP',     # パワーの指標
    '盗塁',     # 機動力
    '犠打',     # 小技・チームプレイ
    '三振',
    '四球'
]
data = df[features].copy()
data.dropna(inplace=True)

# ---------- 標準化 ----------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# ---------- クラスタリング ----------
k_opt = 3  # 最適クラスタ数（必要に応じて調整）
kmeans = KMeans(n_clusters=k_opt, random_state=42)
df = df.loc[data.index].copy()  # 欠損を除いた index に合わせる
df['cluster'] = kmeans.fit_predict(scaled_data)

# ---------- クラスタごとの特徴量平均 ----------
cluster_means = df.groupby('cluster')[features].mean().round(2)

print("\n▼ クラスタごとの特徴量平均（2010年）")
print(cluster_means.to_string())

# ---------- クラスタごとの人数と割合 ----------
cluster_counts = df['cluster'].value_counts().sort_index()
cluster_percentages = df['cluster'].value_counts(normalize=True).sort_index() * 100

cluster_summary = pd.DataFrame({
    '人数': cluster_counts,
    '割合（%）': cluster_percentages.round(1)
})

print("\n▼ クラスタごとの人数と割合（2010年）")
print(cluster_summary.to_string())

# ---------- クラスタごとの特徴量平均をCSV出力 ----------
# 人数・割合の表に 'cluster' をインデックスに追加（揃えるため）
cluster_summary.index.name = 'cluster'

# concatで上下結合（空行を挟むために1行空のDataFrameを入れる）
# ▼ インデックスで横に結合（ズレない）
combined = pd.concat([cluster_means, cluster_summary], axis=1)
combined.index.name = 'cluster'  # インデックス名を明示（任意）


output_path = r"C:\Users\Owner\Pictures\Camera Roll\院１\火曜四限 データ・サイエンス特論\クラスタリング結果\改特徴量平均2024.csv"
combined.to_csv(output_path, encoding='utf-8-sig')
print(f"\n▼ 特徴量平均＋人数・割合をCSV出力しました → {output_path}")
