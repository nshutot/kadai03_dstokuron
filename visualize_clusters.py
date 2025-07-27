import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- ファイル読み込み ----------
files = [
    r"C:\Users\Owner\Pictures\Camera Roll\院１\火曜四限 データ・サイエンス特論\hitter_score_2010.csv"
]
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# ---------- 特徴量選択 ----------
features = [
    '打率',     # 打率
    'IsoP',     # パワーの指標
    '盗塁',       # 機動力
    '犠打',       # 小技・チームプレイ
    '三振',
    '四球'
]
data = df[features].copy()
data.dropna(inplace=True)

# ---------- 標準化 ----------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# ---------- ① エルボー法 ----------
inertia_list = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia_list.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia_list, marker='o')
plt.xlabel("クラスタ数 (k)")
plt.ylabel("クラスタ内誤差平方和 (Inertia)")
plt.title("エルボー法による最適クラスタ数の検討")
plt.grid(True)
plt.show()

# ---------- ② クラスタリング ----------
k_opt = 3  # ←必要に応じて変更
kmeans = KMeans(n_clusters=k_opt, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)

# ---------- ③ PCAで可視化 ----------
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_data)
df['PCA1'] = reduced[:, 0]
df['PCA2'] = reduced[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=df, palette='Set2')
plt.title("2024")  # 年度名は必要に応じて調整

# ▼ 軸を全年度共通に固定
plt.xlim(-5, 8)
plt.ylim(-3, 6)

plt.grid(False)
plt.show()


# ---------- ⑤ 人数と割合 ----------
cluster_counts = df['cluster'].value_counts().sort_index()
cluster_percentages = df['cluster'].value_counts(normalize=True).sort_index() * 100

cluster_summary = pd.DataFrame({
    '人数': cluster_counts,
    '割合（%）': cluster_percentages.round(1)
})

print("\n▼ クラスタごとの人数と割合")
print(cluster_summary.to_string())