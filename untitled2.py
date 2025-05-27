import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Konfigurasi tampilan halaman
st.set_page_config(page_title="K-Means Penjualan Barang", layout="centered")

st.title("📊 Klastering Data Penjualan Barang dengan K-Means")

# ✅ Data penjualan barang (langsung di dalam kode)
data = {
    'Kondisi Barang': ['Baru', 'Bekas', 'Baru', 'Bekas', 'Baru', 'Bekas', 'Baru', 'Bekas'],
    'Harga Barang (IDR)': [150000, 80000, 250000, 120000, 500000, 300000, 100000, 40000]
}
df_penjual = pd.DataFrame(data)

# Tampilkan data awal
st.subheader("📋 Data Penjualan")
st.dataframe(df_penjual)

# Encode 'Kondisi Barang' menjadi angka (0 = Bekas, 1 = Baru)
df_penjual['Kondisi Barang Numeric'] = df_penjual['Kondisi Barang'].apply(lambda x: 1 if x.lower() == 'baru' else 0)

# Pilih jumlah klaster
k = st.slider("🔢 Pilih jumlah klaster (k)", min_value=2, max_value=5, value=2)

# Ambil fitur numerik
X = df_penjual[['Kondisi Barang Numeric', 'Harga Barang (IDR)']].values

# K-Means Clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X)
df_penjual['Cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualisasi data awal sebelum klastering
st.subheader("📈 Visualisasi Data Awal (Sebelum Klastering)")
fig1, ax1 = plt.subplots()
for kondisi in df_penjual['Kondisi Barang'].unique():
    subset = df_penjual[df_penjual['Kondisi Barang'] == kondisi]
    ax1.scatter(subset['Kondisi Barang Numeric'], subset['Harga Barang (IDR)'], label=kondisi, s=80)
ax1.set_xlabel("Kondisi Barang (0=Bekas, 1=Baru)")
ax1.set_ylabel("Harga Barang (IDR)")
ax1.set_title("Penyebaran Data Penjualan")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Visualisasi hasil klastering
st.subheader("🧠 Visualisasi Hasil Klastering")
fig2, ax2 = plt.subplots()
scatter = ax2.scatter(df_penjual['Kondisi Barang Numeric'], df_penjual['Harga Barang (IDR)'],
                      c=df_penjual['Cluster'], cmap='viridis', s=100, alpha=0.8)
ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='red', label='Centroid')
ax2.set_xlabel("Kondisi Barang (0=Bekas, 1=Baru)")
ax2.set_ylabel("Harga Barang (IDR)")
ax2.set_title("Hasil Klastering Data Penjualan")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Menampilkan centroid
st.subheader("📍 Koordinat Centroid")
df_centroids = pd.DataFrame(centroids, columns=['Kondisi Barang Numeric', 'Harga Barang (IDR)'])
st.dataframe(df_centroids)

# Menampilkan data hasil klaster
st.subheader("📋 Data Penjualan + Klaster")
st.dataframe(df_penjual)
