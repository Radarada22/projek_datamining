import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel('pengangguran.xlsx')

# Drop unnecessary columns
X = df.drop(['id', 'kode_provinsi', 'nama_provinsi', 'kode_kabupaten_kota', 'satuan'], axis=1)

st.header("Isi dataset")
st.write(X)

# Get dummies for categorical columns
df = pd.get_dummies(df, columns=['nama_kabupaten_kota', 'pendidikan'], drop_first=True)

# Label encode for remaining categorical columns
label_encoder = LabelEncoder()
X['nama_kabupaten_kota'] = label_encoder.fit_transform(X['nama_kabupaten_kota'])
X['pendidikan'] = label_encoder.fit_transform(X['pendidikan'])

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# KMeans clustering
i = 3
km = KMeans(n_clusters=i).fit(X_imputed)

# Elbow plot
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X_imputed)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

ax.annotate('Possible elbow point', xy=(2, 140000), xytext=(2, 50000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
ax.annotate('Possible elbow point', xy=(4, 80000), xytext=(4, 150000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

# Sidebar UI
st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2, 10, 3, 1)

# KMeans function
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X_imputed)
    X['Labels'] = kmean.labels_

    # Cluster Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X['nama_kabupaten_kota'], y=X['jumlah_pengangguran'], hue=X['Labels'],
                    size=X['Labels'], palette=sns.color_palette('hls', n_clust), markers=True)

    for label in X['Labels']:
        plt.annotate(label,
            (X[X['Labels'] == label]['nama_kabupaten_kota'].mean(),
            X[X['Labels'] == label]['jumlah_pengangguran'].mean()),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            weight='bold',
            color='black')

    st.header('Cluster Plot')
    st.pyplot()
    st.write(X)

    # Display dataset
    st.subheader('Dataset')
    st.dataframe(df)

    # Display cluster information
    st.subheader('Cluster Information')
    st.write("Number of clusters:", n_clust)
    st.write("Cluster centers:")
    st.write(kmean.cluster_centers_)

# Call the KMeans function
k_means(clust)
