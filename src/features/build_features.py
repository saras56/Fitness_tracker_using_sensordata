import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

df.info()

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
# missing values as we removed outliers

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():
    duration = df[df["set"] == s].index[-1] - df[df["set"] == s].index[0]
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

duration_df[0] / 5  # avg duration of a single repetition for heavy sets
duration_df[1] / 10  # avg duration of a single repetition for medium sets

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

# by applying lpf, we are looking at the overall movement pattern and 
# not the small sharp changes in the graph ie the noise from small 
# incremental changes due to hand and position adjustment is filtered out
df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000 / 200
cutoff = 1.3  # chosen by tweaking different values and checking 
# the o/p graph.higher number would allow higher freq to pass through 
# hence less smoothening lower number gives smoothened graphs

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)
# creates a new column with lowpass feature

subset = df_lowpass[df_lowpass["set"] == 6]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(
    loc="upper right", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper right", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)



for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5) #new col added
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# here the original col is overridden by the lowpass col and lowpass col is deleted


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("PCA number")
plt.ylabel("explained variance")
plt.show()

# the optimum component number by elbow method is 3. thats where the
# rate of change in variance diminishes

# Applying PCA based on the PCA number
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)  

#Scree Plot
plt.figure(figsize=(10, 6))
plt.bar(np.arange(1, len(pc_values) + 1), pc_values)
plt.title('Explained Variance Ratio by Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1, len(pc_values) + 1), [f'PCA{i}' for i in range(1, len(pc_values) + 1)])
plt.grid(axis='y')
plt.show() 
# the features have been reduced to 3 while capturing around 95% of variance

subset = df_pca[df_pca["set"] == 65]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# finding out the scalar magnitude as its impartial to device orientation
# and can handle dynamic reorientations
df_squared = df_pca.copy()
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction 
# --------------------------------------------------------------

# for temporal abstraction rolling window average is used
# different statistical properties are extarcted based on the window size
# window size is the no of timepoints from the past
# this whole looping is done set wise as rolling average/std takes past values
# into consideration. This could lead to erroreneos data as data from one label
# and category could mix with the other if not taken setwise
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
ws = int(1000 / 200)

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)  # list type
df_temporal = pd.concat(df_temporal_list)

subset[["acc_x", "acc_x_temp_mean_ws_5", "acc_x_temp_std_ws_5"]].plot()
subset[["gyr_x", "gyr_x_temp_mean_ws_5", "gyr_x_temp_std_ws_5"]].plot()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

# We apply DFT to extract frequency features so as to get more insight from data
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)  # avg length of a repetition

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
df_freq.columns
# visualize results
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.071_Hz_ws_14",
        "acc_y_freq_2.143_Hz_ws_14",
    ]
].plot()
#subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()

df_freq_list = [] 
for s in df_freq["set"].unique():
    print(f"Applying fourier transformation to set {s}:")
    subset = (
        df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    )  # you should reset
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
# since we use rolling wibdow method for deriving few cols, the rows  are highly correlated
# would result in overfitting. Inorder to avoid that, we allow some %of overlap
# and remove the rest of the data.Typically overlap allowance is 50% in literature
# but if the dataset is small you could inc to even 80% overlap

df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]
# now we have a dataset whichn is less prone to overfitting


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

# K Means clustering is implemented here. helps to identify most 
# important features in a dataset.here data is grouped into different
# clusters based on similarity  
df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("sum of squared distances")
plt.show()

# from the elbow method it can be seen that n_clutser = 5
# is a good cluster number

subset = df_cluster[cluster_columns]
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
df_cluster["cluster"] = kmeans.fit_predict(subset)

cluster_labels = kmeans.labels_
silhouette_avg = silhouette_score(subset, cluster_labels)
print("Silhouette Score:", silhouette_avg)
# higher score suggests better separation bw clusters. score ranges
# bw (-1,1). Here the score is .58 which is good

unique, counts = np.unique(cluster_labels, return_counts=True)

# Calculate percentage distribution
percentage_distribution = counts / len(subset) * 100

# Print percentage distribution for each cluster
for cluster, percentage in zip(unique, percentage_distribution):
    print(f"Cluster {cluster}: {percentage:.2f}%")

# plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X_axis")
ax.set_ylabel("Y_axis")
ax.set_zlabel("Z_axis")
plt.legend()
plt.show()

# plot by labels
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for label in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == label]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=label)
ax.set_xlabel("X_axis")
ax.set_ylabel("Y_axis")
ax.set_zlabel("Z_axis")
plt.legend()
plt.show()

# Looking at the plots, its understood that plot by clusters and plot by
# labels have similar clusters.


# ---------------   -----------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
