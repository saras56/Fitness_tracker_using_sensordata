import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../data/interim/03_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set", "duration"], axis=1)


X = df_train.drop("label", axis=1)
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)  # stratify makes sure that all the labels are stratified btw both training and testing

# visualize stratify
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", color="lightblue", ax=ax, label="Total"
)
y_train.value_counts().plot(kind="bar", color="dodgerblue", ax=ax, label="Train")
y_test.value_counts().plot(kind="bar", color="royalblue", ax=ax, label="Test")
plt.legend()


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

df_train.columns[10:]

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
frequency_features = [f for f in df_train.columns if ("_freq" in f) or ("pse" in f)]
cluster_features = ["cluster"]

print("Basic_features:", len(basic_features))
print("square_features:", len(square_features))
print("pca_features:", len(pca_features))
print("time_features:", len(time_features))
print("frequency_features:", len(frequency_features))
print("cluster_features:", len(cluster_features))

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))
# we convert to set to avoid duplicate columns that could occur by mistake
# also we split the feature subsets so as to see if feature engineering is
# beneficial or not

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()

max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

selected_features = [
    "pca_1",
    "gyr_r_freq_0.0_Hz_ws_14",
    "acc_x_freq_0.0_Hz_ws_14",
    "acc_z_freq_0.0_Hz_ws_14",
    "acc_r_freq_0.357_Hz_ws_14",
    "acc_r",
    "gyr_r_freq_2.143_Hz_ws_14",
    "cluster",
    "gyr_y_max_freq",
    "acc_z_freq_1.786_Hz_ws_14",
]

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("No of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()
# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "feature_set_1",
    "feature_set_2",
    "feature_set_3",
    "feature_set_4",
    "selected_features",
]

iterations = 1
score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

score_df.sort_values(by="accuracy", ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

# NN and RF gives the best result. In feature sets, feature set 4 gives the best result
# It can be clearly seen that frequency components add to the increased accuracy

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

# plot confusion matrix
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
# as we have used moving window calculations in the dataset, its possible
# the data has some amount of dependency left between rows(even after removing
# the 50% ). Hence unseen test data is not that difficult to predict for the
# model as the model woukld have seen almost similar data in the train set.
# Hence its important to split the data participant wise

pariticipant_df = df.drop(["set", "category"], axis=1)

X_train = pariticipant_df[pariticipant_df["participant"] != " A"].drop("label", axis=1)
y_train = pariticipant_df[pariticipant_df["participant"] != " A"]["label"]

X_test = pariticipant_df[pariticipant_df["participant"] == " A"].drop("label", axis=1)
y_test = pariticipant_df[pariticipant_df["participant"] == " A"]["label"]

X_train = X_train.drop("participant", axis=1)
X_test = X_test.drop("participant", axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
pariticipant_df["label"].value_counts().plot(
    kind="bar", color="lightblue", ax=ax, label="Total"
)
y_train.value_counts().plot(kind="bar", color="dodgerblue", ax=ax, label="Train")
y_test.value_counts().plot(kind="bar", color="royalblue", ax=ax, label="Test")
plt.legend()
# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

# plot confusion matrix
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# This shows that the model can generalise well as the training data didnt
# contain participant A data. But the test data was of A. This way we made
# sure that the model could accurately predict avctual unseen data

# --------------------------------------------------------------
# Try a more complex model with the selected features
# --------------------------------------------------------------
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=False
)

accuracy = accuracy_score(y_test, class_test_y)

# plot confusion matrix
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
