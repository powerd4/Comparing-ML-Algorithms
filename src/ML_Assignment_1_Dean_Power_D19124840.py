import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import get_scorer
from sklearn.dummy import DummyClassifier

# =============================================================================
# Load the datasets
# =============================================================================

# import dataset
adult_1 = pd.read_csv("data/ADULT/adult.data", sep=",", header=None)
adult_2 = pd.read_csv("data/ADULT/adult.test", sep=",", header=None, skiprows=1)
adult = pd.concat([adult_1, adult_2], axis=0, ignore_index=True)
adult.columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    ">50K",
]
adult["y"] = np.where(adult[">50K"].isin([" >50K", " >50k."]), 1, 0)
adult = adult.drop(columns=">50K")
adult.name = "ADULT"

# import COV_TYPE dataset
cov_type = pd.read_csv("data/COV_TYPE/covtype.data.gz", sep=",", header=None)
wilderness_area_names = ["Wilderness_Area_" + str(i) for i in range(1, 5)]
soil_type_names = ["Soil_Type_" + str(i) for i in range(1, 41)]
cov_type.columns = (
    [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    + wilderness_area_names
    + soil_type_names
    + ["Cover_Type"]
)

# 1 corresponds to Lodgepole Pine, 0 to non-Lodgepole Pine
cov_type["y"] = np.where(
    cov_type["Cover_Type"] == cov_type.Cover_Type.value_counts().idxmax(), 1, 0
)
cov_type = cov_type.drop(columns="Cover_Type")
cov_type.name = "COV_TYPE"

# import LETTER dataset
# The paper imports this as two datsets:
# letter_1 treats ”O” as positive and the remaining 25 letters
# as negative, yielding a very unbalanced problem.
# letter_2 uses letters A-M as positives and the rest as negatives,
# yielding a well balanced problem.
letter_1 = pd.read_csv("data/LETTER/letter-recognition.data", sep=",", header=None)
letter_1.columns = [
    "lettr",
    "x-box",
    "y-box",
    "width",
    "high",
    "onpix",
    "x-bar",
    "y-bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x-ege",
    "xegvy",
    "y-ege",
    "yegvx",
]
letter_1 = letter_1[
    [
        "x-box",
        "y-box",
        "width",
        "high",
        "onpix",
        "x-bar",
        "y-bar",
        "x2bar",
        "y2bar",
        "xybar",
        "x2ybr",
        "xy2br",
        "x-ege",
        "xegvy",
        "y-ege",
        "yegvx",
        "lettr",
    ]
]
letter_1["y"] = np.where(letter_1["lettr"] == "O", 1, 0)
letter_1 = letter_1.drop(columns="lettr")
letter_1.name = "LETTER (O / not O)"

letter_2 = pd.read_csv("data/LETTER/letter-recognition.data", sep=",", header=None)
letter_2.columns = [
    "lettr",
    "x-box",
    "y-box",
    "width",
    "high",
    "onpix",
    "x-bar",
    "y-bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x-ege",
    "xegvy",
    "y-ege",
    "yegvx",
]
letter_2 = letter_2[
    [
        "x-box",
        "y-box",
        "width",
        "high",
        "onpix",
        "x-bar",
        "y-bar",
        "x2bar",
        "y2bar",
        "xybar",
        "x2ybr",
        "xy2br",
        "x-ege",
        "xegvy",
        "y-ege",
        "yegvx",
        "lettr",
    ]
]
letter_2["y"] = np.where(
    letter_2["lettr"].isin(
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
    ),
    1,
    0,
)
letter_2 = letter_2.drop(columns="lettr")
letter_2.name = "LETTER (A-M / N-Z)"

# basic descriptive statistics
adult_stats = adult.describe(include="all")
cov_type_stats = cov_type.describe(include="all")
letter_1_stats = letter_1.describe(include="all")
letter_2_stats = letter_2.describe(include="all")


# One-Hot Coding the categorical variables in adult dataset
adult_dummy = pd.get_dummies(adult)
adult_dummy.name = "ADULT"

# Replicate the description table in the paper
datasets = [adult, cov_type, letter_1, letter_2]
descrs = pd.DataFrame([])
descrs["PROBLEM"] = [dataset.name for dataset in datasets]
descrs["#ATTR"] = [len(dataset.columns) - 1 for dataset in datasets]
descrs["TRAIN SIZE"] = 5000
descrs["TEST SIZE"] = [len(dataset) - 5000 for dataset in datasets]
descrs["%POZ"] = [
    100 * len(dataset[dataset["y"] == 1]) / len(dataset) for dataset in datasets
]

datasets = [adult_dummy, cov_type, letter_1, letter_2]


# =============================================================================
# Train and test the models
# =============================================================================

classifiers = [
    (
        "SVM",
        LinearSVC(max_iter=10000),
        {"C": np.logspace(-7, 3, num=3 + 7 + 1, base=10)},
    ),
    (
        "LOGREG",
        LogisticRegression(max_iter=10000),
        {"C": np.logspace(-8, 4, 4 + 8 + 1, base=10)},
    ),
    ("KNN", KNeighborsClassifier(), {"n_neighbors": np.arange(1, 1000, 40)}),
]

scoring = ["accuracy", "f1", "roc_auc"]

results = pd.DataFrame([])

for df in datasets:
    sub_results = pd.DataFrame([])

    df_train, df_test = train_test_split(df, train_size=5000, random_state=48)
    df_train_X = df_train.drop("y", axis=1)
    df_test_X = df_test.drop("y", axis=1)
    df_train_Y = df_train["y"]
    df_test_Y = df_test["y"]

    for name, classifier, params in classifiers:
        if name in ('SVM', 'LOGREG', 'KNN'):
            min_max_scaler = MinMaxScaler().fit(df_train_X)
            df_train_X = min_max_scaler.transform(df_train_X)
            df_test_X = min_max_scaler.transform(df_test_X)
        clf = GridSearchCV(
            estimator=classifier,
            param_grid=params,
            scoring=scoring,
            cv=None,
            refit=False,
            n_jobs=-1,
            verbose=2,
        )  # NOTE: THIS WILL PARALLELIZE ON ALL CORES. REMOVE PARAM N_JOBS TO RUN IN SERIES.
        print("df = " + str(df.name) + ",    model = " + str(name))
        search = clf.fit(df_train_X, df_train_Y)
        search = pd.DataFrame.from_dict(search.cv_results_)[
            ["params", "mean_test_accuracy", "mean_test_f1", "mean_test_roc_auc"]
        ]
        search["model"] = name
        search["df"] = df.name
        search.columns = search.columns.str.replace("test_", "cval_")

        # baseline classifier
        dum_class = DummyClassifier(strategy="constant", constant=1, random_state=48)
        dum = cross_validate(dum_class, df_train_X, df_train_Y, cv=5, scoring=scoring)
        dum = pd.DataFrame.from_dict(dum).drop(columns=["fit_time", "score_time"])
        dum["model"] = name
        dum["df"] = df.name
        dum = dum.assign(**dum.mean()).iloc[[0]]
        dum.columns = dum.columns.str.replace("test_", "base_")

        search = pd.merge(search, dum, how="left", on=["model", "df"])
        sub_results = sub_results.append(search, ignore_index=True)

    # normalise metrics per df
    sub_results["best_accuracy"] = sub_results.groupby("df")[
        "mean_cval_accuracy"
    ].transform("max")
    sub_results["best_f1"] = sub_results.groupby("df")["mean_cval_f1"].transform("max")
    sub_results["best_roc_auc"] = sub_results.groupby("df")[
        "mean_cval_roc_auc"
    ].transform("max")

    sub_results["mean_cval_accuracy"] = (
        sub_results["mean_cval_accuracy"] - sub_results["base_accuracy"]
    ) / (sub_results["best_accuracy"] - sub_results["base_accuracy"])
    sub_results["mean_cval_f1"] = (
        sub_results["mean_cval_f1"] - sub_results["base_f1"]
    ) / (sub_results["best_f1"] - sub_results["base_f1"])
    sub_results["mean_cval_roc_auc"] = (
        sub_results["mean_cval_roc_auc"] - sub_results["base_roc_auc"]
    ) / (sub_results["best_roc_auc"] - sub_results["base_roc_auc"])

    # choose hyperparameter with the best mean normalised metric
    sub_results["MEAN"] = sub_results[
        ["mean_cval_accuracy", "mean_cval_f1", "mean_cval_roc_auc"]
    ].mean(axis=1)
    sub_results = sub_results.loc[sub_results.groupby("model")["MEAN"].idxmax()]

    for score in scoring:
        sub_results["test_" + score] = ""

    # fit models with best parameters.
    for name, classifier, params in classifiers:
        min_max_scaler = MinMaxScaler().fit(df_train_X)
        df_train_X = min_max_scaler.transform(df_train_X)
        df_test_X = min_max_scaler.transform(df_test_X)
        params = sub_results[sub_results["model"] == name].iloc[0]["params"]
        classifier.set_params(**params)
        classifier.fit(df_train_X, df_train_Y)
        for score in scoring:
            sub_results.loc[
                sub_results["model"] == name, ["test_" + score]
            ] = get_scorer(score)(classifier, df_test_X, df_test_Y)

    # normalise the test_metrics and find mean of test metrics
    sub_results["test_accuracy"] = (
        sub_results["test_accuracy"] - sub_results["base_accuracy"]
    ) / (sub_results["best_accuracy"] - sub_results["base_accuracy"])
    sub_results["test_f1"] = (sub_results["test_f1"] - sub_results["base_f1"]) / (
        sub_results["best_f1"] - sub_results["base_f1"]
    )
    sub_results["test_roc_auc"] = (
        sub_results["test_roc_auc"] - sub_results["base_roc_auc"]
    ) / (sub_results["best_roc_auc"] - sub_results["base_roc_auc"])
    sub_results["OPT_SEL"] = sub_results[
        ["test_accuracy", "test_f1", "test_roc_auc"]
    ].mean(axis=1)

    sub_results = sub_results[
        [
            "df",
            "model",
            "params",
            "mean_cval_accuracy",
            "mean_cval_f1",
            "mean_cval_roc_auc",
            "test_accuracy",
            "test_f1",
            "test_roc_auc",
            "MEAN",
            "OPT_SEL",
        ]
    ]

    results = results.append(sub_results, ignore_index=True)

# =============================================================================
# Display the results
# =============================================================================

model_norm_scores_by_metric = results.groupby("model").mean()
model_norm_scores_by_dataset = (
    results[["df", "model", "MEAN"]]
    .groupby(["model", "df"])["MEAN"]
    .mean()
    .reset_index()
    .pivot(index="model", columns="df", values="MEAN")
)
model_norm_scores_by_dataset["MEAN"] = model_norm_scores_by_dataset.mean(axis=1)

print(descrs)
print(model_norm_scores_by_metric)
print(model_norm_scores_by_dataset)
