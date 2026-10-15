import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import DBSCAN


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_data(filepath=None):
    if filepath and os.path.exists(filepath):
        return pd.read_csv(filepath)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, "data", "placement_predict_50k Dataset1.csv"),
        os.path.join(base_dir, "data", "placement_predict_50k_Dataset1.csv"),
        os.path.join(curr_dir, "placement_predict_50k Dataset1.csv"),
        os.path.join(curr_dir, "placement_predict_50k_Dataset1.csv"),
        os.path.join("data", "placement_predict_50k Dataset1.csv"),
        os.path.join("data", "placement_predict_50k_Dataset1.csv"),
        "placement_predict_50k Dataset1.csv",
        "placement_predict_50k_Dataset1.csv",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return pd.read_csv(path)

    raise FileNotFoundError(
        f"Could not find 'placement_predict_50k Dataset1.csv'. Checked paths:\n"
        + "\n".join(candidate_paths)
    )


# =========================================================
# 2. SUMMARY
# =========================================================

def summarize(df):
    print("\n========== DATA SUMMARY ==========")
    print("\n--- Describe ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- First 5 Rows ---")
    print(df.head())


# =========================================================
# 3. STATISTICAL ANALYSIS
# =========================================================

def stat(df):
    print("\n========== STATISTICS ==========")
    num_cols = df.select_dtypes(include=np.number)
    print("\nNumerical Columns:")
    print(num_cols.columns.tolist())
    mean_values = num_cols.mean()
    print("\nMean:")
    print(mean_values)
    std_values = num_cols.std()
    print("\nStandard Deviation:")
    print(std_values)
    skewness = num_cols.skew()
    print("\nSkewness:")
    print(skewness)
    kurtosis = num_cols.kurt()
    print("\nKurtosis:")
    print(kurtosis)
    correlation = num_cols.corr()
    print("\nCorrelation:")
    print(correlation)
    return (    num_cols,    mean_values,    std_values,    skewness,    kurtosis,    correlation)


# =========================================================
# 4. VISUALIZATION
# =========================================================

def visualize(df):
    numeric_df = df.select_dtypes(include=np.number)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # 1. Correlation heatmap'
    sns.heatmap(    numeric_df.corr(),    annot=True,    cmap="YlGnBu",    ax=axes[0, 0])
    axes[0, 0].set_title("Correlation Heatmap")
    # 2. PlacementStatus count
    if "PlacementStatus" in df.columns:
        sns.countplot(        data=df,        x="PlacementStatus",        ax=axes[0, 1]    )
        axes[0, 1].set_title("Placement Status")
    # 3. Salary Package boxplot
    if "Salary Package" in df.columns:
        sns.boxplot(        y=df["Salary Package"],        ax=axes[0, 2]    )
        axes[0, 2].set_title("Salary Package")
    # 4. CGPA distribution
    if "CGPA" in df.columns:    sns.histplot(        df["CGPA"],        kde=True,        ax=axes[1, 0]    )
    axes[1, 0].set_title("CGPA Distribution")
    # 5. CGPA boxplot
    if "CGPA" in df.columns:    sns.boxplot(        y=df["CGPA"],        ax=axes[1, 1]    )
    axes[1, 1].set_title("CGPA Boxplot")
    # 6. CGPA vs Salary Package
    if "CGPA" in df.columns and "Salary Package" in df.columns:    sns.scatterplot(        data=df,        x="CGPA",        y="Salary Package",        ax=axes[1, 2]    )
    axes[1, 2].set_title("CGPA vs Salary Package")
    plt.tight_layout()
    plt.show()


# =========================================================
# 5. NORMALIZATION AND STANDARDIZATION
# =========================================================

def normalize(df, exclude_cols=None):
    print("\n========== SCALING ==========")

    if exclude_cols is None:
        exclude_cols = []

    # Numeric columns that are eligible for scaling (exclude targets)
    numeric_cols_all = df.select_dtypes(include=np.number).columns
    scale_cols = [c for c in numeric_cols_all if c not in exclude_cols]

    numeric_df = df[scale_cols].copy()
    # Everything else — non-numeric columns AND excluded target columns — stays untouched
    keep_df = df.drop(columns=scale_cols).copy()

    print("\nOriginal Data:")
    print(df.head())

    # Min-Max Normalization
    min_max_scaler = MinMaxScaler()
    normalized_array = min_max_scaler.fit_transform(numeric_df)
    normalized_numeric_df = pd.DataFrame(
        normalized_array,
        columns=numeric_df.columns,
        index=numeric_df.index
    )
    normalized_df = pd.concat([keep_df, normalized_numeric_df], axis=1)[df.columns]

    print("\n========== MIN-MAX NORMALIZED DATA (FULL DATASET) ==========")
    print(normalized_df.head())

    # Standardization
    standard_scaler = StandardScaler()
    standardized_array = standard_scaler.fit_transform(numeric_df)
    standardized_numeric_df = pd.DataFrame(
        standardized_array,
        columns=numeric_df.columns,
        index=numeric_df.index
    )
    standardized_df = pd.concat([keep_df, standardized_numeric_df], axis=1)[df.columns]

    print("\n========== STANDARDIZED DATA (FULL DATASET) ==========")
    print(standardized_df.head())

    return normalized_df, standardized_df# =========================================================
# 6. OUTLIER HANDLING
# =========================================================

def outlier(df):
    if "Salary Package" not in df.columns:
        print("Salary Package column not found.")
        return df
    df = df.copy()
    print("\n========== OUTLIER HANDLING ==========")
    print("\nBefore Clipping:")
    print(df["Salary Package"].describe())
    q1 = df["Salary Package"].quantile(0.25)
    q3 = df["Salary Package"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df["Salary Package"] = df["Salary Package"].clip(    upper=upper)
    print("\nAfter Clipping:")
    print(df["Salary Package"].describe())
    return df


# =========================================================
# 7. ONE-HOT ENCODING
# =========================================================

def onehot(df):
    nominal_cols = [    "Gender",    "City",    "Stream",    "Specialisation",    "Hostel",    "HistoryOfBacklogs"]
    # Check whether columns exist
    nominal_cols = [    col for col in nominal_cols    if col in df.columns]
    train_df, test_df = train_test_split(    df,    test_size=0.2,    random_state=42,    stratify=df["PlacementStatus"])
    ohe = OneHotEncoder(    drop="first",    sparse_output=False,    handle_unknown="ignore")
    # Fit ONLY on training data
    train_ohe = ohe.fit_transform(    train_df[nominal_cols])
    # Transform test data
    test_ohe = ohe.transform(    test_df[nominal_cols])
    ohe_cols = ohe.get_feature_names_out(    nominal_cols)
    train_ohe_df = pd.DataFrame(    train_ohe,    columns=ohe_cols,    index=train_df.index)
    test_ohe_df = pd.DataFrame(    test_ohe,    columns=ohe_cols,    index=test_df.index)
    print("\n========== ONE-HOT ENCODED DATA ==========")
    print("\nTraining Data:")
    print(train_ohe_df.head())
    print("\nTesting Data:")
    print(test_ohe_df.head())
    return train_ohe_df, test_ohe_df


# =========================================================
# 8. GRADIENT DESCENT
# =========================================================

def gradient_descent(    X,    y,    learning_rate=0.001,    iterations=1000
):
    # Convert to 1-D NumPy arraysX = np.asarray(X, dtype=float).ravel()y = np.asarray(y, dtype=float).ravel()
    m = len(y)
    # Initial parameters
    theta0 = 0.0
    theta1 = 0.0
    mse_history = []
    for i in range(iterations):
        # Prediction
        y_pred = theta0 + theta1 * X
        # Error
        error = y_pred - y
        # MSE
        mse = np.mean(error ** 2)
        mse_history.append(mse)
        # Gradients
        d_theta0 = (2 / m) * np.sum(error)
        d_theta1 = (2 / m) * np.sum(        error * X    )
        # Update parameters
        theta0 = theta0 - learning_rate * d_theta0
        theta1 = theta1 - learning_rate * d_theta1
    return theta0, theta1, mse_history


# =========================================================
# 9. SIMPLE LINEAR REGRESSION USING GRADIENT DESCENT
# =========================================================

def simple_linear_regression_GD(df):
    print("\n========== LINEAR REGRESSION USING GRADIENT DESCENT ==========")
    # Check required columnsif "CGPA" not in df.columns:    raise ValueError("CGPA column not found.")
    if "Salary Package" not in df.columns:    raise ValueError("Salary Package column not found.")
    X = df[["CGPA"]]
    y = df["Salary Package"]
    # Train-test split
    train_X, test_X, train_y, test_y = train_test_split(    X,    y,    test_size=0.2,    random_state=42)
    # Gradient Descent
    theta0, theta1, mse_history = gradient_descent(    train_X,    train_y,    learning_rate=0.001,    iterations=1000)
    print("\nTheta0 (Intercept):", theta0)
    print("Theta1 (Slope):", theta1)
    # Prediction
    test_X_array = test_X["CGPA"].values
    y_pred = theta0 + theta1 * test_X_array
    # Evaluation
    mse1 = mean_squared_error(   test_y,    y_pred)
    rsq1 = r2_score(    test_y,    y_pred)
    print("\nMSE:", mse1)
    print("R-Square Score:", rsq1)
    # Plot MSE vs Iterations
    plt.figure(figsize=(8, 5))
    plt.plot(    mse_history)
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title(    "Gradient Descent: MSE vs Iterations")
    plt.grid(True)
    plt.show()
    # Regression Line
    plt.figure(figsize=(8, 5))
    plt.scatter(    test_X["CGPA"],    test_y,    alpha=0.5,    label="Actual")
    sorted_indices = np.argsort(    test_X["CGPA"].values)
    plt.plot(    test_X["CGPA"].values[sorted_indices],    y_pred[sorted_indices],    linewidth=2,    label="Regression Line")
    plt.xlabel("CGPA")
    plt.ylabel("Salary Package")
    plt.title(    "Simple Linear Regression: CGPA vs Salary Package")
    plt.legend()
    plt.grid(True)
    plt.show()
    return theta0, theta1, mse1, rsq1


# =========================================================
# 10. SIMPLE LINEAR REGRESSION USING SKLEARN
# =========================================================

def simple_linear_regression(df):
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.linear_model import LinearRegression
    print("\n========== SKLEARN LINEAR REGRESSION ==========")
    X = df[["CGPA"]]
    y = df["Salary Package"]


    train_X, test_X, train_y, test_y = train_test_split(    X,    y,    test_size=0.2,    random_state=42)
    model = LinearRegression()
    lasso1 = Lasso(alpha=0.1)
    ridge1=Ridge(alpha=0.1)
    model.fit(    train_X,    train_y)
    lasso1.fit(train_X,    train_y)
    ridge1.fit(train_X,    train_y)
    y_pred = model.predict(    test_X)
    y_pred1 = lasso1.predict(test_X)
    y_pred2=ridge1.predict(test_X)

    mse1 = mean_squared_error(    test_y,    y_pred)
    mse2 = mean_squared_error(test_y, y_pred1)
    mse3 = mean_squared_error(test_y, y_pred2)
    rsq1 = r2_score(    test_y,    y_pred)
    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_[0])
    print("Linear Regression MSE:", mse1)
    print("Lasso Regression MSE:", mse2)
    print("Riedge Regression MSE:", mse3)
    print("R-Square Score:", rsq1)
    return model
def simple_logistic_regression(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X = df[["CGPA"]]
    y = df["PlacementStatus"]
    print("Placement status values::::::")
    print(y)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y,
        test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(
        max_iter=1000, random_state=42)
    model.fit(train_X, train_y)
    theta0 = model.intercept_[0]
    theta1 = model.coef_[0][0]
    print("Theta0 / Intercept:", theta0)
    print("Theta1 / Coefficient:", theta1)
    y_pred = model.predict(test_X)
    y_prob = model.predict_proba(test_X)[:, 1]
    print(y_pred)
    accuracy = accuracy_score(test_y, y_pred)
    print("Accuracy:", accuracy)
    # Plot as a heatmap
    cm = confusion_matrix(test_y, y_pred)
    print("\n========== CONFUSION MATRIX ==========")
    print(cm)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — Logistic Regression")
    plt.show()
    print("\n========== SAMPLE PREDICTIONS ==========")
    for i in range(min(10, len(test_X))):
        print(
            "CGPA:", test_X.iloc[i, 0],
            "| Actual:", test_y.iloc[i],
            "| Probability:", round(y_prob[i], 4),
            "| Prediction:", y_pred[i]        )
    return model
def decision_tree(df):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    X = df[["CGPA","Internships","AptitudeTestScore"]]
    Y = df["PlacementStatus"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    dt=DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return dt
def decision_tree_pruned(df):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report
    X = df[["CGPA","Internships","AptitudeTestScore"]]
    y = df['PlacementStatus']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42    )
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Pruned Decision Tree Accuracy:",
          accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model
def random_forest_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    X = df[['CGPA',"Internships","AptitudeTestScore"]]
    y = df['PlacementStatus']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(
        n_estimators=100, criterion='gini',    max_depth=5,
        random_state=42  )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest Accuracy:",
          accuracy_score(y_test, y_pred))
    return model

def dbscan_model(df):
    X = df[["CGPA", "Internships", "AptitudeTestScore"]].copy()

    # Remove missing values
    X = X.fillna(X.median())

    model = DBSCAN(
        eps=0.5,
        min_samples=5
    )

    labels = model.fit_predict(X)

    print("\n========== DBSCAN ==========")
    print("Clusters:", set(labels))
    print("Number of clusters:",
          len(set(labels)) - (1 if -1 in labels else 0))
    print("Number of outliers:",
          list(labels).count(-1))

    return model

def compare(sample_df,m1,m2,m3):
    X_sample = sample_df[['CGPA',"Internships","AptitudeTestScore"]]
    y_actual = sample_df['PlacementStatus']
    pred_dt = m1.predict(X_sample)
    pred_pruned = m2.predict(X_sample)
    pred_rf = m3.predict(X_sample)
    results = pd.DataFrame({
        'Student': sample_df.index,
        'Actual': y_actual.values,
        'Decision Tree': pred_dt,
        'Pruned Decision Tree': pred_pruned,
        'Random Forest': pred_rf
    })
    print("\nPrediction Comparison for 5 Random Students")
    print("=" * 80)
    print(results.to_string(index=False))
# MAIN PROGRAM
if __name__ == "__main__":
# Load
    df = load_data()
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())
    print(df['PlacementStatus'])
    # Uncomment when required
    # summarize(df)
    # stat(df)
    # visualize(df)
    # Scaling
    X=df.drop(["PlacementStatus"], axis=1)
    target_cols = ["PlacementStatus", "IsAnomaly", "Salary Package"]
    normalized_df, standardized_df = normalize(df, target_cols)
    # Outlier handling
    standardized_df = outlier(standardized_df)
    # One-hot encoding
    #train_ohe_df, test_ohe_df = onehot(standardized_df)
    # Linear Regression using original data
    #simple_linear_regression(df)
    # Linear Regression using standardized data
    #simple_linear_regression_GD(standardized_df)
    #print(standardized_df.head())
    #simple_logistic_regression(standardized_df)

m1 = decision_tree(standardized_df)
m2 = decision_tree_pruned(standardized_df)
m3 = random_forest_model(standardized_df)

m4 = dbscan_model(standardized_df)

sample_df = standardized_df.sample(n=10, random_state=42)
compare(sample_df, m1, m2, m3)