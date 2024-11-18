# ml_module.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import io
import base64

def load_and_preprocess_data(df):
    # Create 'Year', 'Month', and 'Day' columns
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Drop unwanted columns
    data = df.drop(columns=["preciptype", "Date", "snow", "snowdepth","feelslikemax","feelslikemin","feelslike",
                            'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 
                            'solarradiation', 'solarenergy', 'uvindex'], errors='ignore')
    
    # Add lag features
    for lag in range(1, 8):  # Lag up to 7 days
        data[f'Inflow_lag_{lag}'] = data['Inflow'].shift(lag)
    
    # Handle missing values (fill with mean)
    data = data.apply(lambda col: col.fillna(col.mean()), axis=0)
    
    # Normalize the data (excluding 'Inflow')
    columns_to_normalize = data.columns.difference(['Inflow'])
    dataset = data.copy()
    dataset[columns_to_normalize] = dataset[columns_to_normalize].apply(
        lambda col: (col - col.min()) / (col.max() - col.min()), axis=0
    )
    
    return dataset, data

def exploratory_data_analysis(cur_data):
    plots = []
    data = cur_data.copy()
    data = data.drop(columns=["Inflow_lag_1", "Inflow_lag_2", "Inflow_lag_3", "Inflow_lag_4",
                              "Inflow_lag_5", "Inflow_lag_6", "Inflow_lag_7"], errors='ignore')
    cur_data = data.drop(columns=["tempmax", "tempmin", "precipprob", "precipcover",
                                  "Year", "Month", "Day"], errors='ignore')


    # Box plots for numerical features
    # numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(20, 12))
    temp_data = cur_data.drop(columns=["Inflow"])
    for i, column in enumerate(temp_data):  # Show up to 8 boxplots
        plt.subplot(2, 4, i + 1)
        sns.boxplot(y=temp_data[column])
        plt.title(f'Box Plot: {column}')
    plt.tight_layout()

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plot_base64 = base64.b64encode(img_data.getvalue()).decode()
    plt.close()
    plots.append({'title': 'Box Plots for Numerical Features', 'image': plot_base64})

    # Distribution of 'Inflow'
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Inflow'], kde=True)
    plt.title('Distribution of Inflow')
    plt.xlabel('Inflow')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plot_base64 = base64.b64encode(img_data.getvalue()).decode()
    plt.close()
    plots.append({'title': 'Distribution of Inflow', 'image': plot_base64})

    # Scatter plots for 'Inflow' vs other features
    plt.figure(figsize=(20, 12))
    for i, column in enumerate(cur_data.columns.difference(['Inflow'])[:8]):  # Up to 8 scatter plots
        plt.subplot(2, 4, i + 1)
        sns.scatterplot(x=data[column], y=data['Inflow'])
        plt.title(f'Scatter Plot: Inflow vs {column}')
        plt.xlabel(column)
        plt.ylabel('Inflow')
    plt.tight_layout()

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plot_base64 = base64.b64encode(img_data.getvalue()).decode()
    plt.close()
    plots.append({'title': 'Scatter Plots for Inflow vs Features', 'image': plot_base64})

        # Correlation heatmap
    correlation_matrix = cur_data.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plot_base64 = base64.b64encode(img_data.getvalue()).decode()
    plt.close()
    plots.append({'title': 'Correlation Heatmap', 'image': plot_base64})

    return plots


def pearson_correlation_analysis(cur_data):
    plots = []
    data = cur_data.copy()
    data = data.drop(columns=["Inflow_lag_1", "Inflow_lag_2", "Inflow_lag_3", "Inflow_lag_4",
                              "Inflow_lag_5","Inflow_lag_6","Inflow_lag_7"], errors='ignore')
    data = data.drop(columns=["tempmax","tempmin","precipprob","precipcover",
                              "Year","Month","Day"], errors='ignore')
    # Calculate Pearson correlation coefficients with 'Inflow'
    correlations = data.corr()['Inflow'].drop('Inflow')
    
    # Bar plot of Pearson correlation coefficients
    plt.figure(figsize=(10, 8))
    correlations.sort_values().plot(kind='barh')
    plt.title('Pearson Correlation Coefficient with Inflow')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plot_base64 = base64.b64encode(img_data.getvalue()).decode()
    plt.close()
    plots.append({'title': 'Pearson Correlation with Inflow', 'image': plot_base64})

    return plots

def compare_models(models_results):
    # models_results is a list of tuples: (model_name, mae, r2)
    model_names = [result[0] for result in models_results]
    maes = [result[1] for result in models_results]
    r2s = [result[2] for result in models_results]
    
    plots = []

    # Plot MAE comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, maes, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('MAE')
    plt.title('Model MAE Comparison')
    
    # Add text on top of each bar for MAE values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(yval, 2), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plot_base64_mae = base64.b64encode(img_data.getvalue()).decode()
    plt.close()
    plots.append({'title': 'Model MAE Comparison', 'image': plot_base64_mae})
    
    # Plot R^2 comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, r2s, color='salmon')
    plt.xlabel('Models')
    plt.ylabel('R^2 Score')
    plt.title('Model R^2 Score Comparison')
    
    # Add text on top of each bar for R² values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(yval, 2), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plot_base64_r2 = base64.b64encode(img_data.getvalue()).decode()
    plt.close()
    plots.append({'title': 'Model R^2 Score Comparison', 'image': plot_base64_r2})

    return plots

def run_analysis(train_data_df, test_data_df):
    results = {}
    plots = []

    # Load and preprocess data
    dataset, unnormalized_data = load_and_preprocess_data(train_data_df)

    # Perform EDA
    eda_plots = exploratory_data_analysis(unnormalized_data)
    plots.extend(eda_plots)

    # Pearson Correlation Analysis
    pca_plots = pearson_correlation_analysis(unnormalized_data)
    plots.extend(pca_plots)

    # Prepare data for modeling
    X = dataset.drop('Inflow', axis=1)
    y = dataset['Inflow']

    # Split the data into training and testing sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    # Prepare for k-fold cross-validation
    n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store the results
    models_results = []

    # Define the models
    def create_mlp():
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_full.shape[1],)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Custom wrapper for Keras model
    class KerasRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, build_fn=None, **sk_params):
            self.build_fn = build_fn
            self.sk_params = sk_params
            self.model = None

        def fit(self, X, y):
            self.model = self.build_fn()
            self.model.fit(X, y, **self.sk_params)
            return self

        def predict(self, X):
            return self.model.predict(X).flatten()

    mlp = KerasRegressor(build_fn=create_mlp, epochs=100, batch_size=32, verbose=0)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    models = [('MLP', mlp), ('Random Forest', rf), ('XGBoost', xgb_model)]

    # Perform k-fold cross-validation for each model
    for name, model in models:
        maes = []
        r2s = []
        for train_index, val_index in kf.split(X_train_full_scaled):
            X_train, X_val = X_train_full_scaled[train_index], X_train_full_scaled[val_index]
            y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            maes.append(mae)
            r2s.append(r2)
        avg_mae = np.mean(maes)
        avg_r2 = np.mean(r2s)
        models_results.append((name, avg_mae, avg_r2))

    # Ensemble model
    maes = []
    r2s = []
    for train_index, val_index in kf.split(X_train_full_scaled):
        X_train, X_val = X_train_full_scaled[train_index], X_train_full_scaled[val_index]
        y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
        # Train models on this fold
        mlp_fold = KerasRegressor(build_fn=create_mlp, epochs=100, batch_size=32, verbose=0)
        rf_fold = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb_fold = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        models_fold = [('MLP', mlp_fold), ('Random Forest', rf_fold), ('XGBoost', xgb_fold)]
        predictions = []
        for name_model, model in models_fold:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            predictions.append(y_pred)
        ensemble_pred = np.mean(predictions, axis=0)
        mae = mean_absolute_error(y_val, ensemble_pred)
        r2 = r2_score(y_val, ensemble_pred)
        maes.append(mae)
        r2s.append(r2)
    avg_mae = np.mean(maes)
    avg_r2 = np.mean(r2s)
    models_results.append(('Ensemble Model', avg_mae, avg_r2))

    # Compare models
    # comparison_plots = compare_models(models_results)
    # plots.extend(comparison_plots)

    # Train final models on full training data and evaluate on test set
    final_results = []
    for name, model in models:
        model.fit(X_train_full_scaled, y_train_full)
        y_pred_test = model.predict(X_test_scaled)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        final_results.append((name, mae_test, r2_test))

    # Ensemble model on test set
    predictions = []
    for name, model in models:
        y_pred = model.predict(X_test_scaled)
        predictions.append(y_pred)
    ensemble_pred_test = np.mean(predictions, axis=0)
    mae_test = mean_absolute_error(y_test, ensemble_pred_test)
    r2_test = r2_score(y_test, ensemble_pred_test)
    final_results.append(('Ensemble Model', mae_test, r2_test))

    # Compare final models on test set
    final_comparison_plots = compare_models(final_results)
    plots.extend(final_comparison_plots)

    # Plot actual vs predicted for Ensemble Model on test set
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, ensemble_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Inflow')
    plt.ylabel('Predicted Inflow')
    plt.title('Actual vs Predicted Inflow - (Test Set)')
    plt.tight_layout()

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plot_base64 = base64.b64encode(img_data.getvalue()).decode()
    plt.close()
    plots.append({'title': 'Actual vs Predicted Inflow - (Test Set)', 'image': plot_base64})

    # Inference on test_data_df
    # Assuming test_data_df is preprocessed similarly
    dataset_test, _ = load_and_preprocess_data(test_data_df)
    X_test_inference = dataset_test.drop('Inflow', axis=1)
    X_test_inference_scaled = scaler.transform(X_test_inference)

    # Evaluating Ensemble Model on test set
    predictions = []
    for name, model in models:
        y_pred = model.predict(X_test_inference_scaled)
        predictions.append(y_pred)

    # Combine predictions into one array (ensemble predictions)
    ensemble_pred_inference = np.mean(predictions, axis=0)

    # Save the predictions to a DataFrame
    predictions_df = pd.DataFrame(ensemble_pred_inference, columns=['Predicted_Inflow'])

    # Encode the predictions to base64
    csv_buffer = io.StringIO()
    predictions_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    predictions_csv_str = csv_buffer.getvalue()
    predictions_csv_base64 = base64.b64encode(predictions_csv_str.encode()).decode()

    # Collect the model evaluation metrics
    metrics = {
        # 'models_results': [{'model': name, 'mae': float(avg_mae), 'r2': float(avg_r2)} for name, avg_mae, avg_r2 in models_results],
        'final_results': [{'model': name, 'mae': float(mae_test), 'r2': float(r2_test)} for name, mae_test, r2_test in final_results]
    }

    results['plots'] = plots
    results['metrics'] = metrics
    results['predictions_csv_base64'] = predictions_csv_base64

    return results
