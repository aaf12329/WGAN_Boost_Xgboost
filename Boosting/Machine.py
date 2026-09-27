# Required Libraries and Environment
# =================================
# Python 3.8.x or higher
# Dependencies:
# - numpy>=1.20.0
# - pandas>=1.3.0
# - tensorflow>=2.6.0
# - scikit-learn>=1.0.0
# - xgboost>=1.5.0
# - shap>=0.40.0
# - scikit-optimize>=0.9.0
# - joblib>=1.1.0
# - matplotlib>=3.5.0 (optional for visualization)
# - seaborn>=0.11.0 (optional for visualization)
#

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import KNNImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from scipy import stats
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import shap
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# DATA PREPROCESSING

def preprocess_data(file_path):
    data_exc = pd.read_excel(file_path)
    data_properties = pd.read_excel(file_path1)

    columns_to_remove = ['Note', 'Reference', ...]  # Define columns to remove
    data_exc = merged_data.drop(columns=columns_to_remove, errors='ignore')
    
    mask = data_exc["Dp"].isna() & data_exc["SBET"].notna() & data_exc["VTotal"].notna()
    data_exc.loc[mask, "Dp"] = (4 * data_exc.loc[mask, "VTotal"] * 1000) / data_exc.loc[mask, "SBET"]
    
    columns_to_impute = ['VTotal', 'Dp', 'C', 'H', 'O', 'N', 'Dosage', 
                         'Temperature', 'Initial pH', 'Initial concentration']
    imputer = KNNImputer(n_neighbors=5)
    data_exc[columns_to_impute] = imputer.fit_transform(data_exc[columns_to_impute])
    
    return data_exc

# WGAN IMPLEMENTATION 

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def build_generator(input_dim, output_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(128, activation="relu")(input_layer)
    x = Dense(256, activation="relu")(x)
    output_layer = Dense(output_dim)(x)
    return Model(input_layer, output_layer)

def build_critic(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(256, activation="relu")(input_layer)
    x = Dense(128, activation="relu")(x)
    output_layer = Dense(1)(x)
    return Model(input_layer, output_layer)

def train_wgan(data_values, epochs=1000, batch_size=64, input_dim=100):
    data_dim = data_values.shape[1]
    
    generator = build_generator(input_dim, data_dim)
    critic = build_critic(data_dim)
    
    optimizer = RMSprop(learning_rate=0.00005)
    
    critic.trainable = True
    critic.compile(loss=wasserstein_loss, optimizer=optimizer)
    generator.compile(optimizer=optimizer, loss=wasserstein_loss)
    
    z = Input(shape=(input_dim,))
    fake_data = generator(z)
    validity = critic(fake_data)
    combined = Model(z, validity)
    combined.compile(loss=wasserstein_loss, optimizer=optimizer)
    
    wasserstein_distances = []
    
    for epoch in range(epochs):
        for _ in range(5):
            idx = np.random.randint(0, data_values.shape[0], batch_size)
            real_data = data_values[idx]
            
            noise = np.random.normal(0, 1, (batch_size, input_dim))
            generated_data = generator.predict(noise)
            
            d_loss_real = critic.train_on_batch(real_data, -np.ones((batch_size, 1)))
            d_loss_fake = critic.train_on_batch(generated_data, np.ones((batch_size, 1)))
            d_loss = d_loss_real - d_loss_fake
            
            wasserstein_distances.append(d_loss)
            
            for layer in critic.layers:
                weights = layer.get_weights()
                weights = [np.clip(w, -0.01, 0.01) for w in weights]
                layer.set_weights(weights)
                
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        g_loss = combined.train_on_batch(noise, -np.ones((batch_size, 1)))
        
        if epoch % 1000 == 0:
            generated_samples = generator.predict(np.random.normal(0, 1, (len(data_values), input_dim)))
            generated_samples = scaler.inverse_transform(generated_samples)
            generated_df = pd.DataFrame(generated_samples, columns=data.columns)
            generated_df.to_csv(f'generated_data_epoch_{epoch}.csv', index=False)
    
    return generator

# 5-FOLD CROSS-VALIDATION

def perform_cross_validation(X, y, model_params=None):
    if model_params is None:
        model_params = {
            'n_estimators': 515,
            'learning_rate': 0.1466,
            'max_depth': 10,
            'min_child_weight': 4,
            'subsample': 0.8591,
            'colsample_bytree': 0.9636,
            'random_state': 42
        }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    all_feature_importances = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler.transform(X_test_fold)
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train_fold_scaled, y_train_fold)
        
        y_train_pred = model.predict(X_train_fold_scaled)
        y_test_pred = model.predict(X_test_fold_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train_fold, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_fold, y_test_pred))
        train_r2 = r2_score(y_train_fold, y_train_pred)
        test_r2 = r2_score(y_test_fold, y_test_pred)
        
        fold_metrics.append({
            'Fold': fold,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'Train_R2': train_r2,
            'Test_R2': test_r2
        })
        
        all_feature_importances.append(model.feature_importances_)
    
    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean().round(4)
    std_metrics = metrics_df.std().round(4)
    
    avg_importance = np.mean(all_feature_importances, axis=0)
    feature_importance = dict(zip(X.columns, avg_importance))
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                key=lambda item: item[1], reverse=True)}
    
    return mean_metrics, sorted_importance

# BAYESIAN HYPERPARAMETER OPTIMIZATION

def bayesian_hyperparameter_optimization(X, y, n_iterations=50):
    search_space = {
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'n_estimators': Integer(50, 800),
        'max_depth': Integer(2, 20),
        'min_child_weight': Integer(1, 6),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'gamma': Real(0, 0.5),
        'reg_alpha': Real(0.0001, 1),
        'reg_lambda': Real(0.0001, 1)
    }
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    bayes_search = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=search_space,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        n_iter=n_iterations,
        random_state=42,
        verbose=0
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    bayes_search.fit(X_scaled, y)
    
    best_model = bayes_search.best_estimator_
    
    return best_model, bayes_search.best_params_

# FEATURE INTERACTION ANALYSIS

def calculate_friedman_h_statistic(model, X, feature_names, n_samples=1000):
    from itertools import combinations
    
    n_features = X.shape[1]
    feature_pairs = list(combinations(range(n_features), 2))
    H_statistics = {}
    feature_medians = np.median(X, axis=0)
    
    for i, j in feature_pairs:
        X_mc = np.zeros((n_samples, n_features))
        
        for col in range(n_features):
            X_mc[:, col] = feature_medians[col]
        
        X_mc[:, i] = np.random.uniform(low=np.min(X[:, i]), high=np.max(X[:, i]), size=n_samples)
        X_mc[:, j] = np.random.uniform(low=np.min(X[:, j]), high=np.max(X[:, j]), size=n_samples)
        
        F_ij = model.predict(X_mc)
        
        X_i = X_mc.copy()
        X_i[:, j] = feature_medians[j]
        F_i = model.predict(X_i)
        
        X_j = X_mc.copy()
        X_j[:, i] = feature_medians[i]
        F_j = model.predict(X_j)
        
        X_0 = X_mc.copy()
        X_0[:, i] = feature_medians[i]
        X_0[:, j] = feature_medians[j]
        F_0 = model.predict(X_0)
        
        interaction_term = F_ij - F_i - F_j + F_0
        
        numerator = np.sum(interaction_term ** 2)
        total_effect = F_ij - F_0
        denominator = np.sum(total_effect ** 2)
        
        if denominator <= 1e-10 or np.isnan(numerator) or np.isnan(denominator):
            H = 0
        else:
            H = numerator / denominator
            H = min(1.0, max(0.0, H))
        
        H_statistics[(i, j)] = H
    
    H_stats_list = []
    for (i, j), value in H_statistics.items():
        H_stats_list.append({
            'Feature1': feature_names[i],
            'Feature2': feature_names[j],
            'H_statistic': value
        })
    
    H_stats_df = pd.DataFrame(H_stats_list)
    H_stats_df = H_stats_df.sort_values(by='H_statistic', ascending=False)
    
    return H_stats_df

# TREE MODEL COMPARISON AND SHAP ANALYSIS

def train_tree_models(X_train, y_train, X_test, y_test):
    # Initialize models
    models = {
        'XGBoost': xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ),
        'MLR': LinearRegression(),
        'SVM': SVR(kernel='rbf', C=10, gamma=0.1),
        'ANN': MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
    }
    
    # Train and evaluate models
    model_results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Store results
        model_results[name] = {
            'model': model,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
        }
        model_filename = f'{name.lower()}_model.joblib'
        joblib.dump(model, model_filename)
        print(f"✅ {name}: R² = {test_r2:.4f}, RMSE = {test_rmse:.2f}")
    
    return model_results

def calculate_shap_values(models, X_train, X_test, feature_names):
    shap_values_dict = {}
    X_sample = X_train[:min(500, len(X_train))]
    
    for name, model_data in models.items():
        if name in ['MLR', 'SVM', 'ANN']:
            print(f"⏭️ 跳过 {name}，不是树模型")
            continue
        model = model_data['model']
        
        # Create explainer
        if name == 'XGBoost':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Store SHAP values
        shap_values_dict[name] = {
            'values': shap_values,
            'expected_value': explainer.expected_value,
            'explainer': explainer
        }
        
        # Get global feature importance from SHAP
        feature_importance = np.abs(shap_values).mean(0)
        shap_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        shap_values_dict[name]['feature_importance'] = shap_importance
        
        # Calculate SHAP interaction values
        if name == 'XGBoost':  # Only do this for one model due to computational cost
            try:
                # This is computationally intensive - use a smaller sample if needed
                smaller_sample = X_sample[:min(200, len(X_sample))]
                shap_interaction_values = explainer.shap_interaction_values(smaller_sample)
                
                # Store interaction values
                shap_values_dict[name]['interaction_values'] = shap_interaction_values
                
                # Process interaction values
                feature_interactions = []
                for i in range(len(feature_names)):
                    for j in range(i+1, len(feature_names)):
                        # Get mean absolute interaction value
                        interaction_strength = np.abs(shap_interaction_values[:, i, j]).mean()
                        feature_interactions.append({
                            'Feature1': feature_names[i],
                            'Feature2': feature_names[j],
                            'Interaction_Strength': interaction_strength
                        })
                
                # Create interaction DataFrame
                interaction_df = pd.DataFrame(feature_interactions)
                interaction_df = interaction_df.sort_values('Interaction_Strength', ascending=False)
                shap_values_dict[name]['feature_interactions'] = interaction_df
            except Exception as e:
                # If SHAP interaction values fail (they can be memory intensive)
                shap_values_dict[name]['interaction_values'] = None
                shap_values_dict[name]['feature_interactions'] = None
    
    return shap_values_dict

# MAIN EXECUTION FUNCTIONS 

def prepare_dataset(file_path, target_variable='Adsorption amount'):
    """
    Prepare dataset for modeling with configurable target variable
    
    Parameters:
    file_path (str): Path to the dataset
    target_variable (str): Target variable to predict, either 'Adsorption amount' or 'Adsorption capacity'
    
    Returns:
    X, y, validation_set: Features, target, and validation dataset
    """
    df = pd.read_csv(file_path)
    df["Number"] = range(1, len(df) + 1)
    # Ensure target variable exists in the dataset
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset. "
                         f"Available options are: {', '.join([col for col in df.columns if 'Adsorption' in col])}")
    
    random_numbers = np.random.choice(df["Number"].dropna().unique(), size=3)
    validation_set = df[df["Number"].isin(random_numbers)]
    drop_df = df[~df["Number"].isin(random_numbers)]
    
    encoded_df = pd.get_dummies(drop_df, columns=['Pollutant'])
    exclude_columns = ['Number', 'Reference', 'Adsorption amount', 'Adsorption capacity', 'pKa3', 'pKa2', 'pKa1']
    X = encoded_df.drop(columns=exclude_columns)
    y = encoded_df[target_variable]
    
    return X, y, validation_set

def run_complete_analysis(file_path, target_variable='Adsorption capacity'):
    # Prepare dataset
    X, y, validation_set = prepare_dataset(file_path, target_variable)
    
    # Split data for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Bayesian optimization for XGBoost
    best_model, best_params = bayesian_hyperparameter_optimization(X, y)
    
    # Cross-validation with best parameters
    cv_metrics, feature_importance = perform_cross_validation(X, y, best_params)
    
    # Train final XGBoost model with best parameters
    final_xgb_model = xgb.XGBRegressor(**best_params)
    final_xgb_model.fit(X_train_scaled, y_train)
    
    # Train multiple tree models for comparison
    tree_models = train_tree_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Calculate Friedman's H-statistics for XGBoost
    H_stats_df = calculate_friedman_h_statistic(final_xgb_model, X_train_scaled, X.columns.tolist())
    
    # Calculate SHAP values for all models
    shap_values = calculate_shap_values(tree_models, X_train_scaled, X_test_scaled, X.columns.tolist())
    
    # Save models
    model_filename = f'xgboost_model_{target_variable.replace(" ", "_").lower()}.joblib'
    scaler_filename = f'scaler_{target_variable.replace(" ", "_").lower()}.joblib'
    joblib.dump(final_xgb_model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    results = {
        'target_variable': target_variable,
        'best_params': best_params,
        'cv_metrics': cv_metrics,
        'feature_importance': feature_importance,
        'H_statistics': H_stats_df,
        'tree_models': {name: {'metrics': data['metrics']} for name, data in tree_models.items()},
        'shap_feature_importance': {name: data['feature_importance'] for name, data in shap_values.items()},
        'shap_interactions': shap_values.get('XGBoost', {}).get('feature_interactions', None)
    }
    
    return results
