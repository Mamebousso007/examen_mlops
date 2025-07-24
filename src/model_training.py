import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HousePricePredictor:
    """
    Classe principale pour la prédiction des prix immobiliers
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.preprocessor = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """
        Prépare les features pour la modélisation
        """
        logger.info("Préparation des features pour la modélisation")
        
        # Sélection des features numériques et catégorielles
        numeric_features = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated', 'house_age', 'years_since_renovation',
            'living_lot_ratio', 'sqft_per_bedroom', 'sqft_per_bathroom',
            'basement_ratio', 'is_renovated', 'has_basement'
        ]
        
        categorical_features = ['city', 'state', 'condition_label']
        
        # Vérifier quelles features existent dans le dataset
        available_numeric = [f for f in numeric_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]
        
        logger.info(f"Features numériques disponibles: {available_numeric}")
        logger.info(f"Features catégorielles disponibles: {available_categorical}")
        
        # Créer le preprocesseur
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, available_numeric),
                ('cat', categorical_transformer, available_categorical)
            ])
        
        return available_numeric, available_categorical
    
    def split_data(self, df, target_column='price', test_size=0.2, random_state=42):
        """
        Divise les données en ensembles d'entraînement et de test
        """
        logger.info("Division des données en train/test")
        
        # Préparer X et y
        X = df.drop([target_column, 'date'], axis=1, errors='ignore')
        y = df[target_column]
        
        # Supprimer les colonnes non nécessaires
        columns_to_drop = ['street', 'statezip', 'country', 'zipcode']
        X = X.drop(columns_to_drop, axis=1, errors='ignore')
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Taille train: {X_train.shape}, Taille test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def define_models(self):
        """
        Définit les modèles à tester
        """
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        logger.info(f"Modèles définis: {list(self.models.keys())}")
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """
        Évalue un modèle et retourne les métriques
        """
        # Entraînement
        model.fit(X_train, y_train)
        joblib.dump(model, "models/house_price_model.pkl")
        print("✅ Modèle sauvegardé dans models/house_price_model.pkl")
        
        # Prédictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Métriques
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        return metrics
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Entraîne et évalue tous les modèles
        """
        logger.info("Entraînement et évaluation des modèles")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Entraînement du modèle: {name}")
            
            # Créer un pipeline avec préprocessing
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            
            try:
                # Évaluation
                metrics = self.evaluate_model(pipeline, X_train, X_test, y_train, y_test)
                results[name] = {
                    'model': pipeline,
                    'metrics': metrics
                }
                
                logger.info(f"{name} - Test RMSE: {metrics['test_rmse']:.2f}, Test R²: {metrics['test_r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Erreur avec le modèle {name}: {str(e)}")
                continue
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='RandomForest'):
        """
        Optimisation des hyperparamètres pour le meilleur modèle
        """
        logger.info(f"Optimisation des hyperparamètres pour {model_name}")
        
        if model_name == 'RandomForest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [10, 20, None],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'GradientBoosting':
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7],
                'regressor__subsample': [0.8, 0.9, 1.0]
            }
        else:
            logger.warning(f"Pas de grille de paramètres définie pour {model_name}")
            return None
        
        # Pipeline avec préprocessing
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', model)
        ])
        
        # Grid Search
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Meilleurs paramètres: {grid_search.best_params_}")
        logger.info(f"Meilleur score CV: {-grid_search.best_score_:.2f}")
        
        return grid_search.best_estimator_
    
    def feature_importance_analysis(self, model, X_train):
        """
        Analyse de l'importance des features
        """
        logger.info("Analyse de l'importance des features")
        
        try:
            # Obtenir les noms des features après preprocessing
            model.fit(X_train, np.zeros(len(X_train)))  # Fit dummy pour obtenir les noms
            
            if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                importances = model.named_steps['regressor'].feature_importances_
                
                # Obtenir les noms des features transformées
                feature_names = []
                
                # Features numériques
                if hasattr(self.preprocessor, 'transformers_'):
                    num_features = self.preprocessor.transformers_[0][2]
                    feature_names.extend(num_features)
                    
                    # Features catégorielles (encodées)
                    if len(self.preprocessor.transformers_) > 1:
                        cat_transformer = self.preprocessor.transformers_[1][1]
                        if hasattr(cat_transformer, 'get_feature_names_out'):
                            cat_features = self.preprocessor.transformers_[1][2]
                            cat_feature_names = cat_transformer.get_feature_names_out(cat_features)
                            feature_names.extend(cat_feature_names)
                
                # Créer le DataFrame d'importance
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Visualisation
                plt.figure(figsize=(12, 8))
                sns.barplot(data=importance_df.head(15), y='feature', x='importance')
                plt.title('Top 15 des features les plus importantes')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.show()
                
                return importance_df
            
        except Exception as e:
            logger.error(f"Erreur dans l'analyse d'importance: {str(e)}")
            return None
    
    def cross_validation(self, model, X, y, cv=5):
        """
        Validation croisée
        """
        logger.info("Validation croisée")
        
        cv_scores = cross_val_score(
            model, X, y, 
            cv=cv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        cv_rmse_scores = np.sqrt(-cv_scores)
        
        logger.info(f"CV RMSE: {cv_rmse_scores.mean():.2f} (+/- {cv_rmse_scores.std() * 2:.2f})")
        
        return cv_rmse_scores
    
    def save_model(self, model, filepath):
        """
        Sauvegarde le modèle
        """
        joblib.dump(model, filepath)
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath):
        """
        Charge un modèle sauvegardé
        """
        model = joblib.load(filepath)
        logger.info(f"Modèle chargé: {filepath}")
        return model
    
    def generate_model_report(self, results):
        """
        Génère un rapport comparatif des modèles
        """
        logger.info("Génération du rapport de modèles")
        
        report_data = []
        for name, result in results.items():
            metrics = result['metrics']
            report_data.append({
                'Model': name,
                'Test_RMSE': metrics['test_rmse'],
                'Test_R2': metrics['test_r2'],
                'Test_MAE': metrics['test_mae'],
                'Train_RMSE': metrics['train_rmse'],
                'Train_R2': metrics['train_r2'],
                'Overfitting': metrics['train_r2'] - metrics['test_r2']
            })
        
        report_df = pd.DataFrame(report_data).sort_values('Test_R2', ascending=False)
        
        print("\n=== RAPPORT COMPARATIF DES MODÈLES ===")
        print(report_df.round(4))
        
        # Visualisation
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # RMSE Comparaison
        x = np.arange(len(report_df))
        width = 0.35
        axes[0].bar(x - width/2, report_df['Train_RMSE'], width, label='Train', alpha=0.8)
        axes[0].bar(x + width/2, report_df['Test_RMSE'], width, label='Test', alpha=0.8)
        axes[0].set_xlabel('Modèles')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Comparaison RMSE Train vs Test')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(report_df['Model'], rotation=45)
        axes[0].legend()
        
        # R² Comparaison
        axes[1].bar(x - width/2, report_df['Train_R2'], width, label='Train', alpha=0.8)
        axes[1].bar(x + width/2, report_df['Test_R2'], width, label='Test', alpha=0.8)
        axes[1].set_xlabel('Modèles')
        axes[1].set_ylabel('R²')
        axes[1].set_title('Comparaison R² Train vs Test')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(report_df['Model'], rotation=45)
        axes[1].legend()
        
        # Overfitting
        colors = ['red' if x > 0.1 else 'green' for x in report_df['Overfitting']]
        axes[2].bar(report_df['Model'], report_df['Overfitting'], color=colors, alpha=0.7)
        axes[2].set_xlabel('Modèles')
        axes[2].set_ylabel('Overfitting (Train_R² - Test_R²)')
        axes[2].set_title('Détection de Surapprentissage')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Seuil critique')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        return report_df

# Fonction principale d'entraînement
def main_model_training(df_path):
    """
    Pipeline complet d'entraînement des modèles
    """
    print("🤖 DÉMARRAGE DE L'ENTRAÎNEMENT DES MODÈLES 🤖\n")
    
    # Initialisation
    predictor = HousePricePredictor()
    
    # Chargement des données
    logger.info("Chargement des données préprocessées")
    df = pd.read_csv(df_path)
    
    # Préparation des features
    numeric_features, categorical_features = predictor.prepare_features(df)
    
    # Division des données
    X_train, X_test, y_train, y_test = predictor.split_data(df)
    
    # Définition des modèles
    predictor.define_models()
    
    # Entraînement et évaluation
    results = predictor.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Rapport comparatif


if __name__ == "__main__":
    # main_model_training("data/data.csv")
    main_model_training("data/data_preprocessed.csv")
    import joblib
