"""
Classe de gestion du modèle de prédiction des prix immobiliers
"""
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import os
from pathlib import Path

from config.config import settings

logger = logging.getLogger(__name__)

class HousePriceModel:
    """
    Classe pour gérer le modèle de prédiction des prix immobiliers
    """
    
    def __init__(self):
        self.model = None
        self.model_metadata = {}
        self.expected_features = []
        self.model_loaded = False
        
    def load_model(self, model_path: str = None) -> bool:
        """
        Charge le modèle depuis le fichier spécifié
        
        Args:
            model_path: Chemin vers le fichier du modèle
            
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            if model_path is None:
                model_path = settings.MODEL_PATH
            
            # Vérifier que le fichier existe
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle non trouvé à: {model_path}")
            
            # Charger le modèle
            self.model = joblib.load(model_path)
            
            # Extraire les métadonnées du modèle
            self._extract_model_metadata(model_path)
            
            # Définir les features attendues
            self._set_expected_features()
            
            self.model_loaded = True
            
            logger.info(f"✅ Modèle chargé avec succès depuis: {model_path}")
            logger.info(f"Type de modèle: {type(self.model).__name__}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            self.model_loaded = False
            return False
    
    def _extract_model_metadata(self, model_path: str):
        """Extrait les métadonnées du modèle"""
        file_stat = os.stat(model_path)
        
        self.model_metadata = {
            "version": "1.0.0",
            "model_type": type(self.model).__name__,
            "file_size": file_stat.st_size,
            "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "model_path": model_path
        }
    
    def _set_expected_features(self):
        """Définit les features attendues par le modèle"""
        # Features basées sur votre preprocessing
        self.expected_features = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated', 'house_age', 'years_since_renovation',
            'living_lot_ratio', 'sqft_per_bedroom', 'sqft_per_bathroom',
            'basement_ratio', 'is_renovated', 'has_basement', 'city', 'state'
        ]
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Effectue une prédiction sur les données fournies
        
        Args:
            data: DataFrame contenant les features
            
        Returns:
            np.ndarray: Prédictions
        """
        if not self.model_loaded:
            raise RuntimeError("Modèle non chargé")
        
        try:
            # Préprocessing des données
            processed_data = self._preprocess_data(data)
            
            # Prédiction
            predictions = self.model.predict(processed_data)
            
            logger.info(f"Prédiction effectuée pour {len(data)} échantillons")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Préprocesse les données avant prédiction
        
        Args:
            data: Données brutes
            
        Returns:
            pd.DataFrame: Données préprocessées
        """
        processed_data = data.copy()
        
        # Calcul des features dérivées (similaire à votre preprocessing)
        current_year = datetime.now().year
        
        # Age de la maison
        if 'house_age' not in processed_data.columns and 'yr_built' in processed_data.columns:
            processed_data['house_age'] = current_year - processed_data['yr_built']
        
        # Indicateur de rénovation
        if 'is_renovated' not in processed_data.columns and 'yr_renovated' in processed_data.columns:
            processed_data['is_renovated'] = (processed_data['yr_renovated'] > 0).astype(int)
        
        # Age depuis rénovation
        if 'years_since_renovation' not in processed_data.columns:
            processed_data['years_since_renovation'] = np.where(
                processed_data['yr_renovated'] > 0,
                current_year - processed_data['yr_renovated'],
                processed_data.get('house_age', 0)
            )
        
        # Ratios de surface
        if 'living_lot_ratio' not in processed_data.columns:
            processed_data['living_lot_ratio'] = (
                processed_data['sqft_living'] / 
                np.maximum(processed_data['sqft_lot'], 1)
            )
        
        if 'sqft_per_bedroom' not in processed_data.columns:
            processed_data['sqft_per_bedroom'] = (
                processed_data['sqft_living'] / 
                np.maximum(processed_data['bedrooms'], 1)
            )
        
        if 'sqft_per_bathroom' not in processed_data.columns:
            processed_data['sqft_per_bathroom'] = (
                processed_data['sqft_living'] / 
                np.maximum(processed_data['bathrooms'], 1)
            )
        
        # Indicateur de sous-sol
        if 'has_basement' not in processed_data.columns:
            processed_data['has_basement'] = (
                processed_data.get('sqft_basement', 0) > 0
            ).astype(int)
        
        # Ratio sous-sol
        if 'basement_ratio' not in processed_data.columns:
            processed_data['basement_ratio'] = (
                processed_data.get('sqft_basement', 0) / 
                np.maximum(processed_data['sqft_living'], 1)
            )
        
        # Gestion des valeurs manquantes
        processed_data = self._handle_missing_values(processed_data)
        
        # Nettoyage des valeurs infinies
        processed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed_data.fillna(processed_data.median(numeric_only=True), inplace=True)
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gère les valeurs manquantes"""
        # Valeurs par défaut pour les colonnes importantes
        defaults = {
            'waterfront': 0,
            'view': 0,
            'yr_renovated': 0,
            'sqft_basement': 0,
            'city': 'Unknown',
            'state': 'WA'
        }
        
        for col, default_val in defaults.items():
            if col in data.columns:
                data[col].fillna(default_val, inplace=True)
        
        return data
    
    def validate_input(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Valide les données d'entrée
        
        Args:
            data: Données à valider
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            # Vérifications de base
            if data.empty:
                return False, "Données vides"
            
            # Vérifier les colonnes critiques
            critical_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
            missing_critical = [col for col in critical_columns if col not in data.columns]
            
            if missing_critical:
                return False, f"Colonnes critiques manquantes: {missing_critical}"
            
            # Vérifier les valeurs négatives
            for col in critical_columns:
                if (data[col] < 0).any():
                    return False, f"Valeurs négatives détectées dans {col}"
            
            # Vérifier les valeurs aberrantes
            if (data['bedrooms'] > 20).any():
                return False, "Nombre de chambres trop élevé (>20)"
            
            if (data['bathrooms'] > 20).any():
                return False, "Nombre de salles de bain trop élevé (>20)"
            
            if (data['sqft_living'] > 50000).any():
                return False, "Surface habitable trop importante (>50000 sqft)"
            
            return True, "Données valides"
            
        except Exception as e:
            return False, f"Erreur lors de la validation: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle
        
        Returns:
            Dict: Informations du modèle
        """
        if not self.model_loaded:
            return {"error": "Modèle non chargé"}
        
        info = self.model_metadata.copy()
        info.update({
            "features_count": len(self.expected_features),
            "is_loaded": self.model_loaded,
            "expected_features": self.expected_features[:10]  # Limiter pour l'affichage
        })
        
        return info
    
    def get_expected_features(self) -> List[str]:
        """
        Retourne la liste des features attendues
        
        Returns:
            List[str]: Liste des features
        """
        return self.expected_features.copy()
    
    def predict_with_confidence(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédiction avec intervalle de confiance (si supporté par le modèle)
        
        Args:
            data: Données d'entrée
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predictions, intervals)
        """
        predictions = self.predict(data)
        
        # Calcul simple d'intervalle de confiance (à améliorer selon le modèle)
        # Vous pouvez implémenter une méthode plus sophistiquée selon votre modèle
        std_error = predictions.std() if len(predictions) > 1 else predictions[0] * 0.1
        confidence_intervals = np.column_stack([
            predictions - 1.96 * std_error,
            predictions + 1.96 * std_error
        ])
        
        return predictions, confidence_intervals
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retourne l'importance des features (si supporté par le modèle)
        
        Returns:
            Dict[str, float]: Importance des features
        """
        if not self.model_loaded:
            return {}
        
        try:
            # Vérifier si le modèle a des importances de features
            if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
                importances = self.model.named_steps['regressor'].feature_importances_
                
                # Obtenir les noms des features (approximation)
                feature_names = self.expected_features[:len(importances)]
                
                return dict(zip(feature_names, importances))
            else:
                return {"message": "Importance des features non disponible pour ce modèle"}
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'importance: {e}")
            return {"error": str(e)}