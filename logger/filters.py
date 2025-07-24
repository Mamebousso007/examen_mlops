"""
Filtres personnalisés pour le logging
"""
import logging
from typing import Dict, Any
import re

class PredictionFilter(logging.Filter):
    """
    Filtre pour les logs de prédiction
    Ajoute des informations contextuelles aux logs de prédiction
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filtre et enrichit les logs de prédiction
        
        Args:
            record: Enregistrement de log
            
        Returns:
            bool: True pour conserver le log, False pour le supprimer
        """
        # Ajouter un marqueur pour les logs de prédiction
        if hasattr(record, 'prediction') or 'prediction' in record.getMessage().lower():
            record.log_type = 'prediction'
            record.component = 'model'
            return True
        
        return True

class APIFilter(logging.Filter):
    """
    Filtre pour les logs d'API
    Enrichit les logs avec des informations sur les requêtes HTTP
    """
    
    def __init__(self, name: str = ""):
        super().__init__(name)
        # Patterns pour identifier les types de requêtes
        self.prediction_patterns = [
            re.compile(r'/predict'),
            re.compile(r'/batch'),
            re.compile(r'prediction', re.IGNORECASE)
        ]
        
        self.health_patterns = [
            re.compile(r'/health'),
            re.compile(r'/status')
        ]
        
        self.model_patterns = [
            re.compile(r'/model'),
            re.compile(r'model.*info', re.IGNORECASE)
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filtre et enrichit les logs d'API
        
        Args:
            record: Enregistrement de log
            
        Returns:
            bool: True pour conserver le log
        """
        message = record.getMessage()
        
        # Catégoriser le type de requête
        if any(pattern.search(message) for pattern in self.prediction_patterns):
            record.request_type = 'prediction'
            record.component = 'api'
            record.priority = 'high'
        elif any(pattern.search(message) for pattern in self.health_patterns):
            record.request_type = 'health'
            record.component = 'api'
            record.priority = 'low'
        elif any(pattern.search(message) for pattern in self.model_patterns):
            record.request_type = 'model_info'
            record.component = 'api'
            record.priority = 'medium'
        else:
            record.request_type = 'other'
            record.component = 'api'
            record.priority = 'medium'
        
        # Ajouter des métadonnées temporelles
        record.timestamp = record.created
        
        return True

class ErrorFilter(logging.Filter):
    """
    Filtre pour les logs d'erreur
    Enrichit les logs d'erreur avec des informations de débogage
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Enrichit les logs d'erreur
        
        Args:
            record: Enregistrement de log
            
        Returns:
            bool: True pour conserver le log
        """
        if record.levelno >= logging.ERROR:
            record.log_type = 'error'
            record.needs_attention = True
            
            # Marquer les erreurs critiques
            if 'model' in record.getMessage().lower() and 'load' in record.getMessage().lower():
                record.error_category = 'model_loading'
                record.severity = 'critical'
            elif 'predict' in record.getMessage().lower():
                record.error_category = 'prediction_error'
                record.severity = 'high'
            elif 'connection' in record.getMessage().lower() or 'timeout' in record.getMessage().lower():
                record.error_category = 'connection_error'
                record.severity = 'medium'
            else:
                record.error_category = 'general_error'
                record.severity = 'medium'
        
        return True

class PerformanceFilter(logging.Filter):
    """
    Filtre pour les logs de performance
    Identifie et marque les opérations lentes
    """
    
    def __init__(self, name: str = "", slow_threshold: float = 1.0):
        super().__init__(name)
        self.slow_threshold = slow_threshold  # seuil en secondes
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Identifie les opérations lentes
        
        Args:
            record: Enregistrement de log
            
        Returns:
            bool: True pour conserver le log
        """
        message = record.getMessage()
        
        # Rechercher des informations de durée dans le message
        import re
        duration_match = re.search(r'(\d+\.\d+)s', message)
        
        if duration_match:
            duration = float(duration_match.group(1))
            record.duration = duration
            
            if duration > self.slow_threshold:
                record.performance_flag = 'slow'
                record.needs_optimization = True
            else:
                record.performance_flag = 'normal'
                record.needs_optimization = False
        
        return True

class SensitiveDataFilter(logging.Filter):
    """
    Filtre pour masquer les données sensibles dans les logs
    """
    
    def __init__(self, name: str = ""):
        super().__init__(name)
        # Patterns pour identifier les données sensibles
        self.sensitive_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'ip': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'phone': re.compile(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'),
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Masque les données sensibles dans les logs
        
        Args:
            record: Enregistrement de log
            
        Returns:
            bool: True pour conserver le log
        """
        message = record.getMessage()
        
        # Masquer les données sensibles
        for data_type, pattern in self.sensitive_patterns.items():
            if data_type == 'email':
                message = pattern.sub('***@***.***', message)
            elif data_type == 'ip':
                message = pattern.sub('***.***.***.***', message)
            elif data_type == 'phone':
                message = pattern.sub('***-***-****', message)
        
        # Remplacer le message dans le record
        record.msg = message
        record.args = ()
        
        return True

class DebugFilter(logging.Filter):
    """
    Filtre pour les logs de débogage
    Active uniquement en mode debug
    """
    
    def __init__(self, name: str = "", debug_mode: bool = False):
        super().__init__(name)
        self.debug_mode = debug_mode
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filtre les logs de débogage selon le mode
        
        Args:
            record: Enregistrement de log
            
        Returns:
            bool: True pour conserver le log
        """
        if record.levelno == logging.DEBUG:
            return self.debug_mode
        
        return True

class RateLimitFilter(logging.Filter):
    """
    Filtre pour limiter le taux de certains logs répétitifs
    """
    
    def __init__(self, name: str = "", max_rate: int = 10, time_window: int = 60):
        super().__init__(name)
        self.max_rate = max_rate  # nombre maximum de logs
        self.time_window = time_window  # fenêtre de temps en secondes
        self.log_counts = {}  # compteur par type de message
        self.last_reset = {}  # dernière réinitialisation par type
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Limite le taux de logs répétitifs
        
        Args:
            record: Enregistrement de log
            
        Returns:
            bool: True pour conserver le log
        """
        import time
        
        # Créer une clé basée sur le message (premières 50 chars)
        message_key = record.getMessage()[:50]
        current_time = time.time()
        
        # Réinitialiser le compteur si la fenêtre de temps est écoulée
        if (message_key not in self.last_reset or 
            current_time - self.last_reset[message_key] > self.time_window):
            self.log_counts[message_key] = 0
            self.last_reset[message_key] = current_time
        
        # Incrémenter le compteur
        self.log_counts[message_key] += 1
        
        # Vérifier si on dépasse la limite
        if self.log_counts[message_key] <= self.max_rate:
            return True
        elif self.log_counts[message_key] == self.max_rate + 1:
            # Ajouter un message d'avertissement pour indiquer la limitation
            record.msg = f"{record.msg} [Note: Messages similaires supprimés pour éviter le spam]"
            return True
        
        return False