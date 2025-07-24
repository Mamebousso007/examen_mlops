"""
Configuration du système de logging pour l'API House Price Prediction
"""

import logging
import logging.config
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

class StructuredLogger:
    """
    Logger structuré pour les prédictions et audits
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def log_prediction(self, 
                      timestamp: str,
                      input_features: Dict[str, Any],
                      prediction: float,
                      duration: float,
                      model_version: str = "1.0.0",
                      request_id: Optional[str] = None,
                      client_ip: Optional[str] = None):
        log_data = {
            "event_type": "prediction",
            "timestamp": timestamp,
            "request_id": request_id,
            "client_ip": client_ip,
            "model_version": model_version,
            "input_features": input_features,
            "prediction": prediction,
            "duration_seconds": duration,
            "status": "success"
        }
        
        self.logger.info(
            "Prédiction réalisée",
            extra={
                "structured_data": json.dumps(log_data),
                "prediction": prediction,
                "duration": f"{duration:.3f}s",
                "input_features": str(input_features)
            }
        )
    
    def log_prediction_error(self,
                           timestamp: str,
                           input_features: Dict[str, Any],
                           error: str,
                           duration: float,
                           request_id: Optional[str] = None,
                           client_ip: Optional[str] = None):
        log_data = {
            "event_type": "prediction_error",
            "timestamp": timestamp,
            "request_id": request_id,
            "client_ip": client_ip,
            "input_features": input_features,
            "error": error,
            "duration_seconds": duration,
            "status": "error"
        }
        
        self.logger.error(
            f"Erreur de prédiction: {error}",
            extra={
                "structured_data": json.dumps(log_data),
                "error": error,
                "duration": f"{duration:.3f}s",
                "input_features": str(input_features)
            }
        )
    
    def log_api_request(self,
                       timestamp: str,
                       method: str,
                       url: str,
                       status_code: int,
                       duration: float,
                       client_ip: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       request_id: Optional[str] = None):
        log_data = {
            "event_type": "api_request",
            "timestamp": timestamp,
            "request_id": request_id,
            "method": method,
            "url": url,
            "status_code": status_code,
            "duration_seconds": duration,
            "client_ip": client_ip,
            "user_agent": user_agent
        }
        
        self.logger.info(
            f"{method} {url} - {status_code}",
            extra={
                "structured_data": json.dumps(log_data),
                "method": method,
                "url": url,
                "status_code": status_code,
                "duration": f"{duration:.3f}s",
                "client_ip": client_ip
            }
        )

class CustomJsonFormatter(logging.Formatter):
    """
    Formateur JSON personnalisé pour les logs structurés
    """
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'structured_data'):
            try:
                structured = json.loads(record.structured_data)
                log_obj.update(structured)
            except (json.JSONDecodeError, TypeError):
                log_obj['structured_data_error'] = 'Failed to parse structured data'
        
        custom_attrs = ['prediction', 'duration', 'input_features', 'method', 
                        'url', 'status_code', 'client_ip', 'request_type', 
                        'error_category', 'performance_flag']
        
        for attr in custom_attrs:
            if hasattr(record, attr):
                log_obj[attr] = getattr(record, attr)
        
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj, ensure_ascii=False)

def setup_logging(config_path: Optional[str] = None, log_level: str = "INFO") -> None:
    """
    Configure le système de logging
    """
    if config_path is None:
        config_path = Path(__file__).parent / "logging_config.yaml"
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            for handler_name, handler_config in config.get('handlers', {}).items():
                if 'filename' in handler_config:
                    filename = handler_config['filename']
                    if not os.path.isabs(filename):
                        handler_config['filename'] = str(logs_dir / Path(filename).name)
            
            logging.config.dictConfig(config)
        else:
            setup_default_logging(log_level, logs_dir)

        # Forcer le formateur JSON sur les handlers existants
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.setFormatter(CustomJsonFormatter())
        
        logger = logging.getLogger(__name__)
        logger.info("✅ Système de logging initialisé avec succès")

    except Exception as e:
        print(f"Erreur lors de la configuration du logging : {e}")
        setup_default_logging(log_level, logs_dir)

def setup_default_logging(log_level: str = "INFO", logs_dir: Path = Path("logs")) -> None:
    """
    Configuration par défaut si YAML manquant
    """
    log_path = logs_dir / "app.log"
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
