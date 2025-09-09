"""
Google Cloud Monitoring and Logging configuration.
"""

import logging
import os
from typing import Dict, Any
from google.cloud import logging as cloud_logging
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import query

# Configure structured logging
def setup_cloud_logging():
    """Set up Google Cloud Logging."""
    try:
        # Initialize Cloud Logging client
        client = cloud_logging.Client()
        client.setup_logging()
        
        # Create structured logger
        logger = logging.getLogger("document-processing-api")
        logger.setLevel(logging.INFO)
        
        # Add custom formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s"}'
        )
        
        # Remove default handlers to avoid duplication
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add console handler for local development
        if os.getenv("ENVIRONMENT") != "production":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    except Exception as e:
        # Fallback to standard logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("document-processing-api")

# Initialize logger
logger = setup_cloud_logging()

class MetricsCollector:
    """Collect and send custom metrics to Cloud Monitoring."""
    
    def __init__(self, project_id: str = None):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{self.project_id}"
    
    def create_metric_descriptor(self, metric_type: str, description: str, unit: str = "1"):
        """Create a custom metric descriptor."""
        try:
            descriptor = monitoring_v3.MetricDescriptor(
                type=f"custom.googleapis.com/{metric_type}",
                metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
                value_type=monitoring_v3.MetricDescriptor.ValueType.INT64,
                description=description,
                unit=unit,
            )
            
            descriptor = self.client.create_metric_descriptor(
                name=self.project_name, descriptor=descriptor
            )
            logger.info(f"Created metric descriptor: {descriptor.name}")
            return descriptor
        except Exception as e:
            logger.error(f"Failed to create metric descriptor: {e}")
            return None
    
    def write_time_series(self, metric_type: str, value: int, labels: Dict[str, str] = None):
        """Write a time series data point."""
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = f"custom.googleapis.com/{metric_type}"
            series.resource.type = "cloud_run_revision"
            series.resource.labels = {
                "service_name": "document-processing-api",
                "location": os.getenv("GOOGLE_CLOUD_REGION", "us-central1"),
            }
            
            if labels:
                series.metric.labels.update(labels)
            
            now = monitoring_v3.TimeInterval()
            now.end_time.seconds = int(time.time())
            now.end_time.nanos = 0
            now.start_time.seconds = now.end_time.seconds - 1
            now.start_time.nanos = 0
            
            point = monitoring_v3.Point()
            point.interval = now
            point.value.int64_value = value
            series.points = [point]
            
            self.client.create_time_series(
                name=self.project_name, time_series=[series]
            )
            
        except Exception as e:
            logger.error(f"Failed to write time series: {e}")

# Global metrics collector
metrics = MetricsCollector()

def log_document_processed(file_name: str, chunks_count: int, processing_time: float):
    """Log document processing metrics."""
    logger.info(f"Document processed: {file_name}, chunks: {chunks_count}, time: {processing_time}s")
    
    # Send custom metrics
    try:
        metrics.write_time_series(
            "documents_processed",
            value=1,
            labels={"file_type": file_name.split('.')[-1]}
        )
        metrics.write_time_series(
            "chunks_created",
            value=chunks_count,
            labels={"file_type": file_name.split('.')[-1]}
        )
        metrics.write_time_series(
            "processing_time_seconds",
            value=int(processing_time),
            labels={"file_type": file_name.split('.')[-1]}
        )
    except Exception as e:
        logger.error(f"Failed to send metrics: {e}")

def log_api_request(endpoint: str, status_code: int, response_time: float):
    """Log API request metrics."""
    logger.info(f"API request: {endpoint}, status: {status_code}, time: {response_time}s")
    
    try:
        metrics.write_time_series(
            "api_requests",
            value=1,
            labels={"endpoint": endpoint, "status_code": str(status_code)}
        )
        metrics.write_time_series(
            "api_response_time",
            value=int(response_time * 1000),  # Convert to milliseconds
            labels={"endpoint": endpoint}
        )
    except Exception as e:
        logger.error(f"Failed to send API metrics: {e}")

def log_error(error_type: str, error_message: str, context: Dict[str, Any] = None):
    """Log error with structured data."""
    error_data = {
        "error_type": error_type,
        "error_message": error_message,
        "context": context or {}
    }
    logger.error(f"Error occurred: {error_data}")
    
    try:
        metrics.write_time_series(
            "errors",
            value=1,
            labels={"error_type": error_type}
        )
    except Exception as e:
        logger.error(f"Failed to send error metrics: {e}")
