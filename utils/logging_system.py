import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
import json
from functools import wraps
import time

class CustomFormatter(logging.Formatter):
    """Custom log formatter with colors and detailed information"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_colors=True):
        self.use_colors = use_colors
        super().__init__()
    
    def format(self, record):
        # Add timestamp
        record.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add color if enabled
        if self.use_colors and hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        log_format = "%(timestamp)s | %(levelname)s | %(name)s | %(message)s"
        
        if record.exc_info:
            log_format += "\n" + traceback.format_exception(*record.exc_info)
        
        formatter = logging.Formatter(log_format)
        return formatter.format(record)

class LoggerManager:
    """Centralized logging management"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.loggers = {}
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup root logger
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Setup the root logger with file and console handlers"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(CustomFormatter(use_colors=True))
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = os.path.join(self.log_dir, "application.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(CustomFormatter(use_colors=False))
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_log_file = os.path.join(self.log_dir, "errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(CustomFormatter(use_colors=False))
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        return self.loggers[name]
    
    def set_level(self, level: str):
        """Set the logging level"""
        self.log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Update all loggers
        for logger in self.loggers.values():
            logger.setLevel(self.log_level)
        
        # Update root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Update handlers
        for handler in root_logger.handlers:
            handler.setLevel(self.log_level)

class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self, logger_manager: LoggerManager):
        self.logger_manager = logger_manager
        self.logger = logger_manager.get_logger("ErrorHandler")
        self.error_counts = {}
        self.error_history = []
    
    def handle_error(self, error: Exception, context: str = "", 
                    user_message: str = "", log_level: str = "ERROR") -> Dict[str, Any]:
        """
        Handle an error with comprehensive logging and user feedback
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            user_message: User-friendly error message
            log_level: Logging level (ERROR, WARNING, etc.)
            
        Returns:
            Dictionary containing error information
        """
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'user_message': user_message or str(error),
            'traceback': traceback.format_exc()
        }
        
        # Log the error
        log_level_func = getattr(self.logger, log_level.lower(), self.logger.error)
        log_level_func(f"Error in {context}: {error}")
        
        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to error history
        self.error_history.append(error_info)
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        return error_info
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_counts.clear()
        self.error_history.clear()

def error_handler_decorator(context: str = "", user_message: str = "", 
                          log_level: str = "ERROR", reraise: bool = True):
    """
    Decorator for automatic error handling
    
    Args:
        context: Context description for the function
        user_message: User-friendly error message
        log_level: Logging level
        reraise: Whether to reraise the exception
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get logger manager from global scope or create new one
                logger_manager = getattr(error_handler_decorator, 'logger_manager', None)
                if logger_manager is None:
                    logger_manager = LoggerManager()
                    error_handler_decorator.logger_manager = logger_manager
                
                error_handler = ErrorHandler(logger_manager)
                error_info = error_handler.handle_error(
                    e, context or func.__name__, user_message, log_level
                )
                
                if reraise:
                    raise
                else:
                    return error_info
        return wrapper
    return decorator

class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self, logger_manager: LoggerManager):
        self.logger_manager = logger_manager
        self.logger = logger_manager.get_logger("PerformanceMonitor")
        self.performance_data = {}
        self.start_times = {}
    
    def start_timer(self, operation_name: str):
        """Start timing an operation"""
        self.start_times[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return duration"""
        if operation_name not in self.start_times:
            self.logger.warning(f"Timer for {operation_name} was not started")
            return 0.0
        
        duration = time.time() - self.start_times[operation_name]
        del self.start_times[operation_name]
        
        # Store performance data
        if operation_name not in self.performance_data:
            self.performance_data[operation_name] = []
        
        self.performance_data[operation_name].append({
            'timestamp': datetime.now().isoformat(),
            'duration': duration
        })
        
        # Keep only last 100 measurements
        if len(self.performance_data[operation_name]) > 100:
            self.performance_data[operation_name] = self.performance_data[operation_name][-100:]
        
        self.logger.info(f"Operation {operation_name} completed in {duration:.3f} seconds")
        return duration
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for operation, data in self.performance_data.items():
            if data:
                durations = [d['duration'] for d in data]
                stats[operation] = {
                    'count': len(durations),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations)
                }
        
        return stats
    
    def clear_performance_data(self):
        """Clear performance data"""
        self.performance_data.clear()
        self.start_times.clear()

def performance_monitor_decorator(operation_name: str = ""):
    """
    Decorator for automatic performance monitoring
    
    Args:
        operation_name: Name of the operation being monitored
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get performance monitor from global scope or create new one
            monitor = getattr(performance_monitor_decorator, 'monitor', None)
            if monitor is None:
                logger_manager = LoggerManager()
                monitor = PerformanceMonitor(logger_manager)
                performance_monitor_decorator.monitor = monitor
            
            op_name = operation_name or func.__name__
            monitor.start_timer(op_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.end_timer(op_name)
        
        return wrapper
    return decorator

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, logger_manager: LoggerManager):
        self.logger_manager = logger_manager
        self.logger = logger_manager.get_logger("SystemMonitor")
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: int = 60):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        self.logger.info("System monitoring stopped")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'process_count': len(psutil.pids()),
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            self.logger.warning("psutil not available for system monitoring")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {}

class LoggingConfig:
    """Configuration for logging system"""
    
    def __init__(self):
        self.config = {
            'log_level': 'INFO',
            'log_dir': 'logs',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'enable_console': True,
            'enable_file': True,
            'enable_error_file': True,
            'enable_performance_monitoring': True,
            'enable_system_monitoring': False
        }
    
    def load_from_file(self, config_path: str):
        """Load configuration from file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
        except Exception as e:
            print(f"Failed to load logging config: {e}")
    
    def save_to_file(self, config_path: str):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Failed to save logging config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value

# Global instances
_logger_manager = None
_error_handler = None
_performance_monitor = None
_system_monitor = None

def initialize_logging(config_path: str = "config/logging.json"):
    """Initialize the logging system"""
    global _logger_manager, _error_handler, _performance_monitor, _system_monitor
    
    # Load configuration
    config = LoggingConfig()
    config.load_from_file(config_path)
    
    # Initialize logger manager
    _logger_manager = LoggerManager(
        log_dir=config.get('log_dir', 'logs'),
        log_level=config.get('log_level', 'INFO')
    )
    
    # Initialize error handler
    _error_handler = ErrorHandler(_logger_manager)
    
    # Initialize performance monitor
    if config.get('enable_performance_monitoring', True):
        _performance_monitor = PerformanceMonitor(_logger_manager)
    
    # Initialize system monitor
    if config.get('enable_system_monitoring', False):
        _system_monitor = SystemMonitor(_logger_manager)
        _system_monitor.start_monitoring()
    
    return _logger_manager

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = initialize_logging()
    return _logger_manager.get_logger(name)

def get_error_handler() -> ErrorHandler:
    """Get the error handler instance"""
    global _error_handler
    if _error_handler is None:
        initialize_logging()
    return _error_handler

def get_performance_monitor() -> PerformanceMonitor:
    """Get the performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        initialize_logging()
    return _performance_monitor
