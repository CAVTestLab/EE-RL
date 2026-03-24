import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import deque, defaultdict
import json


class PerformanceTracker:
    def __init__(self, max_history: int = 1000):
        """Initialize tracker state with bounded metric history."""
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.timers = {}
        self.start_time = time.time()
        
        # Reentrant lock allows nested metric calls from timer helpers.
        self._lock = threading.RLock()
    
    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            
            self.metrics[name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def increment_counter(self, name: str, delta: int = 1):
        with self._lock:
            self.counters[name] += delta
    
    def start_timer(self, name: str):
        with self._lock:
            self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        with self._lock:
            if name in self.timers:
                duration = time.time() - self.timers[name]
                self.record_metric(f"{name}_duration", duration)
                del self.timers[name]
                return duration
            return 0.0
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = [item['value'] for item in self.metrics[name]]
            
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1] if values else 0,
            }
    
    def get_rate(self, counter_name: str, time_window: float = 60.0) -> float:
        """Estimate per-second event rate using timestamps when available."""
        with self._lock:
            current_time = time.time()
            window_start = current_time - time_window
            
            metric_name = f"{counter_name}_timestamp"
            if metric_name in self.metrics:
                recent_events = [
                    item for item in self.metrics[metric_name] 
                    if item['timestamp'] >= window_start
                ]
                return len(recent_events) / time_window
            else:
                total_time = current_time - self.start_time
                return self.counters[counter_name] / max(total_time, 1.0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Build a snapshot of uptime, counters, metric stats, and rates."""
        with self._lock:
            summary = {
                'uptime': time.time() - self.start_time,
                'counters': dict(self.counters),
                'metrics': {},
                'rates': {},
            }
            
            for name in self.metrics:
                if not name.endswith('_timestamp'):
                    summary['metrics'][name] = self.get_metric_stats(name)
            
            for counter_name in self.counters:
                summary['rates'][f"{counter_name}_per_sec"] = self.get_rate(counter_name)
            
            return summary
    
    def reset(self):
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.timers.clear()
            self.start_time = time.time()


class VLMMonitor:
    """Unified monitor for VLM components and performance signals."""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize component registry, tracker, alerts, and thresholds."""
        self.update_interval = update_interval
        self.performance_tracker = PerformanceTracker()
        
        self.monitored_components = {}
        
        self.is_monitoring = False
        self.monitor_thread = None
        
        self.alert_callbacks = []
        
        self.thresholds = {
            'error_rate': 0.1,
            'processing_time': 5.0,
            'queue_utilization': 0.9,
        }
    
    def record_vlm_processing_time(self, processing_time: float):
        self.performance_tracker.record_metric("vlm_processing_time", processing_time)
        self.performance_tracker.increment_counter("vlm_processing_count")
    
    def record_vlm_correction_rate(self, correction_rate: float):
        self.performance_tracker.record_metric("vlm_correction_rate", correction_rate)
    
    def record_buffer_usage(self, buffer):
        """Capture generic and VLM-specific buffer utilization metrics."""
        try:
            if hasattr(buffer, 'size'):
                buffer_size = buffer.size()
                self.performance_tracker.record_metric("buffer_current_size", buffer_size)
            
            if hasattr(buffer, 'buffer_size'):
                max_size = buffer.buffer_size
                self.performance_tracker.record_metric("buffer_max_size", max_size)
                
                if hasattr(buffer, 'size'):
                    utilization = buffer.size() / max_size if max_size > 0 else 0
                    self.performance_tracker.record_metric("buffer_utilization", utilization)
            
            if hasattr(buffer, 'get_vlm_queue_size'):
                vlm_queue_size = buffer.get_vlm_queue_size()
                self.performance_tracker.record_metric("vlm_queue_size", vlm_queue_size)
                
        except Exception:
            pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        return self.performance_tracker.get_summary()
    
    def reset(self):
        self.performance_tracker.reset()
        if self.is_monitoring:
            self.stop_monitoring()
    
    def add_component(self, name: str, component: Any, getter_fn: Optional[Callable] = None):
        self.monitored_components[name] = {
            'component': component,
            'getter_fn': getter_fn or self._default_getter,
            'last_update': 0,
        }
    
    def _default_getter(self, component: Any) -> Dict[str, Any]:
        if hasattr(component, 'get_status'):
            return component.get_status()
        elif hasattr(component, 'get_info'):
            return component.get_info()
        else:
            return {'status': 'unknown'}
    
    def start_monitoring(self):
        """Start the periodic monitoring worker."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Refresh component metrics and evaluate alerts on a fixed interval."""
        while self.is_monitoring:
            try:
                self._update_metrics()
                self._check_alerts()
                time.sleep(self.update_interval)
            except Exception:
                time.sleep(self.update_interval)
    
    def _update_metrics(self):
        """Pull status from each component and record numeric metrics."""
        current_time = time.time()
        
        for name, info in self.monitored_components.items():
            try:
                component = info['component']
                getter_fn = info['getter_fn']
                
                status = getter_fn(component)
                
                self._record_component_metrics(name, status, current_time)
                
                info['last_update'] = current_time
                
            except Exception:
                self.performance_tracker.increment_counter(f"{name}_errors")
    
    def _record_component_metrics(self, name: str, status: Dict[str, Any], timestamp: float):
        prefix = f"{name}_"
        
        for key, value in status.items():
            if isinstance(value, (int, float)):
                self.performance_tracker.record_metric(f"{prefix}{key}", value, timestamp)
        
        if 'error_count' in status and 'processed_count' in status:
            total = status['processed_count'] + status['error_count']
            if total > 0:
                error_rate = status['error_count'] / total
                self.performance_tracker.record_metric(f"{prefix}error_rate", error_rate, timestamp)
    
    def _check_alerts(self):
        """Trigger alerts when live metrics exceed configured thresholds."""
        for component_name in self.monitored_components:
            error_rate_metric = f"{component_name}_error_rate"
            error_rate_stats = self.performance_tracker.get_metric_stats(error_rate_metric)
            
            if error_rate_stats and error_rate_stats['latest'] > self.thresholds['error_rate']:
                self._trigger_alert(
                    "High error-rate alert",
                    f"Component {component_name} error rate {error_rate_stats['latest']:.2%} exceeds threshold {self.thresholds['error_rate']:.2%}"
                )
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Dispatch alert payload to registered callbacks."""
        alert_info = {
            'type': alert_type,
            'message': message,
            'timestamp': time.time(),
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_info)
            except Exception:
                pass
    
    def add_alert_callback(self, callback: Callable[[Dict], None]):
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, metric_name: str, threshold: float):
        self.thresholds[metric_name] = threshold
    
    def get_system_status(self) -> Dict[str, Any]:
        """Return monitor state, component status, and tracker summary."""
        status = {
            'monitoring': self.is_monitoring,
            'components': {},
            'performance': self.performance_tracker.get_summary(),
            'thresholds': self.thresholds.copy(),
        }
        
        for name, info in self.monitored_components.items():
            try:
                status['components'][name] = {
                    'status': info['getter_fn'](info['component']),
                    'last_update': info['last_update'],
                }
            except Exception as e:
                status['components'][name] = {'error': str(e)}
        
        return status
    
    def export_metrics(self, filepath: str):
        metrics_data = {
            'timestamp': time.time(),
            'system_status': self.get_system_status(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    def __del__(self):
        try:
            self.stop_monitoring()
        except:
            pass
