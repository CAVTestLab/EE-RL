import queue
import sys
import threading
from typing import Any, Dict

import numpy as np
from PySide6.QtCore import QMetaObject, QObject, Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPixmap, QTextCharFormat, QTextCursor, QColor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class _MonitorMainWindow(QMainWindow):
    def __init__(self, owner: "VLMMonitorWindow"):
        super().__init__()
        self._owner = owner

    def closeEvent(self, event):
        self._owner.is_running = False
        super().closeEvent(event)


class VLMMonitorWindow:
    """VLM monitor window implemented with PySide6."""

    def __init__(self, title: str = "VLM Monitor"):
        self.title = title
        self.is_running = False
        self.update_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.window_thread = None

        self.app = None
        self.main_window = None

        self.image_label = None
        self.prompt_text = None
        self.response_text = None
        self.stats_labels: Dict[str, QLabel] = {}
        self.vehicle_status_labels: Dict[str, QLabel] = {}

        self.stats_data = {
            "call_count": 0,
            "error_count": 0,
            "cache_hit_count": 0,
            "avg_response_time": 0,
        }
        self.vehicle_data = {
            "speed": 0.0,
            "throttle": 0.0,
            "steer": 0.0,
            "maneuver": "Unknown",
        }

        self.colors = {
            "bg": "#ffffff",
            "frame_bg": "#f8f9fa",
            "title_fg": "#495057",
            "text_fg": "#6c757d",
            "prompt_bg": "#ffffff",
            "response_bg": "#ffffff",
            "error_fg": "#dc3545",
            "success_fg": "#28a745",
            "cache_fg": "#ffc107",
            "accent": "#007bff",
            "border": "#dee2e6",
        }

        self.base_font_sizes = {
            "title": 16,
            "label": 14,
            "text": 12,
            "stats": 13,
            "image_hint": 13,
        }

    def start(self):
        """Start monitor window in a dedicated GUI thread."""
        if self.is_running:
            return

        self.is_running = True
        self.window_thread = threading.Thread(target=self._run_window, daemon=True)
        self.window_thread.start()

    def stop(self):
        """Stop monitor window."""
        if not self.is_running:
            return

        self.is_running = False
        if self.app is not None:
            QMetaObject.invokeMethod(self.app, "quit", Qt.QueuedConnection)

    def _pick_font_family(self) -> str:
        families = {f.lower(): f for f in QFontDatabase.families()}

        if sys.platform.startswith("linux"):
            candidates = [
                "Noto Sans",
                "Noto Sans CJK SC",
                "Ubuntu",
                "DejaVu Sans",
                "Liberation Sans",
                "Cantarell",
            ]
        elif sys.platform == "darwin":
            candidates = ["SF Pro Text", "Helvetica Neue", "Helvetica", "Arial"]
        else:
            candidates = ["Segoe UI", "Arial", "Calibri", "Verdana", "Tahoma"]

        for candidate in candidates:
            match = families.get(candidate.lower())
            if match:
                return match
        return QApplication.font().family()

    def _make_font(self, family: str, size_key: str) -> QFont:
        return QFont(family, self.base_font_sizes[size_key])

    def _run_window(self):
        self.app = QApplication.instance()
        owns_app = self.app is None
        if owns_app:
            self.app = QApplication([])

        family = self._pick_font_family()
        self.app.setFont(self._make_font(family, "text"))

        self.main_window = _MonitorMainWindow(self)
        self.main_window.setWindowTitle(self.title)
        self.main_window.resize(1300, 850)
        self.main_window.setMinimumSize(1000, 700)

        central = QWidget(self.main_window)
        self.main_window.setCentralWidget(central)

        self._create_ui(central, family)

        timer = QTimer(self.main_window)
        timer.timeout.connect(self._process_update_queue)
        timer.start(100)

        self.main_window.show()

        if owns_app:
            self.app.exec()
        else:
            while self.is_running and self.main_window.isVisible():
                self.app.processEvents()
                threading.Event().wait(0.05)

    def _create_ui(self, parent: QWidget, font_family: str):
        root_layout = QVBoxLayout(parent)
        root_layout.setContentsMargins(15, 15, 15, 15)
        root_layout.setSpacing(10)

        content_row = QHBoxLayout()
        content_row.setSpacing(10)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        image_group = self._make_group("Input Image", font_family)
        image_layout = QVBoxLayout(image_group)
        self.image_label = QLabel("Waiting for image input...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setFrameShape(QFrame.StyledPanel)
        self.image_label.setStyleSheet(
            f"color: {self.colors['text_fg']}; background: {self.colors['frame_bg']};"
        )
        self.image_label.setFont(self._make_font(font_family, "image_hint"))
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_label)

        vehicle_group = self._make_group("Vehicle Status", font_family)
        vehicle_grid = QGridLayout(vehicle_group)
        vehicle_grid.setHorizontalSpacing(8)
        vehicle_grid.setVerticalSpacing(4)

        vehicle_items = [
            ("Speed:", "speed", "km/h", 0, 0),
            ("Throttle:", "throttle", "", 0, 2),
            ("Steer:", "steer", "", 1, 0),
            ("Maneuver:", "maneuver", "", 1, 2),
        ]
        for text, key, unit, row, col in vehicle_items:
            lbl = QLabel(text)
            lbl.setFont(self._make_font(font_family, "stats"))
            lbl.setStyleSheet(f"color: {self.colors['text_fg']};")
            vehicle_grid.addWidget(lbl, row, col)

            value = QLabel("0.0" + (f" {unit}" if unit else ""))
            value.setFont(self._make_font(font_family, "stats"))
            value.setStyleSheet(f"color: {self.colors['accent']};")
            vehicle_grid.addWidget(value, row, col + 1)
            self.vehicle_status_labels[key] = value

        left_col.addWidget(image_group, 4)
        left_col.addWidget(vehicle_group, 0)

        prompt_group = self._make_group("Input Prompt", font_family)
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_text = QTextEdit()
        self.prompt_text.setReadOnly(True)
        self.prompt_text.setFont(self._make_font(font_family, "text"))
        self.prompt_text.setStyleSheet(
            f"background: {self.colors['prompt_bg']}; color: {self.colors['text_fg']};"
        )
        prompt_layout.addWidget(self.prompt_text)

        response_group = self._make_group("VLM Response", font_family)
        response_layout = QVBoxLayout(response_group)
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setFont(self._make_font(font_family, "text"))
        self.response_text.setStyleSheet(
            f"background: {self.colors['response_bg']}; color: {self.colors['text_fg']};"
        )
        response_layout.addWidget(self.response_text)

        right_col.addWidget(prompt_group, 1)
        right_col.addWidget(response_group, 1)

        content_row.addLayout(left_col, 1)
        content_row.addLayout(right_col, 1)

        stats_group = self._make_group("Statistics", font_family)
        stats_grid = QGridLayout(stats_group)
        stats_items = [
            ("API Calls:", "call_count"),
            ("Errors:", "error_count"),
            ("Cache Hits:", "cache_hit_count"),
            ("Avg Response:", "avg_response_time"),
        ]
        for i, (label_text, key) in enumerate(stats_items):
            c = i * 2
            label = QLabel(label_text)
            label.setFont(self._make_font(font_family, "stats"))
            label.setStyleSheet(f"color: {self.colors['text_fg']};")
            stats_grid.addWidget(label, 0, c)

            value = QLabel("0")
            value.setFont(self._make_font(font_family, "stats"))
            value.setStyleSheet(f"color: {self.colors['accent']};")
            stats_grid.addWidget(value, 0, c + 1)
            self.stats_labels[key] = value

        root_layout.addLayout(content_row, 4)
        root_layout.addWidget(stats_group, 1)

    def _make_group(self, title: str, family: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setFont(self._make_font(family, "title"))
        group.setStyleSheet(
            f"""
            QGroupBox {{
                border: 1px solid {self.colors['border']};
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px;
                background: {self.colors['frame_bg']};
                color: {self.colors['title_fg']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
            """
        )
        return group

    def _process_update_queue(self):
        if not self.is_running:
            return
        try:
            while True:
                update_data = self.update_queue.get_nowait()
                self._handle_update(update_data)
        except queue.Empty:
            pass
        except Exception as exc:
            print(f"Update loop error: {exc}")

    def _handle_update(self, update_data: Dict[str, Any]):
        update_type = update_data.get("type")
        if update_type == "image":
            self._update_image(update_data["image"])
        elif update_type == "prompt":
            self._update_prompt(update_data["prompt"])
        elif update_type == "response_chunk":
            self._append_response(update_data["chunk"])
        elif update_type == "response_complete":
            self._finalize_response(update_data.get("final_reward", 0))
        elif update_type == "stats":
            self._update_stats(update_data["stats"])
        elif update_type == "clear":
            self._clear_response()
        elif update_type == "vehicle_status":
            self._update_vehicle_status(update_data["vehicle_data"])

    def _update_image(self, image_data: np.ndarray):
        try:
            if not isinstance(image_data, np.ndarray):
                return

            if image_data.dtype != np.uint8:
                image_data = (
                    (image_data * 255).astype(np.uint8)
                    if image_data.max() <= 1.0
                    else image_data.astype(np.uint8)
                )

            if image_data.ndim == 2:
                h, w = image_data.shape
                qimage = QImage(image_data.data, w, h, w, QImage.Format_Grayscale8).copy()
            elif image_data.ndim == 3 and image_data.shape[2] == 3:
                h, w, _ = image_data.shape
                qimage = QImage(
                    image_data.data, w, h, 3 * w, QImage.Format_RGB888
                ).copy()
            elif image_data.ndim == 3 and image_data.shape[2] == 4:
                h, w, _ = image_data.shape
                qimage = QImage(
                    image_data.data, w, h, 4 * w, QImage.Format_RGBA8888
                ).copy()
            else:
                self.image_label.setText("Unsupported image shape")
                return

            pixmap = QPixmap.fromImage(qimage)
            target_size = self.image_label.size()
            scaled = pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
            self.image_label.setText("")
        except Exception as exc:
            print(f"Image update failed: {exc}")
            self.image_label.setText("Image load failed")

    def _update_prompt(self, prompt: str):
        try:
            self.prompt_text.setPlainText(prompt)
            self.prompt_text.moveCursor(QTextCursor.End)
        except Exception as exc:
            print(f"Prompt update failed: {exc}")

    def _append_colored_response(self, text: str, color: str = ""):
        cursor = self.response_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        if color:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            cursor.insertText(text, fmt)
        else:
            cursor.insertText(text)
        self.response_text.setTextCursor(cursor)
        self.response_text.ensureCursorVisible()

    def _append_response(self, chunk: str):
        try:
            if (
                chunk.startswith("[缓存命中]")
                or chunk.startswith("[Cache Hit]")
                or chunk.startswith("[Similar Cache Hit]")
                or chunk.startswith("[Exact Cache Hit]")
            ):
                self._append_colored_response(chunk, self.colors["cache_fg"])
            elif chunk.startswith("[错误]") or chunk.startswith("[Error]"):
                self._append_colored_response(chunk, self.colors["error_fg"])
            elif (
                chunk.startswith("[最终奖励修正值")
                or chunk.startswith("[Final Reward Adjustment")
                or chunk.startswith("[Final Reward Value")
            ):
                self._append_colored_response(chunk, self.colors["success_fg"])
            else:
                self._append_colored_response(chunk)
        except Exception as exc:
            print(f"Response append failed: {exc}")

    def _finalize_response(self, final_reward: float):
        try:
            reward_text = f"\n\n[Final Reward Adjustment: {final_reward:.4f}]"
            self._append_colored_response(reward_text, self.colors["success_fg"])
        except Exception as exc:
            print(f"Response finalize failed: {exc}")

    def _clear_response(self):
        try:
            self.response_text.clear()
        except Exception as exc:
            print(f"Response clear failed: {exc}")

    def _update_stats(self, stats: Dict[str, Any]):
        try:
            self.stats_data.update(stats)
            for key, label in self.stats_labels.items():
                if key not in self.stats_data:
                    continue
                value = self.stats_data[key]
                label.setText(f"{value:.2f}s" if key == "avg_response_time" else str(value))

                if key == "error_count" and value > 0:
                    color = self.colors["error_fg"]
                elif key == "cache_hit_count" and value > 0:
                    color = self.colors["cache_fg"]
                else:
                    color = self.colors["success_fg"]
                label.setStyleSheet(f"color: {color};")
        except Exception as exc:
            print(f"Stats update failed: {exc}")

    def _update_vehicle_status(self, vehicle_data: Dict[str, Any]):
        try:
            self.vehicle_data.update(vehicle_data)

            speed = self.vehicle_data.get("speed", 0.0)
            self.vehicle_status_labels["speed"].setText(f"{speed:.1f} km/h")

            throttle = self.vehicle_data.get("throttle", 0.0)
            self.vehicle_status_labels["throttle"].setText(f"{throttle:.3f}")
            if throttle > 0:
                throttle_color = self.colors["success_fg"]
            elif throttle < 0:
                throttle_color = self.colors["error_fg"]
            else:
                throttle_color = self.colors["accent"]
            self.vehicle_status_labels["throttle"].setStyleSheet(
                f"color: {throttle_color};"
            )

            steer = self.vehicle_data.get("steer", 0.0)
            self.vehicle_status_labels["steer"].setText(f"{steer:.3f}")
            steer_color = self.colors["cache_fg"] if abs(steer) > 0.1 else self.colors["accent"]
            self.vehicle_status_labels["steer"].setStyleSheet(f"color: {steer_color};")

            maneuver = str(self.vehicle_data.get("maneuver", "Unknown"))
            self.vehicle_status_labels["maneuver"].setText(maneuver)
        except Exception as exc:
            print(f"Vehicle status update failed: {exc}")

    def update_image(self, image: np.ndarray):
        self.update_queue.put({"type": "image", "image": image})

    def update_prompt(self, prompt: str):
        self.update_queue.put({"type": "prompt", "prompt": prompt})

    def add_response_chunk(self, chunk: str):
        self.update_queue.put({"type": "response_chunk", "chunk": chunk})

    def finalize_response(self, final_reward: float):
        self.update_queue.put({"type": "response_complete", "final_reward": final_reward})

    def update_stats(self, stats: Dict[str, Any]):
        self.update_queue.put({"type": "stats", "stats": stats})

    def clear_response(self):
        self.update_queue.put({"type": "clear"})

    def update_vehicle_status(self, speed: float, throttle: float, steer: float, maneuver: str):
        self.update_queue.put(
            {
                "type": "vehicle_status",
                "vehicle_data": {
                    "speed": speed,
                    "throttle": throttle,
                    "steer": steer,
                    "maneuver": maneuver,
                },
            }
        )