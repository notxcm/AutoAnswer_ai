"""
悬浮窗模块 - 显示解答结果
"""
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush


class AnswerOverlay(QWidget):
    """
    半透明悬浮窗，显示答案、解析和状态信息
    """
    _show_answer_signal = pyqtSignal(dict)
    _show_message_signal = pyqtSignal(str)
    _update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        # 必须开启才能实现半透明效果
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # 不使用 QPalette/autoFillBackground，改用 paintEvent 绘制圆角半透明背景

        # 圆角半径
        self._border_radius = 12

        # 布局
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)

        self.answer_label = QLabel("准备就绪")
        self.answer_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        self.answer_label.setStyleSheet("color: #00FF88; background: transparent;")
        self.answer_label.setAlignment(Qt.AlignCenter)

        self.detail_label = QLabel("")
        self.detail_label.setFont(QFont("Microsoft YaHei", 10))
        self.detail_label.setStyleSheet("color: #FFFFFF; background: transparent;")
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("状态：等待扫描")
        self.status_label.setFont(QFont("Microsoft YaHei", 8))
        self.status_label.setStyleSheet("color: #AAAAAA; background: transparent;")

        layout.addWidget(self.answer_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.resize(300, 120)

        # 连接信号
        self._show_answer_signal.connect(self._do_show_answer)
        self._show_message_signal.connect(self._do_show_message)
        self._update_status_signal.connect(self._do_update_status)

        # 默认位置：屏幕右下角
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.move(screen_geometry.width() - self.width() - 50,
                  screen_geometry.height() - self.height() - 100)
        self.show()

    def paintEvent(self, _event):
        """绘制圆角半透明背景"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), self._border_radius, self._border_radius)

    def show_answer(self, answer_data: dict):
        """显示答案（线程安全）"""
        self._show_answer_signal.emit(answer_data)

    def _do_show_answer(self, answer_data: dict):
        answer = str(answer_data.get("answer", "?"))
        # 多选答案 A,C 展示为更符合中文习惯的 A、C
        if answer.count(",") >= 1 and len(answer) <= 16:
            answer = answer.replace(",", "、")
        detail = answer_data.get("detail", "")
        source = answer_data.get("source", "")
        self.answer_label.setText(f"答案：{answer}")
        if source:
            detail = f"[{source}] {detail}"
        self.detail_label.setText(detail)
        # 自动调整窗口大小以适应内容
        self.adjustSize()
        QTimer.singleShot(5000, lambda: self.detail_label.setText(""))

    def show_message(self, message: str):
        """显示临时消息（线程安全）"""
        self._show_message_signal.emit(message)

    def _do_show_message(self, message: str):
        self.detail_label.setText(message)
        self.adjustSize()
        QTimer.singleShot(3000, lambda: self.detail_label.setText(""))

    def update_status(self, status: str):
        """更新状态栏（线程安全）"""
        self._update_status_signal.emit(status)

    def _do_update_status(self, status: str):
        self.status_label.setText(status)
