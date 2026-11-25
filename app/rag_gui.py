import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QProgressBar, QFrame,
    QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from rag_engine import RAGEngine


class LoadFileWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, engine, file_path):
        super().__init__()
        self.engine = engine
        self.file_path = file_path

    def run(self):
        try:
            self.engine.load_and_build(self.file_path, self.progress.emit)
            self.finished.emit(True, "✅ پایگاه دانش با موفقیت ساخته شد!")
        except Exception as e:
            self.finished.emit(False, f"❌ خطا: {str(e)}")


class AskWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, str)  # answer, debug_text
    error = pyqtSignal(str)

    def __init__(self, engine, question):
        super().__init__()
        self.engine = engine
        self.question = question

    def run(self):
        try:
            # Get debug info first
            docs = self.engine.get_debug_info(self.question)
            debug_text = ""
            for i, doc in enumerate(docs):
                debug_text += f"--- تکه {i+1} ---\n{doc.page_content[:400]}\n\n"

            answer = self.engine.ask(self.question, self.progress.emit)
            self.finished.emit(answer, debug_text)
        except Exception as e:
            self.error.emit(f"❌ خطا: {str(e)}")


class RAGMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = RAGEngine()
        self.file_loaded = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🧠 سیستم پرسش هوشمند از اسناد مهندسی")
        self.resize(1100, 800)

        self.setStyleSheet("""
            QMainWindow { background-color: #D4E8D0; }
            QWidget { background-color: #D4E8D0; color: #1a1a1a; }
            QLabel { color: #1565C0; font-weight: bold; }
            QTextEdit {
                background-color: #FFFFFF;
                border: 2px solid #1976D2;
                border-radius: 8px;
                padding: 10px;
                font-size: 14pt;
                color: #1a1a1a;
            }
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 13pt;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1565C0; }
            QPushButton:pressed { background-color: #0D47A1; }
            QPushButton:disabled { background-color: #90CAF9; }
            QProgressBar {
                border: 2px solid #1976D2;
                border-radius: 5px;
                text-align: center;
                background-color: #FFFFFF;
                font-size: 11pt;
            }
            QProgressBar::chunk { background-color: #1976D2; border-radius: 3px; }
            QTabWidget::pane { border: 2px solid #1976D2; border-radius: 8px; background: #FFFFFF; }
            QTabBar::tab {
                background: #A5D6A7; color: #1a1a1a;
                padding: 8px 20px; font-size: 12pt; font-weight: bold;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
            }
            QTabBar::tab:selected { background: #1976D2; color: white; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(12)

        title = QLabel("🧠 سیستم پرسش هوشمند از اسناد مهندسی")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #0D47A1;")
        layout.addWidget(title)

        subtitle = QLabel("مخصوص مهندسین - بر پایه RAG و Ollama")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: #2E7D32;")
        layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #1976D2;")
        layout.addWidget(sep)

        file_layout = QHBoxLayout()
        self.file_label = QLabel("📂 هیچ فایلی انتخاب نشده است")
        self.file_label.setFont(QFont("Segoe UI", 12))
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label, 1)
        self.btn_select_file = QPushButton("📁 انتخاب فایل PDF")
        self.btn_select_file.setFont(QFont("Segoe UI", 12))
        self.btn_select_file.clicked.connect(self.select_file)
        file_layout.addWidget(self.btn_select_file)
        layout.addLayout(file_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        q_label = QLabel("❓ سوال خود را بنویسید (فارسی یا انگلیسی):")
        q_label.setFont(QFont("Segoe UI", 13))
        layout.addWidget(q_label)

        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("مثال: درجه بالانس برای پمپ ها چقدر است؟")
        self.question_input.setMaximumHeight(80)
        self.question_input.setFont(QFont("Segoe UI", 13))
        layout.addWidget(self.question_input)

        self.btn_ask = QPushButton("🚀 پرسش از سند")
        self.btn_ask.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.btn_ask.setMinimumHeight(50)
        self.btn_ask.clicked.connect(self.ask_question)
        self.btn_ask.setEnabled(False)
        layout.addWidget(self.btn_ask)

        # Tabs for Answer and Debug
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 12))

        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("پاسخ هوشمند در اینجا نمایش داده می‌شود...")
        self.answer_output.setFont(QFont("Segoe UI", 13))
        self.tabs.addTab(self.answer_output, "💡 پاسخ")

        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        self.debug_output.setPlaceholderText("تکه‌های پیدا شده از سند در اینجا نمایش داده می‌شود...")
        self.debug_output.setFont(QFont("Consolas", 10))
        self.debug_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.tabs.addTab(self.debug_output, "🔍 دیباگ")

        layout.addWidget(self.tabs)

        footer = QLabel("طراحی شده برای مهندس رضا | Powered by Qwen 2.5 + ChromaDB")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setFont(QFont("Segoe UI", 9))
        footer.setStyleSheet("color: #666;")
        layout.addWidget(footer)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "انتخاب فایل PDF", "", "PDF Files (*.pdf)")
        if not file_path:
            return
        self.file_label.setText(f"📂 {os.path.basename(file_path)}")
        self.btn_select_file.setEnabled(False)
        self.btn_ask.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("⏳ در حال پردازش فایل...")
        self.worker = LoadFileWorker(self.engine, file_path)
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.on_file_loaded)
        self.worker.start()

    def update_status(self, message):
        self.status_label.setText(message)

    def on_file_loaded(self, success, message):
        self.progress_bar.setVisible(False)
        self.btn_select_file.setEnabled(True)
        self.status_label.setText(message)
        if success:
            self.file_loaded = True
            self.btn_ask.setEnabled(True)
            self.status_label.setStyleSheet("color: #2E7D32; font-weight: bold;")
        else:
            self.file_loaded = False
            self.status_label.setStyleSheet("color: #C62828; font-weight: bold;")

    def ask_question(self):
        question = self.question_input.toPlainText().strip()
        if not question:
            self.answer_output.setText("⚠️ لطفاً سوالی بنویسید.")
            return
        if not self.file_loaded:
            self.answer_output.setText("⚠️ ابتدا باید یک فایل PDF بارگذاری کنید.")
            return
        self.btn_ask.setEnabled(False)
        self.btn_select_file.setEnabled(False)
        self.answer_output.setText("⏳ در حال پردازش...")
        self.progress_bar.setVisible(True)
        self.ask_worker = AskWorker(self.engine, question)
        self.ask_worker.progress.connect(self.update_status)
        self.ask_worker.finished.connect(self.on_answer_ready)
        self.ask_worker.error.connect(self.on_answer_error)
        self.ask_worker.start()

    def on_answer_ready(self, answer, debug_text):
        self.progress_bar.setVisible(False)
        self.btn_ask.setEnabled(True)
        self.btn_select_file.setEnabled(True)
        self.answer_output.setText(answer)
        self.debug_output.setText(debug_text)
        self.tabs.setCurrentIndex(0)
        self.status_label.setText("✅ پاسخ آماده است.")
        self.status_label.setStyleSheet("color: #2E7D32; font-weight: bold;")

    def on_answer_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.btn_ask.setEnabled(True)
        self.btn_select_file.setEnabled(True)
        self.answer_output.setText(error_msg)
        self.status_label.setStyleSheet("color: #C62828; font-weight: bold;")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = RAGMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()