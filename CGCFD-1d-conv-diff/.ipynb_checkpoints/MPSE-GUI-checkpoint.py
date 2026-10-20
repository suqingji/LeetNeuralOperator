import sys
import time
import ctypes
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDockWidget,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QStackedWidget,
    QLabel, QPushButton, QFormLayout, QLineEdit, QToolBar, QMessageBox
)
import qtawesome as qta  # 引入强大的矢量图标库

# ==========================================
# 1. 后端计算线程 (Worker Thread)
# ==========================================
class ComputeThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def run(self):
        self.log_signal.emit(">>> 开始解析模型树对象...")
        time.sleep(0.5)
        self.log_signal.emit(">>> 正在划分有限元网格...")
        time.sleep(1.0)
        self.log_signal.emit(">>> 正在组装刚度矩阵并求解...")
        time.sleep(2.0)
        self.log_signal.emit(">>> 计算完成")
        self.finished_signal.emit()

# ==========================================
# 2. 核心主控界面 (Main Window / Controller)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PolyGalerkin Multiphysics (Modern UI)")
        self.resize(1300, 850)

        # 1. 应用极简现代化的 QSS 浅色主题
        self.apply_modern_stylesheet()

        self.compute_thread = None

        # 2. 搭建组件
        self.setup_ribbon_toolbar()
        self.setup_central_widget()
        self.setup_dock_widgets()

    def apply_modern_stylesheet(self):
        """完全仿照 COMSOL 的浅色扁平化主题"""
        qss = """
        QWidget {
            font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            font-size: 13px;
            color: #333333;
        }
        QMainWindow { background-color: #f3f3f3; }
        
        /* Ribbon 风格工具栏 */
        QToolBar {
            background-color: #ffffff;
            border-bottom: 1px solid #dcdcdc;
            spacing: 15px;
            padding: 5px;
        }
        QToolButton {
            padding: 5px;
            border-radius: 4px;
        }
        QToolButton:hover { background-color: #e5f3ff; border: 1px solid #cce8ff; }
        
        /* 停靠面板 */
        QDockWidget {
            font-weight: bold;
            color: #00509d;
            border: 1px solid #dcdcdc;
        }
        QDockWidget::title {
            background: #f7f7f7;
            padding: 8px;
            border-bottom: 1px solid #dcdcdc;
        }
        
        /* 模型树 */
        QTreeWidget {
            background-color: #ffffff;
            border: none;
        }
        QTreeWidget::item {
            padding: 4px;
            border-radius: 3px;
            margin: 1px;
        }
        QTreeWidget::item:hover { background-color: #f0f8ff; }
        QTreeWidget::item:selected {
            background-color: #cce8ff;
            color: #000000;
        }
        
        /* 日志窗口与面板 */
        QTextEdit { background-color: #ffffff; border: none; font-family: Consolas; }
        QPushButton {
            background-color: #0078d7; color: white;
            border: none; padding: 6px 15px; border-radius: 4px; font-weight: bold;
        }
        QPushButton:hover { background-color: #005a9e; }
        QPushButton:disabled { background-color: #cccccc; color: #666666; }
        QLineEdit { border: 1px solid #cccccc; padding: 4px; border-radius: 3px; }
        """
        self.setStyleSheet(qss)

    def setup_ribbon_toolbar(self):
        """仿 COMSOL 的大型带字图标工具栏"""
        toolbar = QToolBar("主功能区")
        toolbar.setIconSize(QSize(32, 32)) 
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon) 
        self.addToolBar(toolbar)

        act_new = QAction(qta.icon('fa5s.file-alt', color='#00509d'), "新建", self)
        act_open = QAction(qta.icon('fa5s.folder-open', color='#e6a817'), "打开", self)
        act_save = QAction(qta.icon('fa5s.save', color='#555555'), "保存", self)
        
        self.action_compute = QAction(qta.icon('fa5s.play', color='#28a745'), "计算", self)
        self.action_compute.triggered.connect(self.start_computation)

        toolbar.addAction(act_new)
        toolbar.addAction(act_open)
        toolbar.addAction(act_save)
        toolbar.addSeparator()
        
        toolbar.addAction(QAction(qta.icon('fa5s.cube', color='#17a2b8'), "几何", self))
        toolbar.addAction(QAction(qta.icon('fa5s.atom', color='#6f42c1'), "物理场", self))
        toolbar.addAction(QAction(qta.icon('fa5s.border-all', color='#6c757d'), "网格", self))
        
        toolbar.addSeparator()
        toolbar.addAction(self.action_compute)

    def setup_central_widget(self):
        central = QWidget()
        central.setStyleSheet("background-color: #ffffff; border-left: 1px solid #dcdcdc; border-right: 1px solid #dcdcdc;")
        layout = QVBoxLayout(central)
        
        lbl = QLabel("3D 图形渲染区 (Graphic View)\n可在此嵌入 PyVista/VTK 视口")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #999999; font-size: 20px; font-weight: bold;")
        layout.addWidget(lbl)
        self.setCentralWidget(central)

    def setup_dock_widgets(self):
        # === 左侧模型树 ===
        self.dock_tree = QDockWidget("模型开发器", self)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderHidden(True)
        
        tree_data = [
            ("全局定义", 'fa5s.globe', '#00509d'),
            ("几何", 'fa5s.cube', '#17a2b8'),
            ("材料", 'fa5s.swatchbook', '#e6a817'),
            ("物理场", 'fa5s.atom', '#6f42c1'),
            ("网格", 'fa5s.border-all', '#6c757d'),
            ("研究", 'fa5s.cogs', '#555555'),
            ("结果", 'fa5s.chart-bar', '#28a745')
        ]
        
        self.tree_items = {}
        for text, icon_name, color in tree_data:
            item = QTreeWidgetItem([text])
            item.setIcon(0, qta.icon(icon_name, color=color))
            self.tree_widget.addTopLevelItem(item)
            self.tree_items[text.strip()] = item
            
        self.dock_tree.setWidget(self.tree_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_tree)

        # === 右侧属性面板 ===
        self.dock_settings = QDockWidget("设置", self)
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("background-color: #ffffff;")
        
        # 页面 0: 默认空页面
        page_empty = QLabel("请在左侧模型树中选择节点", alignment=Qt.AlignCenter)
        self.stacked_widget.addWidget(page_empty)

        # 页面 1: 几何设置面板
        page_geo = QWidget()
        layout_geo = QVBoxLayout(page_geo)
        
        # 修复点：使用水平布局组合 QLabel 实现带图标的标题
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 10)
        
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon('fa5s.cube', color='#17a2b8').pixmap(QSize(20, 20)))
        text_label = QLabel("几何设置")
        text_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        
        title_layout.addWidget(icon_label)
        title_layout.addWidget(text_label)
        title_layout.addStretch()
        
        form_layout = QFormLayout()
        form_layout.addRow("宽度 (X):", QLineEdit("1.0 [m]"))
        form_layout.addRow("深度 (Y):", QLineEdit("2.0 [m]"))
        form_layout.addRow("高度 (Z):", QLineEdit("0.5 [m]"))
        
        layout_geo.addWidget(title_widget)
        layout_geo.addLayout(form_layout)
        layout_geo.addStretch()
        self.stacked_widget.addWidget(page_geo)

        # 页面 2: 材料设置面板
        page_mat = QWidget()
        layout_mat = QVBoxLayout(page_mat)
        
        mat_title_widget = QWidget()
        mat_title_layout = QHBoxLayout(mat_title_widget)
        mat_title_layout.setContentsMargins(0, 0, 0, 10)
        mat_icon_label = QLabel()
        mat_icon_label.setPixmap(qta.icon('fa5s.swatchbook', color='#e6a817').pixmap(QSize(20, 20)))
        mat_text_label = QLabel("力学材料属性")
        mat_text_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        mat_title_layout.addWidget(mat_icon_label)
        mat_title_layout.addWidget(mat_text_label)
        mat_title_layout.addStretch()

        form_layout_mat = QFormLayout()
        form_layout_mat.addRow("杨氏模量 (E):", QLineEdit("200e9 [Pa]"))
        form_layout_mat.addRow("泊松比 (nu):", QLineEdit("0.33"))
        form_layout_mat.addRow("密度 (rho):", QLineEdit("7850 [kg/m^3]"))
        
        layout_mat.addWidget(mat_title_widget)
        layout_mat.addLayout(form_layout_mat)
        layout_mat.addStretch()
        self.stacked_widget.addWidget(page_mat)

        # 页面 3: 研究 (计算) 面板
        page_study = QWidget()
        layout_study = QVBoxLayout(page_study)
        
        study_title_widget = QWidget()
        study_title_layout = QHBoxLayout(study_title_widget)
        study_title_layout.setContentsMargins(0, 0, 0, 10)
        study_icon_label = QLabel()
        study_icon_label.setPixmap(qta.icon('fa5s.cogs', color='#555555').pixmap(QSize(20, 20)))
        study_text_label = QLabel("研究与求解器")
        study_text_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        study_title_layout.addWidget(study_icon_label)
        study_title_layout.addWidget(study_text_label)
        study_title_layout.addStretch()

        self.btn_compute_in_panel = QPushButton("▶ 开始计算 (Compute)")
        self.btn_compute_in_panel.clicked.connect(self.start_computation)
        
        layout_study.addWidget(study_title_widget)
        layout_study.addWidget(self.btn_compute_in_panel)
        layout_study.addStretch()
        self.stacked_widget.addWidget(page_study)

        self.dock_settings.setWidget(self.stacked_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_settings)

        # === 底部日志 ===
        self.dock_log = QDockWidget("进度与日志", self)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.dock_log.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_log)

        # 绑定树节点点击信号，更新右侧面板
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        # 默认选中几何节点展开演示
        self.tree_widget.setCurrentItem(self.tree_items["几何"])
        self.on_tree_item_clicked(self.tree_items["几何"], 0)

    @Slot(QTreeWidgetItem, int)
    def on_tree_item_clicked(self, item, column):
        text = item.text(0)
        if "几何" in text:
            self.stacked_widget.setCurrentIndex(1)
        elif "材料" in text:
            self.stacked_widget.setCurrentIndex(2)
        elif "研究" in text:
            self.stacked_widget.setCurrentIndex(3)
        else:
            self.stacked_widget.setCurrentIndex(0)

    @Slot()
    def start_computation(self):
        if self.compute_thread is not None and self.compute_thread.isRunning():
            return

        self.action_compute.setEnabled(False)
        self.btn_compute_in_panel.setEnabled(False)
        self.log_text.clear()
        
        self.compute_thread = ComputeThread()
        self.compute_thread.log_signal.connect(self.update_log)
        self.compute_thread.finished_signal.connect(self.on_computation_finished)
        self.compute_thread.start()

    @Slot(str)
    def update_log(self, message):
        self.log_text.append(message)

    @Slot()
    def on_computation_finished(self):
        self.action_compute.setEnabled(True)
        self.btn_compute_in_panel.setEnabled(True)
        QMessageBox.information(self, "求解器通知", "计算完成！\n请切换到“结果”节点查看可视化数据。")

# ==========================================
# 3. 程序入口
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 1. Windows 任务栏图标修复：声明独立进程
    # 告诉 Windows 不要把这个程序折叠在 python.exe 默认的图标下
    if sys.platform == 'win32':
        # 这里的字符串可以任意自定义，只要确保唯一即可
        myappid = 'mycompany.polygalerkin.multiphysics.1.0'
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
            
    # 2. 设置全局应用程序图标（左上角和任务栏都会生效）
    # 方案 A: 使用 qtawesome 自带的矢量图标（推荐，无需外部文件）
    app.setWindowIcon(qta.icon('fa5s.cube', color='#00509d'))
    
    # 方案 B: 如果你有自己设计的 .ico 或 .png 图标文件，请注释掉方案A，使用下面这行：
    # app.setWindowIcon(QIcon('path/to/your_logo.ico'))
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())