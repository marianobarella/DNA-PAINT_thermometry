import os
# Force qtpy to use PyQt5 for better appearance
os.environ['QT_API'] = 'pyqt5'

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QSettings
from qtpy.QtGui import QIntValidator
import threading
import subprocess
import yaml



class AnalysisThread(threading.Thread):
    def __init__(self, selected_file, working_folder, steps_to_execute, params):
        super().__init__()
        self.selected_file = selected_file[0]
        self.working_folder = working_folder
        self.steps_to_execute = steps_to_execute
        self.params = params

    def run(self):
        params_str = str(self.params)
        
        # Use the virtual environment's Python executable
        venv_python = os.path.join("thermometry_env_new", "Scripts", "python.exe")
        python_executable = venv_python if os.path.exists(venv_python) else "python"
        
        command = ([python_executable, "find_super_resolved_temp_clean.py", "--selected_file", self.selected_file, "--working_folder",
                    self.working_folder, "--step"] + [str(step) for step in self.steps_to_execute] + ["--params", params_str])
        result = subprocess.run(command)
        if result.returncode == 0:
            print("Subprocess finished successfully.")
        else:
            print(f"Command failed with return code {result.returncode}")


class Ui_Parameters(object):
    def setupUi(self, Parameters):
        size = 550
        size_y = 570
        Parameters.setObjectName("Parameters")
        Parameters.resize(size, size_y)  # Increased from 500x490 to show all controls
        Parameters.setMinimumSize(QtCore.QSize(size, size_y))  # Increased minimum size
        Parameters.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.centralwidget = QtWidgets.QWidget(parent=Parameters)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_6.setSpacing(6)  # Reduced spacing between main elements from 8
        self.verticalLayout_6.setContentsMargins(8, 8, 8, 8)  # Reduced all margins
        self.groupBox_5 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_5.setWhatsThis("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(6, 6, 6, 6)  # Reduced margins from 8
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.checkStep1 = QtWidgets.QCheckBox(parent=self.groupBox_5)
        self.checkStep1.setChecked(True)
        self.checkStep1.setObjectName("checkStep1")
        self.horizontalLayout_5.addWidget(self.checkStep1)
        self.checkStep2 = QtWidgets.QCheckBox(parent=self.groupBox_5)
        self.checkStep2.setChecked(True)
        self.checkStep2.setObjectName("checkStep2")
        self.horizontalLayout_5.addWidget(self.checkStep2)
        self.checkStep3 = QtWidgets.QCheckBox(parent=self.groupBox_5)
        self.checkStep3.setChecked(True)
        self.checkStep3.setObjectName("checkStep3")
        self.horizontalLayout_5.addWidget(self.checkStep3)
        self.checkStep4 = QtWidgets.QCheckBox(parent=self.groupBox_5)
        self.checkStep4.setChecked(True)
        self.checkStep4.setObjectName("checkStep4")
        self.horizontalLayout_5.addWidget(self.checkStep4)
        self.verticalLayout_7.addLayout(self.horizontalLayout_5)
        self.checkVerbose = QtWidgets.QCheckBox(parent=self.groupBox_5)
        self.checkVerbose.setObjectName("checkVerbose")
        self.verticalLayout_7.addWidget(self.checkVerbose)
        self.checkPositionAveraging = QtWidgets.QCheckBox(parent=self.groupBox_5)
        self.checkPositionAveraging.setObjectName("checkPositionAveraging")
        self.verticalLayout_7.addWidget(self.checkPositionAveraging)
        self.horizontalLayout_7.addLayout(self.verticalLayout_7)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.saveButton = QtWidgets.QPushButton(parent=self.groupBox_5)
        self.saveButton.setObjectName("saveButton")
        self.verticalLayout.addWidget(self.saveButton, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.runButton = QtWidgets.QPushButton(parent=self.groupBox_5)
        self.runButton.setObjectName("runButton")
        self.verticalLayout.addWidget(self.runButton)
        self.horizontalLayout_7.addLayout(self.verticalLayout)
        self.verticalLayout_6.addWidget(self.groupBox_5)
        
        # Use a scroll area to ensure all content is accessible
        self.scrollArea = QtWidgets.QScrollArea(parent=self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scrollContent = QtWidgets.QWidget()
        self.scrollContent.setContentsMargins(0, 0, 0, 5)  # Add extra padding at the bottom
        self.scrollArea.setWidget(self.scrollContent)
        
        self.gridLayout = QtWidgets.QGridLayout(self.scrollContent)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setSpacing(8)  # Slightly reduced spacing
        self.gridLayout.setContentsMargins(0, 0, 0, 8)  # Slightly reduced bottom padding
        
        # Set stretching factors for the grid columns to ensure proper resizing
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 1)
        
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.setSpacing(4)  # Further reduced spacing from 6
        self.horizontalLayout.setContentsMargins(6, 6, 6, 6)  # Reduced margins from 8
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        # Configure form layout for proper resizing
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.formLayout.setVerticalSpacing(4)  # Reduced vertical spacing from 6
        self.checkNP = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.checkNP.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkNP.setIconSize(QtCore.QSize(17, 16))
        self.checkNP.setObjectName("checkNP")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.checkNP)
        self.checkPlot = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.checkPlot.setObjectName("checkPlot")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.checkPlot)
        self.checkDBSCAN = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.checkDBSCAN.setObjectName("checkDBSCAN")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.checkDBSCAN)
        self.exposureTimeLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.exposureTimeLabel.setObjectName("exposureTimeLabel")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.exposureTimeLabel)
        self.pixelSizeLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.pixelSizeLabel.setObjectName("pixelSizeLabel")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.pixelSizeLabel)
        self.sizeToAverageLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.sizeToAverageLabel.setObjectName("sizeToAverageLabel")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.LabelRole, self.sizeToAverageLabel)
        self.sitesLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.sitesLabel.setObjectName("sitesLabel")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.LabelRole, self.sitesLabel)
        self.exposure_time = QtWidgets.QDoubleSpinBox(parent=self.groupBox_2)
        self.exposure_time.setProperty("value", 0.1)
        self.exposure_time.setObjectName("exposure_time")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.exposure_time)
        self.pixel_size = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.pixel_size.setMaximum(1000000)
        self.pixel_size.setProperty("value", 130)
        self.pixel_size.setObjectName("pixel_size")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.pixel_size)
        self.size_to_average = QtWidgets.QDoubleSpinBox(parent=self.groupBox_2)
        self.size_to_average.setProperty("value", 0.2)
        self.size_to_average.setObjectName("size_to_average")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.FieldRole, self.size_to_average)
        self.docking_sites = QtWidgets.QSpinBox(parent=self.groupBox_2)
        self.docking_sites.setProperty("value", 3)
        self.docking_sites.setObjectName("docking_sites")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.FieldRole, self.docking_sites)
        self.horizontalLayout.addLayout(self.formLayout)
        self.gridLayout.addWidget(self.groupBox_2, 2, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_3.setSpacing(4)  # Further reduced spacing from 6
        self.horizontalLayout_3.setContentsMargins(6, 6, 6, 6)  # Reduced margins from 8
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        # Configure form layout for proper resizing
        self.formLayout_2.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.formLayout_2.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.formLayout_2.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.formLayout_2.setVerticalSpacing(4)  # Reduced vertical spacing from 6
        self.photonLabel = QtWidgets.QLabel(parent=self.groupBox_3)
        self.photonLabel.setObjectName("photonLabel")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.photonLabel)
        self.photon_threshold = QtWidgets.QDoubleSpinBox(parent=self.groupBox_3)
        self.photon_threshold.setMaximum(1000000.0)
        self.photon_threshold.setProperty("value", 300.0)
        self.photon_threshold.setObjectName("photon_threshold")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.photon_threshold)
        self.bkgLabel = QtWidgets.QLabel(parent=self.groupBox_3)
        self.bkgLabel.setObjectName("bkgLabel")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.bkgLabel)
        self.background_level = QtWidgets.QDoubleSpinBox(parent=self.groupBox_3)
        self.background_level.setMaximum(1000000.0)
        self.background_level.setProperty("value", 600.0)
        self.background_level.setObjectName("background_level")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.background_level)
        self.maskLabel = QtWidgets.QLabel(parent=self.groupBox_3)
        self.maskLabel.setObjectName("maskLabel")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.maskLabel)
        self.mask = QtWidgets.QComboBox(parent=self.groupBox_3)
        self.mask.setObjectName("mask")
        self.mask.addItem("")
        self.mask.addItem("")
        self.mask.addItem("")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.mask)
        self.checkSingles = QtWidgets.QCheckBox(parent=self.groupBox_3)
        self.checkSingles.setObjectName("checkSingles")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.checkSingles)
        self.horizontalLayout_3.addLayout(self.formLayout_2)
        self.gridLayout.addWidget(self.groupBox_3, 1, 1, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.groupBox_4.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_4.setSpacing(4)  # Further reduced spacing from 6
        self.verticalLayout_4.setContentsMargins(6, 6, 6, 6)  # Added reduced margins
        
        # Add top controls (checkboxes) in a horizontal layout
        self.topControlsLayout = QtWidgets.QHBoxLayout()
        self.topControlsLayout.setObjectName("topControlsLayout")
        self.checkOptimizationDisplay = QtWidgets.QCheckBox(parent=self.groupBox_4)
        self.checkOptimizationDisplay.setIconSize(QtCore.QSize(17, 16))
        self.checkOptimizationDisplay.setChecked(False)
        self.checkOptimizationDisplay.setObjectName("checkOptimizationDisplay")
        self.topControlsLayout.addWidget(self.checkOptimizationDisplay)
        self.checkHyperExponential = QtWidgets.QCheckBox(parent=self.groupBox_4)
        self.checkHyperExponential.setObjectName("checkHyperExponential")
        self.topControlsLayout.addWidget(self.checkHyperExponential)
        self.verticalLayout_4.addLayout(self.topControlsLayout)
        
        # Add initialization group box
        self.groupBox_6 = QtWidgets.QGroupBox(parent=self.groupBox_4)
        self.groupBox_6.setObjectName("groupBox_6")
        self.groupBox_6.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.horizontalLayout_6.setSpacing(6)  # Reduced spacing from 8
        self.horizontalLayout_6.setContentsMargins(6, 6, 6, 6)  # Reduced margins from 8
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_3.setSpacing(8)
        self.long_tauLabel = QtWidgets.QLabel(parent=self.groupBox_6)
        self.long_tauLabel.setObjectName("long_tauLabel")
        self.verticalLayout_3.addWidget(self.long_tauLabel)
        self.short_tauLabel = QtWidgets.QLabel(parent=self.groupBox_6)
        self.short_tauLabel.setObjectName("short_tauLabel")
        self.verticalLayout_3.addWidget(self.short_tauLabel)
        self.ratioLabel = QtWidgets.QLabel(parent=self.groupBox_6)
        self.ratioLabel.setObjectName("ratioLabel")
        self.verticalLayout_3.addWidget(self.ratioLabel)
        self.horizontalLayout_6.addLayout(self.verticalLayout_3)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_5.setSpacing(8)  # Add spacing between input fields
        self.long_tau = QtWidgets.QDoubleSpinBox(parent=self.groupBox_6)
        self.long_tau.setMaximum(10000.0)
        self.long_tau.setProperty("value", 5.0)
        self.long_tau.setObjectName("long_tau")
        self.verticalLayout_5.addWidget(self.long_tau)
        self.short_tau = QtWidgets.QDoubleSpinBox(parent=self.groupBox_6)
        self.short_tau.setProperty("value", 0.1)
        self.short_tau.setObjectName("short_tau")
        self.verticalLayout_5.addWidget(self.short_tau)
        self.ratio = QtWidgets.QDoubleSpinBox(parent=self.groupBox_6)
        self.ratio.setProperty("value", 1.0)
        self.ratio.setObjectName("ratio")
        self.verticalLayout_5.addWidget(self.ratio)
        self.horizontalLayout_6.addLayout(self.verticalLayout_5)
        self.verticalLayout_4.addWidget(self.groupBox_6)
        
        # Add bottom controls in a form layout
        self.bottomControlsLayout = QtWidgets.QFormLayout()
        self.bottomControlsLayout.setObjectName("bottomControlsLayout")
        self.bottomControlsLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.bottomControlsLayout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.bottomControlsLayout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.bottomControlsLayout.setVerticalSpacing(4)  # Reduced from 6
        
        # Range controls
        self.rangeLabel = QtWidgets.QLabel(parent=self.groupBox_4)
        self.rangeLabel.setObjectName("rangeLabel")
        self.bottomControlsLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.rangeLabel)
        self.range = QtWidgets.QLineEdit(parent=self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.range.sizePolicy().hasHeightForWidth())
        self.range.setSizePolicy(sizePolicy)
        self.range.setAutoFillBackground(False)
        self.range.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.range.setObjectName("range")
        self.bottomControlsLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.range)
        
        # Likelihood error controls
        self.likelihoodErrorLabel = QtWidgets.QLabel(parent=self.groupBox_4)
        self.likelihoodErrorLabel.setObjectName("likelihoodErrorLabel")
        self.bottomControlsLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.likelihoodErrorLabel)
        self.likelihood_error = QtWidgets.QDoubleSpinBox(parent=self.groupBox_4)
        self.likelihood_error.setProperty("value", 2.0)
        self.likelihood_error.setObjectName("likelihood_error")
        self.bottomControlsLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.likelihood_error)
        
        self.verticalLayout_4.addLayout(self.bottomControlsLayout)
        
        # Add to the grid
        self.gridLayout.addWidget(self.groupBox_4, 2, 1, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout_2.setSpacing(4)  # Further reduced spacing from 6
        self.horizontalLayout_2.setContentsMargins(6, 6, 6, 6)  # Reduced margins from 8
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.checkRecursive = QtWidgets.QCheckBox(parent=self.groupBox)
        self.checkRecursive.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkRecursive.setObjectName("checkRecursive")
        self.verticalLayout_2.addWidget(self.checkRecursive)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setObjectName("formLayout_4")
        # Configure form layout for proper resizing
        self.formLayout_4.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.formLayout_4.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.formLayout_4.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.formLayout_4.setVerticalSpacing(4)  # Reduced vertical spacing from 6
        self.lpxLabel = QtWidgets.QLabel(parent=self.groupBox)
        self.lpxLabel.setObjectName("lpxLabel")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.lpxLabel)
        self.lpx = QtWidgets.QDoubleSpinBox(parent=self.groupBox)
        self.lpx.setToolTip("")
        self.lpx.setProperty("value", 0.15)
        self.lpx.setObjectName("lpx")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.lpx)
        self.lpyLabel = QtWidgets.QLabel(parent=self.groupBox)
        self.lpyLabel.setObjectName("lpyLabel")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.lpyLabel)
        self.lpy = QtWidgets.QDoubleSpinBox(parent=self.groupBox)
        self.lpy.setToolTip("")
        self.lpy.setProperty("value", 0.15)
        self.lpy.setObjectName("lpy")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.lpy)
        self.verticalLayout_2.addLayout(self.formLayout_4)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 1)
        self.verticalLayout_6.addWidget(self.scrollArea)
        Parameters.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=Parameters)
        self.statusbar.setObjectName("statusbar")
        Parameters.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(parent=Parameters)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 713, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(parent=self.menubar)
        self.menuFile.setObjectName("menuFile")
        Parameters.setMenuBar(self.menubar)
        self.actionOpen = QtGui.QAction(parent=Parameters)
        self.actionOpen.setObjectName("actionOpen")
        self.actionRun = QtGui.QAction(parent=Parameters)
        self.actionRun.setObjectName("actionRun")
        self.actionReset_Parameters = QtGui.QAction(parent=Parameters)
        self.actionReset_Parameters.setObjectName("actionReset_Parameters")
        self.actionSave_Parameters = QtGui.QAction(parent=Parameters)
        self.actionSave_Parameters.setObjectName("actionSave_Parameters")
        self.actionOpen_2 = QtGui.QAction(parent=Parameters)
        self.actionOpen_2.setObjectName("actionOpen_2")
        self.menuFile.addAction(self.actionReset_Parameters)
        self.menuFile.addAction(self.actionSave_Parameters)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(Parameters)
        self.mask.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(Parameters)

        # DON'T REMOVE BELOW
        # The stored parameters.
        self.settings_parameters = QSettings('Parameter GUI', 'Parameters')



        # Get previously stored values.
        def convert_parameters_to_text(parameter, value):
            value = str(value)
            if value == 'false':
                value = False
            if value == 'true':
                value = True
            convert_dict = {
                "step1": lambda: self.checkStep1.setChecked(value),
                "step2": lambda: self.checkStep2.setChecked(value),
                "step3": lambda: self.checkStep3.setChecked(value),
                "step4": lambda: self.checkStep4.setChecked(value),
                "verbose": lambda: self.checkVerbose.setChecked(value),
                "mask_singles": lambda: self.checkSingles.setChecked(value),
                "recursive": lambda: self.checkRecursive.setChecked(value),
                "lpx_filter": lambda: self.lpx.setValue(float(value)),
                "lpy_filter": lambda: self.lpy.setValue(float(value)),
                "exposure_time": lambda: self.exposure_time.setValue(float(value)),
                "docking_sites": lambda: self.docking_sites.setValue(int(value)),
                "checkNP": lambda: self.checkNP.setChecked(value),
                "pixel_size": lambda: self.pixel_size.setValue(int(value)),
                "size_to_average": lambda: self.size_to_average.setValue(float(value)),
                "th": lambda: None,
                "checkPlot": lambda: self.checkPlot.setChecked(value),
                "photon_threshold": lambda: self.photon_threshold.setValue(float(value)),
                "background_level": lambda: self.background_level.setValue(float(value)),
                "mask_level": lambda: self.mask.setCurrentText(value),
                "likelihood_error": lambda: self.likelihood_error.setValue(float(value)),
                "checkOptimizationDisplay": lambda: self.checkOptimizationDisplay.setChecked(value),
                "checkHyperExponential": lambda: self.checkHyperExponential.setChecked(value),
                "use_position_averaging": lambda: self.checkPositionAveraging.setChecked(value),
                "use_dbscan": lambda: self.checkDBSCAN.setChecked(value)
            }
            if parameter in convert_dict:
                convert_dict[parameter]()
        try:
            for parameter in self.settings_parameters.allKeys():
                # self.parameters[parameter] = self.settings_parameters.value(parameter)
                convert_parameters_to_text(parameter, self.settings_parameters.value(parameter))
        except:
            pass


        # Default parameters.
        def load_default_parameters():
            try:
                self.settings_default_parameters = QSettings('Parameter GUI', 'Default Parameters')
                for parameter in self.settings_default_parameters.allKeys():
                    # self.parameters[parameter] = self.settings_parameters.value(parameter)
                    convert_parameters_to_text(parameter, self.settings_default_parameters.value(parameter))
            except:
                print('Save a set of parameters as default parameters first.')

        self.actionReset_Parameters.triggered.connect(lambda: load_default_parameters())

        # Connect save buttons to our new methods
        self.saveButton.clicked.connect(self.save_parameters)
        self.actionSave_Parameters.triggered.connect(self.save_default_parameters)
        
        # Initialize greyed out widgets.
        self.short_tau.setEnabled(self.checkHyperExponential.isChecked())
        self.short_tauLabel.setEnabled(self.checkHyperExponential.isChecked())

        self.ratio.setEnabled(self.checkHyperExponential.isChecked())
        self.ratioLabel.setEnabled(self.checkHyperExponential.isChecked())

        # When pressing run, run the run() function.
        self.runButton.clicked.connect(lambda: self.run())
        self.actionRun.triggered.connect(lambda: self.run())
        #self.actionOpen_2.triggered.connect(lambda: self.two_channels())

        # Disable short tau and ratio initialization if we are not fitting hyper exp.
        def disable_widget(widget, bool_variable):
            widget.setEnabled(bool_variable)

        self.checkHyperExponential.stateChanged.connect(
            lambda: disable_widget(self.short_tau, self.checkHyperExponential.isChecked()))
        self.checkHyperExponential.stateChanged.connect(
            lambda: disable_widget(self.short_tauLabel, self.checkHyperExponential.isChecked()))
        self.checkHyperExponential.stateChanged.connect(
            lambda: disable_widget(self.ratio, self.checkHyperExponential.isChecked()))
        self.checkHyperExponential.stateChanged.connect(
            lambda: disable_widget(self.ratioLabel, self.checkHyperExponential.isChecked()))

        # Set a small consistent minimum width for all inputs to ensure usability
        # but don't restrict the height for better dynamic sizing
        for widget in [self.exposure_time, self.pixel_size, self.size_to_average,
                      self.docking_sites, self.photon_threshold, self.background_level,
                      self.mask, self.long_tau, self.short_tau, self.ratio,
                      self.range, self.likelihood_error, self.lpx, self.lpy]:
            widget.setMinimumWidth(70)  # Use a smaller minimum width

    def retranslateUi(self, Parameters):
        _translate = QtCore.QCoreApplication.translate
        Parameters.setWindowTitle(_translate("Parameters", "Parameters"))
        self.groupBox_5.setTitle(_translate("Parameters", "Run"))
        self.checkStep1.setToolTip(_translate("Parameters",
                                              "This step reads and saves the output of Picasso software into seperate .dat files."))
        self.checkStep1.setText(_translate("Parameters", "Step 1"))
        self.checkStep2.setToolTip(_translate("Parameters", "This step analyzes already-processed Picasso data."))
        self.checkStep2.setText(_translate("Parameters", "Step 2"))
        self.checkStep3.setToolTip(_translate("Parameters",
                                              "This step produces traces of different localization metrics for every binding site."))
        self.checkStep3.setText(_translate("Parameters", "Step 3"))
        self.checkStep4.setToolTip(
            _translate("Parameters", "This step finds the kinetic binding time of an imager probe."))
        self.checkStep4.setText(_translate("Parameters", "Step 4"))
        self.checkVerbose.setText(_translate("Parameters", "Verbose"))
        self.checkPositionAveraging.setToolTip(_translate("Parameters", "Use position averaging method that integrates kinetics analysis into Step 2 and skips Step 3."))
        self.checkPositionAveraging.setText(_translate("Parameters", "Position Averaging"))
        self.saveButton.setText(_translate("Parameters", "Save Parameters"))
        self.runButton.setText(_translate("Parameters", "Run"))
        self.groupBox_2.setTitle(_translate("Parameters", "Step 2"))
        self.checkNP.setToolTip(_translate("Parameters", "Check if there are nanoparticles present in the picks."))
        self.checkNP.setText(_translate("Parameters", "Nanoparticle"))
        self.checkPlot.setToolTip(_translate("Parameters", "Check for plots."))
        self.checkPlot.setText(_translate("Parameters", "Plot"))
        self.checkDBSCAN.setToolTip(_translate("Parameters", "Use DBSCAN clustering instead of peak detection for binding site identification."))
        self.checkDBSCAN.setText(_translate("Parameters", "Use DBSCAN"))
        self.exposureTimeLabel.setToolTip(_translate("Parameters", "Exposure time of the camera."))
        self.exposureTimeLabel.setText(_translate("Parameters", "Exposure Time (s):"))
        self.pixelSizeLabel.setToolTip(_translate("Parameters", "Pixel size of the camera."))
        self.pixelSizeLabel.setText(_translate("Parameters", "Pixel Size (nm):"))
        self.sizeToAverageLabel.setToolTip(
            _translate("Parameters", "The size of the area to average around a docking site in camera pixels."))
        self.sizeToAverageLabel.setText(_translate("Parameters", "Averaging size (pixels):"))
        self.sitesLabel.setToolTip(_translate("Parameters", "How many docking sites are present on the DNA Origami."))
        self.sitesLabel.setText(_translate("Parameters", "Docking Sites:"))
        self.groupBox_3.setTitle(_translate("Parameters", "Step 3"))
        self.photonLabel.setToolTip(_translate("Parameters",
                                               "The least amount of photons registered in order to be considered a localization."))
        self.photonLabel.setText(_translate("Parameters", "Photon Threshold:"))
        self.bkgLabel.setToolTip(_translate("Parameters", "The background level in photons."))
        self.bkgLabel.setText(_translate("Parameters", "Background Level (photons):"))
        self.maskLabel.setText(_translate("Parameters", "Mask Level:"))
        self.mask.setCurrentText(_translate("Parameters", "2"))
        self.mask.setItemText(0, _translate("Parameters", "0"))
        self.mask.setItemText(1, _translate("Parameters", "1"))
        self.mask.setItemText(2, _translate("Parameters", "2"))
        self.checkSingles.setToolTip(
            _translate("Parameters", "Check to remove binding times lasting less than the exposure time."))
        self.checkSingles.setText(_translate("Parameters", "Mask Singles"))
        self.groupBox_4.setTitle(_translate("Parameters", "Step 4"))
        self.checkOptimizationDisplay.setToolTip(
            _translate("Parameters", "Print optimization results when fitting exponentials."))
        self.checkOptimizationDisplay.setText(_translate("Parameters", "Optimization Display"))
        self.checkHyperExponential.setToolTip(
            _translate("Parameters", "Fit hyper exponential for a short and a long binding time (tau)."))
        self.checkHyperExponential.setText(_translate("Parameters", "Fit Hyper Exponential"))
        self.groupBox_6.setTitle(_translate("Parameters", "Initialization"))
        self.long_tauLabel.setToolTip(
            _translate("Parameters", "Initialization value of the long tau (binding time) in seconds."))
        self.long_tauLabel.setText(_translate("Parameters", "Long Tau (s):"))
        self.short_tauLabel.setToolTip(
            _translate("Parameters", "Initialization value of the short tau (binding time) in seconds."))
        self.short_tauLabel.setText(_translate("Parameters", "Short Tau (s):"))
        self.ratioLabel.setText(_translate("Parameters", "Ratio:"))
        self.rangeLabel.setToolTip(_translate("Parameters", "Range of the histogram in seconds."))
        self.rangeLabel.setText(_translate("Parameters", "Range:"))
        self.range.setText(_translate("Parameters", "0,13"))
        self.likelihoodErrorLabel.setToolTip(_translate("Parameters",
                                                        "This value is related to the likelihood error interval when estimating the error."))
        self.likelihoodErrorLabel.setText(_translate("Parameters", "Likelihood Error:"))
        self.groupBox.setTitle(_translate("Parameters", "Step 1"))
        self.checkRecursive.setToolTip(
            _translate("Parameters", "If checked runs all files inside the selected folder."))
        self.checkRecursive.setText(_translate("Parameters", "Recursive"))
        self.lpxLabel.setToolTip(_translate("Parameters", "Filter the localization precision in the x-direction."))
        self.lpxLabel.setText(_translate("Parameters", "Filter lpx:"))
        self.lpyLabel.setToolTip(_translate("Parameters", "Filter the localization precision in the y-direction."))
        self.lpyLabel.setText(_translate("Parameters", "Filter lpy:"))
        self.menuFile.setTitle(_translate("Parameters", "File"))
        self.actionOpen.setText(_translate("Parameters", "Open"))
        self.actionRun.setText(_translate("Parameters", "Run"))
        self.actionReset_Parameters.setText(_translate("Parameters", "Reset Parameters"))
        self.actionSave_Parameters.setText(_translate("Parameters", "Save Parameters"))
        self.actionOpen_2.setText(_translate("Parameters", "Open"))


    def get_parameters_from_ui(self):
        """
        Helper method to collect all parameters from UI controls.
        This centralizes parameter collection to avoid code duplication.
        """
        parameters = {
            # Step checkboxes
            "step1": self.checkStep1.isChecked(),
            "step2": self.checkStep2.isChecked(),
            "step3": self.checkStep3.isChecked(),
            "step4": self.checkStep4.isChecked(),
            "verbose": self.checkVerbose.isChecked(),
            "use_position_averaging": self.checkPositionAveraging.isChecked(),
            
            # Step 1 params
            "recursive": self.checkRecursive.isChecked(),
            "lpx_filter": self.lpx.value(),
            "lpy_filter": self.lpy.value(),
            
            # Step 2 params
            "exposure_time": self.exposure_time.value(),
            "docking_sites": self.docking_sites.value(),
            "checkNP": self.checkNP.isChecked(),
            "pixel_size": self.pixel_size.value(),
            "size_to_average": self.size_to_average.value(),
            "checkPlot": self.checkPlot.isChecked(),
            "use_dbscan": self.checkDBSCAN.isChecked(),
            
            # Step 3 params
            "mask_singles": self.checkSingles.isChecked(),
            "photon_threshold": self.photon_threshold.value(),
            "background_level": self.background_level.value(),
            "mask_level": int(self.mask.currentText()),  # Convert to int
            
            # Step 4 params
            "likelihood_error": self.likelihood_error.value(),
            "checkOptimizationDisplay": self.checkOptimizationDisplay.isChecked(),
            "checkHyperExponential": self.checkHyperExponential.isChecked(),
            
            # Other params
            "th": 1,
            "rectangle": False,  # Add rectangle parameter - will be overridden by YAML data in run()
        }
        
        return parameters
        
    def save_parameters(self):
        # Get parameters from UI
        self.parameters = self.get_parameters_from_ui()
        
        # Save to settings
        for parameter in self.parameters.keys():
            self.settings_parameters.setValue(parameter, self.parameters[parameter])
            
    def save_default_parameters(self):
        self.settings_default_parameters = QSettings('Parameter GUI', 'Default Parameters')
        
        # Get parameters from UI
        self.parameters = self.get_parameters_from_ui()
        
        # Save to default settings
        for parameter in self.parameters.keys():
            self.settings_default_parameters.setValue(parameter, self.parameters[parameter])
            
    def run(self):
        try:
            selected_file = QtWidgets.QFileDialog.getOpenFileName(filter=".hdf5 files (*.hdf5)")
            if selected_file != ('', ''):
                try:
                    # ----------------- GET INFO FROM YAML FILE -----------------
                    path_yaml = selected_file[0].replace('.hdf5', '.yaml')

                    # Check if file exists
                    if not os.path.exists(path_yaml):
                        print("Error: YAML file not found:", path_yaml)
                        return

                    # Load YAML documents from file
                    with open(path_yaml, 'r') as file:
                        docs = list(yaml.safe_load_all(file))

                    print(f"Debug: Processing YAML file: {path_yaml}")
                    print(f"Debug: Found {len(docs)} documents in YAML file")

                    # Robust extraction with error handling
                    try:
                        number_of_frames = docs[0].get('Frames')
                        if number_of_frames is None:
                            raise KeyError("Frames not found in first YAML document")
                    except (IndexError, KeyError) as e:
                        print(f"Error: Could not find 'Frames' in YAML file: {e}")
                        return

                    # Find pick information in any document
                    pick_shape = None
                    pick_diameter = None
                    for doc in docs:
                        if doc and 'Pick Shape' in doc:
                            pick_shape = doc.get('Pick Shape')
                            pick_diameter = doc.get('Pick Diameter')
                            break
                    
                    if pick_shape is None or pick_diameter is None:
                        print("Error: Pick Shape or Pick Diameter not found in YAML file")
                        print("Available documents in YAML:")
                        for i, doc in enumerate(docs):
                            if doc:
                                print(f"  Document {i}: {list(doc.keys())}")
                        return

                    # ----------------- GET INFO FROM GUI -----------------
                    path = selected_file[0]
                    working_folder = os.path.dirname(path)

                    steps_to_execute = [self.checkStep1.isChecked(), self.checkStep2.isChecked(), self.checkStep3.isChecked(), self.checkStep4.isChecked()]

                    initial_params = [self.long_tau.value(), self.short_tau.value(), self.ratio.value()]
                    range_list_string = list(self.range.text().split(','))
                    range_list_int = [int(s) for s in range_list_string]

                    # ----------------- CREATE PARAMETERS DICTIONARY -----------------
                    # Get base parameters from UI
                    parameters = self.get_parameters_from_ui()
                    
                    # Add file/path specific parameters
                    parameters.update({
                        "path": path,
                        "working_folder": working_folder,
                        "rectangle": pick_shape != 'Circle',  # Get rectangle value from YAML file
                        "number_of_frames": int(number_of_frames),
                        "pick_size": float(pick_diameter),
                        "range": range_list_int,
                        "initial_params": initial_params,
                    })

                    # Run the command in the terminal
                    analysis_thread = AnalysisThread(selected_file, working_folder, steps_to_execute, parameters)

                    # Start the thread
                    analysis_thread.start()
                    analysis_thread.join()
                except Exception as e:
                    print(f"An error occurred in running the analysis: {str(e)}")
        except:
            print('Choose a file.')

    #
    # def two_channels(self):
    #     try:
    #         selected_file = QtWidgets.QFileDialog.getOpenFileName(filter=".tif files (*.tif)")
    #         if selected_file != ('', ''):
    #             path = selected_file[0]
    #             two_channel_ui = TIFF_UI.Ui_Form()
    #             self.Form = QtWidgets.QWidget()
    #             two_channel_ui.setupUi(self.Form, movie_path=path)
    #             self.Form.show()
    #     except:
    #         print("An error occurred in analyzing two channels.")





if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    
    # Apply better styling for PyQt6
    app.setStyle('Fusion')  # Modern cross-platform style
    
    Parameters = QtWidgets.QMainWindow()
    ui = Ui_Parameters()
    ui.setupUi(Parameters)
    Parameters.show()
    sys.exit(app.exec())
