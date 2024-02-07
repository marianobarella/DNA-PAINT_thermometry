from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QIntValidator
import os
import threading
import subprocess


class AnalysisThread(threading.Thread):
    def __init__(self, selected_file, working_folder, steps_to_execute, params):
        super().__init__()
        self.selected_file = selected_file[0]
        self.working_folder = working_folder
        self.steps_to_execute = steps_to_execute
        self.params = params

    def run(self):
        params_str = str(self.params)
        command = (["python", "find_super_resolved_temp_clean.py", "--selected_file", self.selected_file, "--working_folder",
                    self.working_folder, "--step"] + [str(step) for step in self.steps_to_execute] + ["--params", params_str])
        result = subprocess.run(command)
        if result.returncode == 0:
            print("Subprocess finished successfully.")
        else:
            print(f"Command failed with return code {result.returncode}")


class Ui_Parameters(object):
    def setupUi(self, Parameters):
        Parameters.setObjectName("Parameters")
        Parameters.setWindowTitle('Parameters')
        Parameters.resize(200, 800)
        self.centralwidget = QtWidgets.QWidget(parent=Parameters)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_5 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_5.setWhatsThis("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
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
        self.horizontalLayout_7.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4.addWidget(self.groupBox_5)
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.checkRectangle = QtWidgets.QCheckBox(parent=self.groupBox)
        self.checkRectangle.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkRectangle.setObjectName("checkRectangle")
        self.verticalLayout_2.addWidget(self.checkRectangle)
        self.checkRecursive = QtWidgets.QCheckBox(parent=self.groupBox)
        self.checkRecursive.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkRecursive.setObjectName("checkRecursive")
        self.verticalLayout_2.addWidget(self.checkRecursive)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_4.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.checkNP = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.checkNP.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.checkNP.setIconSize(QtCore.QSize(17, 16))
        self.checkNP.setObjectName("checkNP")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.checkNP)
        self.checkPlot = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.checkPlot.setObjectName("checkPlot")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.checkPlot)
        self.numberOfFramesLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.numberOfFramesLabel.setObjectName("numberOfFramesLabel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.numberOfFramesLabel)
        self.number_of_frames = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.number_of_frames.setObjectName("number_of_frames")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.number_of_frames)
        self.exposureTimeLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.exposureTimeLabel.setObjectName("exposureTimeLabel")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.exposureTimeLabel)
        self.exposure_time = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.exposure_time.setObjectName("exposure_time")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.exposure_time)
        self.pixelSizeLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.pixelSizeLabel.setObjectName("pixelSizeLabel")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.pixelSizeLabel)
        self.pixel_size = QtWidgets.QLineEdit(parent=self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pixel_size.sizePolicy().hasHeightForWidth())
        self.pixel_size.setSizePolicy(sizePolicy)
        self.pixel_size.setAutoFillBackground(False)
        self.pixel_size.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.pixel_size.setObjectName("pixel_size")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.pixel_size)
        self.pickSizeLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.pickSizeLabel.setObjectName("pickSizeLabel")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.LabelRole, self.pickSizeLabel)
        self.pick_size = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.pick_size.setObjectName("pick_size")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.FieldRole, self.pick_size)
        self.sizeToAverageLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.sizeToAverageLabel.setObjectName("sizeToAverageLabel")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.LabelRole, self.sizeToAverageLabel)
        self.size_to_average = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.size_to_average.setObjectName("size_to_average")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.FieldRole, self.size_to_average)
        self.sitesLabel = QtWidgets.QLabel(parent=self.groupBox_2)
        self.sitesLabel.setObjectName("sitesLabel")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.ItemRole.LabelRole, self.sitesLabel)
        self.docking_sites = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.docking_sites.setObjectName("docking_sites")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.ItemRole.FieldRole, self.docking_sites)
        self.horizontalLayout.addLayout(self.formLayout)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.photonLabel = QtWidgets.QLabel(parent=self.groupBox_3)
        self.photonLabel.setObjectName("photonLabel")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.photonLabel)
        self.photon_threshold = QtWidgets.QLineEdit(parent=self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.photon_threshold.sizePolicy().hasHeightForWidth())
        self.photon_threshold.setSizePolicy(sizePolicy)
        self.photon_threshold.setAutoFillBackground(False)
        self.photon_threshold.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.photon_threshold.setObjectName("photon_threshold")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.photon_threshold)
        self.bkgLabel = QtWidgets.QLabel(parent=self.groupBox_3)
        self.bkgLabel.setObjectName("bkgLabel")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.bkgLabel)
        self.background_level = QtWidgets.QLineEdit(parent=self.groupBox_3)
        self.background_level.setObjectName("background_level")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.background_level)
        self.maskLabel = QtWidgets.QLabel(parent=self.groupBox_3)
        self.maskLabel.setObjectName("maskLabel")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.maskLabel)
        self.comboBox = QtWidgets.QComboBox(parent=self.groupBox_3)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.comboBox)
        self.horizontalLayout_3.addLayout(self.formLayout_2)
        self.verticalLayout_4.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.checkOptimizationDisplay = QtWidgets.QCheckBox(parent=self.groupBox_4)
        self.checkOptimizationDisplay.setIconSize(QtCore.QSize(17, 16))
        self.checkOptimizationDisplay.setChecked(False)
        self.checkOptimizationDisplay.setObjectName("checkOptimizationDisplay")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.checkOptimizationDisplay)
        self.checkHyperExponential = QtWidgets.QCheckBox(parent=self.groupBox_4)
        self.checkHyperExponential.setObjectName("checkHyperExponential")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.checkHyperExponential)
        self.rangeLabel = QtWidgets.QLabel(parent=self.groupBox_4)
        self.rangeLabel.setObjectName("rangeLabel")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.rangeLabel)
        self.range = QtWidgets.QLineEdit(parent=self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.range.sizePolicy().hasHeightForWidth())
        self.range.setSizePolicy(sizePolicy)
        self.range.setAutoFillBackground(False)
        self.range.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.range.setObjectName("range")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.range)
        self.likelihoodErrorLabel = QtWidgets.QLabel(parent=self.groupBox_4)
        self.likelihoodErrorLabel.setObjectName("likelihoodErrorLabel")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.ItemRole.LabelRole, self.likelihoodErrorLabel)
        self.likelihood_error = QtWidgets.QLineEdit(parent=self.groupBox_4)
        self.likelihood_error.setObjectName("likelihood_error")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.ItemRole.FieldRole, self.likelihood_error)
        self.groupBox_6 = QtWidgets.QGroupBox(parent=self.groupBox_4)
        self.groupBox_6.setObjectName("groupBox_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
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
        self.long_tau = QtWidgets.QLineEdit(parent=self.groupBox_6)
        self.long_tau.setObjectName("long_tau")
        self.verticalLayout_5.addWidget(self.long_tau)
        self.short_tau = QtWidgets.QLineEdit(parent=self.groupBox_6)
        self.short_tau.setObjectName("short_tau")
        self.verticalLayout_5.addWidget(self.short_tau)
        self.ratio = QtWidgets.QLineEdit(parent=self.groupBox_6)
        self.ratio.setObjectName("ratio")
        self.verticalLayout_5.addWidget(self.ratio)
        self.horizontalLayout_6.addLayout(self.verticalLayout_5)
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.ItemRole.SpanningRole, self.groupBox_6)
        self.horizontalLayout_4.addLayout(self.formLayout_3)
        self.verticalLayout_4.addWidget(self.groupBox_4)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.saveButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout_8.addWidget(self.saveButton)
        self.runButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.runButton.setObjectName("runButton")
        self.horizontalLayout_8.addWidget(self.runButton)
        self.verticalLayout_4.addLayout(self.horizontalLayout_8)
        self.verticalLayout_6.addLayout(self.verticalLayout_4)
        Parameters.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=Parameters)
        self.statusbar.setObjectName("statusbar")
        Parameters.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(parent=Parameters)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 329, 22))
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
        self.menuFile.addAction(self.actionReset_Parameters)
        self.actionSave_Parameters = QtGui.QAction(parent=Parameters)
        self.actionSave_Parameters.setObjectName("actionSave_Parameters")
        self.actionSave_Parameters.setText('Save Default Parameters')
        self.menuFile.addAction(self.actionSave_Parameters)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(Parameters)
        QtCore.QMetaObject.connectSlotsByName(Parameters)

        # DON'T REMOVE BELOW
        # The stored parameters.
        self.settings_parameters = QSettings('Parameter GUI', 'Parameters')
        self.number_of_frames.setValidator(QIntValidator())
        self.docking_sites.setValidator(QIntValidator())



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
                "recursive": lambda: self.checkRecursive.setChecked(value),
                "rectangle": lambda: self.checkRectangle.setChecked(value),
                "number_of_frames": lambda: self.number_of_frames.setText(value),
                "exposure_time": lambda: self.exposure_time.setText(value),
                "docking_sites": lambda: self.docking_sites.setText(value),
                "checkNP": lambda: self.checkNP.setChecked(value),
                "pixel_size": lambda: self.pixel_size.setText(value),
                "pick_size": lambda: self.pick_size.setText(value),
                "size_to_average": lambda: self.size_to_average.setText(value),
                "th": lambda: None,
                "checkPlot": lambda: self.checkPlot.setChecked(value),
                "photon_threshold": lambda: self.photon_threshold.setText(value),
                "background_level": lambda: self.background_level.setText(value),
                "mask_level": lambda: self.comboBox.setCurrentText(value),
                "likelihood_error": lambda: self.likelihood_error.setText(value),
                "checkOptimizationDisplay": lambda: self.checkOptimizationDisplay.setChecked(value),
                "checkHyperExponential": lambda: self.checkHyperExponential.setChecked(value)
            }
            if parameter in convert_dict:
                convert_dict[parameter]()

        for parameter in self.settings_parameters.allKeys():
            # self.parameters[parameter] = self.settings_parameters.value(parameter)
            convert_parameters_to_text(parameter, self.settings_parameters.value(parameter))


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

        def save_default_parameters():
            self.settings_default_parameters = QSettings('Parameter GUI', 'Default Parameters')
            self.parameters = {
                "step1": self.checkStep1.isChecked(),
                "step2": self.checkStep2.isChecked(),
                "step3": self.checkStep3.isChecked(),
                "step4": self.checkStep4.isChecked(),
                "recursive": self.checkRecursive.isChecked(),
                "rectangle": self.checkRectangle.isChecked(),
                "number_of_frames": int(self.number_of_frames.text()),
                "exposure_time": float(self.exposure_time.text()),
                "docking_sites": float(self.docking_sites.text()),
                "checkNP": self.checkNP.isChecked(),
                "pixel_size": float(self.pixel_size.text()),
                "pick_size": float(self.pick_size.text()),
                "size_to_average": float(self.size_to_average.text()),
                "th": 1,
                "checkPlot": self.checkPlot.isChecked(),
                "photon_threshold": float(self.photon_threshold.text()),
                "background_level": float(self.background_level.text()),
                "mask_level": float(self.comboBox.currentText()),
                "likelihood_error": float(self.likelihood_error.text()),
                "checkOptimizationDisplay": self.checkOptimizationDisplay.isChecked(),
                "checkHyperExponential": self.checkHyperExponential.isChecked()
            }
            for parameter in self.parameters.keys():
                self.settings_default_parameters.setValue(parameter, self.parameters[parameter])

        self.actionSave_Parameters.triggered.connect(lambda: save_default_parameters())
        def save_parameters():
            self.parameters = {
                "step1": self.checkStep1.isChecked(),
                "step2": self.checkStep2.isChecked(),
                "step3": self.checkStep3.isChecked(),
                "step4": self.checkStep4.isChecked(),
                "recursive": self.checkRecursive.isChecked(),
                "rectangle": self.checkRectangle.isChecked(),
                "number_of_frames": int(self.number_of_frames.text()),
                "exposure_time": float(self.exposure_time.text()),
                "docking_sites": float(self.docking_sites.text()),
                "checkNP": self.checkNP.isChecked(),
                "pixel_size": float(self.pixel_size.text()),
                "pick_size": float(self.pick_size.text()),
                "size_to_average": float(self.size_to_average.text()),
                "th": 1,
                "checkPlot": self.checkPlot.isChecked(),
                "photon_threshold": float(self.photon_threshold.text()),
                "background_level": float(self.background_level.text()),
                "mask_level": float(self.comboBox.currentText()),
                "likelihood_error": float(self.likelihood_error.text()),
                "checkOptimizationDisplay": self.checkOptimizationDisplay.isChecked(),
                "checkHyperExponential": self.checkHyperExponential.isChecked()
            }
            for parameter in self.parameters.keys():
                self.settings_parameters.setValue(parameter, self.parameters[parameter])
                #self.settings_parameters.


        self.saveButton.clicked.connect(lambda: save_parameters())

        # Initialize greyed out widgets.
        self.short_tau.setEnabled(self.checkHyperExponential.isChecked())
        self.short_tauLabel.setEnabled(self.checkHyperExponential.isChecked())

        self.ratio.setEnabled(self.checkHyperExponential.isChecked())
        self.ratioLabel.setEnabled(self.checkHyperExponential.isChecked())

        # When pressing run, run the run() function.
        self.runButton.clicked.connect(lambda: self.run())
        self.actionRun.triggered.connect(lambda: self.run())

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





    def retranslateUi(self, Parameters):
        _translate = QtCore.QCoreApplication.translate
        Parameters.setWindowTitle(_translate("Parameters", "MainWindow"))
        self.groupBox_5.setTitle(_translate("Parameters", "Steps to run"))
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
        self.groupBox.setTitle(_translate("Parameters", "Step 1"))
        self.checkRectangle.setToolTip(_translate("Parameters", "If unchecked the picks are assumed to be circles."))
        self.checkRectangle.setText(_translate("Parameters", "Rectangle"))
        self.checkRecursive.setToolTip(
            _translate("Parameters", "If checked runs all files inside the selected folder."))
        self.checkRecursive.setText(_translate("Parameters", "Recursive"))
        self.groupBox_2.setTitle(_translate("Parameters", "Step 2"))
        self.checkNP.setToolTip(_translate("Parameters", "Check if there are nanoparticles present in the picks."))
        self.checkNP.setText(_translate("Parameters", "Nanoparticle"))
        self.checkPlot.setToolTip(_translate("Parameters", "Check for plots."))
        self.checkPlot.setText(_translate("Parameters", "Plot"))
        self.numberOfFramesLabel.setToolTip(_translate("Parameters", "Number of frames in the data."))
        self.numberOfFramesLabel.setText(_translate("Parameters", "Number of Frames:"))
        self.number_of_frames.setText(_translate("Parameters", "36000"))
        self.exposureTimeLabel.setToolTip(_translate("Parameters", "Exposure time of the camera."))
        self.exposureTimeLabel.setText(_translate("Parameters", "Exposure Time (s):"))
        self.exposure_time.setText(_translate("Parameters", "0.1"))
        self.pixelSizeLabel.setToolTip(_translate("Parameters", "Pixel size of the camera."))
        self.pixelSizeLabel.setText(_translate("Parameters", "Pixel Size (um):"))
        self.pixel_size.setText(_translate("Parameters", "0.13"))
        self.pixel_size.setPlaceholderText(_translate("Parameters", "0"))
        self.pickSizeLabel.setToolTip(_translate("Parameters", "Pick size in camera pixels (pick size in Picasso)."))
        self.pickSizeLabel.setText(_translate("Parameters", "Pick Size (camera pixels):"))
        self.pick_size.setText(_translate("Parameters", "2"))
        self.sizeToAverageLabel.setToolTip(
            _translate("Parameters", "The size of the area to average around a docking site in camera pixels."))
        self.sizeToAverageLabel.setText(_translate("Parameters", "Averaging size (camera pixels):"))
        self.size_to_average.setText(_translate("Parameters", "0.2"))
        self.sitesLabel.setToolTip(_translate("Parameters", "How many docking sites are present on the DNA Origami."))
        self.sitesLabel.setText(_translate("Parameters", "Docking Sites:"))
        self.docking_sites.setText(_translate("Parameters", "3"))
        self.groupBox_3.setTitle(_translate("Parameters", "Step 3"))
        self.photonLabel.setToolTip(_translate("Parameters",
                                               "The least amount of photons registered in order to be considered a localization."))
        self.photonLabel.setText(_translate("Parameters", "Photon Threshold:"))
        self.photon_threshold.setText(_translate("Parameters", "300"))
        self.bkgLabel.setToolTip(_translate("Parameters", "The background level in photons."))
        self.bkgLabel.setText(_translate("Parameters", "Background Level (photons):"))
        self.background_level.setText(_translate("Parameters", "50"))
        self.maskLabel.setText(_translate("Parameters", "Mask Level:"))
        self.comboBox.setCurrentText(_translate("Parameters", "0"))
        self.comboBox.setItemText(0, _translate("Parameters", "0"))
        self.comboBox.setItemText(1, _translate("Parameters", "1"))
        self.comboBox.setItemText(2, _translate("Parameters", "2"))
        self.groupBox_4.setTitle(_translate("Parameters", "Step 4"))
        self.checkOptimizationDisplay.setToolTip(
            _translate("Parameters", "Print optimization results when fitting exponentials."))
        self.checkOptimizationDisplay.setText(_translate("Parameters", "Optimization Display"))
        self.checkHyperExponential.setToolTip(
            _translate("Parameters", "Fit hyper exponential for a short and a long binding time (tau)."))
        self.checkHyperExponential.setText(_translate("Parameters", "Fit Hyper Exponential"))
        self.rangeLabel.setText(_translate("Parameters", "Range:"))
        self.range.setText(_translate("Parameters", "0,13"))
        self.likelihoodErrorLabel.setText(_translate("Parameters", "Likelihood Error:"))
        self.likelihood_error.setText(_translate("Parameters", "2"))
        self.groupBox_6.setTitle(_translate("Parameters", "Initialization"))
        self.long_tauLabel.setToolTip(
            _translate("Parameters", "Initialization value of the long tau (binding time) in seconds."))
        self.long_tauLabel.setText(_translate("Parameters", "<html><head/><body><p>Long Tau (s):</p></body></html>"))
        self.short_tauLabel.setToolTip(
            _translate("Parameters", "Initialization value of the short tau (binding time) in seconds."))
        self.short_tauLabel.setText(_translate("Parameters", "Short Tau (s):"))
        self.ratioLabel.setText(_translate("Parameters", "Ratio:"))
        self.long_tau.setText(_translate("Parameters", "5"))
        self.short_tau.setText(_translate("Parameters", "0.1"))
        self.ratio.setText(_translate("Parameters", "1"))
        self.saveButton.setText(_translate("Parameters", "Save"))
        self.runButton.setText(_translate("Parameters", "Run"))
        self.menuFile.setTitle(_translate("Parameters", "File"))
        self.actionOpen.setText(_translate("Parameters", "Open"))
        self.actionRun.setText(_translate("Parameters", "Run"))
        self.actionReset_Parameters.setText(_translate("Parameters", "Reset Parameters"))




    def run(self):
        try:
            selected_file = QtWidgets.QFileDialog.getOpenFileName(filter=".hdf5 files (*.hdf5)")
            path = selected_file[0]
            working_folder = os.path.dirname(path)

            steps_to_execute = [self.checkStep1.isChecked(), self.checkStep2.isChecked(), self.checkStep3.isChecked(), self.checkStep4.isChecked()]

            initial_params = [float(self.long_tau.text()), float(self.short_tau.text()), float(self.ratio.text())]
            range_list_string = list(self.range.text().split(','))
            range_list_int = [int(s) for s in range_list_string]

            parameters = {
                "path": path,
                "working_folder": working_folder,
                "recursive": self.checkRecursive.isChecked(),
                "rectangle": self.checkRectangle.isChecked(),
                "number_of_frames": int(self.number_of_frames.text()),
                "exposure_time": float(self.exposure_time.text()),
                "docking_sites": float(self.docking_sites.text()),
                "checkNP": self.checkNP.isChecked(),
                "pixel_size": float(self.pixel_size.text()),
                "pick_size": float(self.pick_size.text()),
                "size_to_average": float(self.size_to_average.text()),
                "th": 1,
                "checkPlot": self.checkPlot.isChecked(),
                "photon_threshold": float(self.photon_threshold.text()),
                "background_level": float(self.background_level.text()),
                "mask_level": float(self.comboBox.currentText()),
                "range": range_list_int,
                "initial_params": initial_params,
                "likelihood_error": float(self.likelihood_error.text()),
                "checkOptimizationDisplay": self.checkOptimizationDisplay.isChecked(),
                "checkHyperExponential": self.checkHyperExponential.isChecked()
            }

            # Run the command in the terminal
            analysis_thread = AnalysisThread(selected_file, working_folder, steps_to_execute, parameters)

            # Start the thread
            analysis_thread.start()
            analysis_thread.join()
        except:
            print("An error occurred")




        # selected_file = QtWidgets.QFileDialog.getOpenFileName(filter=".hdf5 files (*.hdf5)")
        # path = selected_file[0]
        # working_folder = os.path.dirname(path)
        # print(working_folder)
        # if self.checkStep1.isChecked():
        #     step1_thread = threading.Thread(target=step1.split_hdf5, args=[path, working_folder, self.checkRecursive.isChecked(), self.checkRectangle.isChecked()])
        #     step1_thread.start()
        #     step1_thread.join()
        #     split_hdf5_command = ["python", "find_super_resolved_temp_clean.py", "--path", "/path/to/hdf5/file.h5",
        #                           "--working_folder", "/path/to/working/folder", "--recursive", "--rectangle"]
        #     result = subprocess.run(split_hdf5_command)
        #
        # if self.checkStep2.isChecked():
        #     step2_working_folder = os.path.join(working_folder, 'split_data')
        #     step2_thread = threading.Thread(target=step2.process_dat_files, args=[int(self.number_of_frames.text()), float(self.exposure_time.text()), step2_working_folder,
        #                             float(self.docking_sites.text()), self.checkNP.isChecked(), float(self.pixel_size.text()), float(self.pick_size.text()), \
        #                             float(self.size_to_average.text()), 1, self.checkPlot.isChecked()])
        #     step2_thread.start()
        #     step2_thread.join()
        #
        #
        # if self.checkStep3.isChecked():
        #
        #
        #     step3_thread = threading.Thread(target=step3.calculate_kinetics, args=[float(self.exposure_time.text()), float(self.photon_threshold.text()),
        #                             float(self.background_level.text()), step3_working_folder, all_traces_filename, float(self.mask_level.text())])
        #     step3_thread.start()
        #     step3_thread.join()
        #     pass
        #
        # if self.checkStep4.isChecked():
        #     step2_working_folder = os.path.join(working_folder, 'split_data')
        #     step4_working_folder = os.path.join(step2_working_folder, 'kinetics_data')
        #     initial_params = [float(self.long_tau.text()), float(self.short_tau.text()), float(self.ratio.text())]
        #     range_list_string = list(self.range.text().split(','))
        #     range_list_int = [int(s) for s in range_list_string]
        #     step4_thread = threading.Thread(target=step4.estimate_binding_unbinding_times, args=[float(self.exposure_time.text()), range_list_int, step4_working_folder,
        #                                            initial_params, float(self.likelihood_error.text()),
        #                                            self.checkOptimizationDisplay.isChecked(), self.checkHyperExponential.isChecked()])
        #     step4_thread.start()
        #     step4_thread.join()
        #     pass



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Parameters = QtWidgets.QMainWindow()
    ui = Ui_Parameters()
    ui.setupUi(Parameters)
    Parameters.show()
    sys.exit(app.exec())
