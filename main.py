from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from PyQt5 import QtMultimedia as qtmm

import subprocess, os, wexpect, time, random, psutil, zmq, win32api, shutil, configparser

import datetime as dt
from datetime import datetime
import pandas as pd

import numpy as np

from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.pyplot as plt
import logging


def plot_distance(title:str, data: pd.DataFrame, tone: float, filename:str) -> None:
    sns.set(rc={'figure.figsize': (11, 4)})
    plt.figure(figsize=(12,4))
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.grid(color='#F2F2F2', alpha=1, zorder=0)
    plt.xlabel('Time (s)', fontsize=13)
    plt.ylabel('Distance (mm)', fontsize=13)
    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.title(title + " - Distance from Leap origin", fontsize=17)
    # plt.ylim([0, 30])
    plt.plot(data.timestamp, data.palm_position_x, label='x direction')
    plt.plot(data.timestamp, data.palm_position_y, label='y direction')
    plt.plot(data.timestamp, data.palm_position_z, label='z direction')
    plt.axvline(tone, alpha=0.5, color='tab:orange', label='Auditory tone', dashes=(5, 2, 1, 2))

    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def read_data(filename):
    polhemus_df = pd.read_csv(filename, lineterminator='\r')

    polhemus_df = polhemus_df.apply(lambda x: savgol_filter(x, 33, 3))

    starting_timestamp = polhemus_df['timestamp'].min()

    polhemus_df['timestamp'] = (polhemus_df['timestamp'] - starting_timestamp) / 1000000


    polhemus_df[['x', 'y', 'z']] = polhemus_df[['x', 'y', 'z']].apply(lambda x: x * 2.54)  # inches to cm
    polhemus_df['distance'] = polhemus_df[['x', 'y', 'z']].apply(np.linalg.norm, axis=1)   # euclidean distance

    d_time = polhemus_df['timestamp'].diff().fillna(0.)
    d_distance = polhemus_df['distance'].diff().fillna(0.)

    polhemus_df['velocity'] = d_distance / d_time
    polhemus_df['acceleration'] = polhemus_df['velocity'].diff() / d_time

    return polhemus_df, starting_timestamp

def get_drives():
    """
    This function returns a list of drives that are suitable for storing the data files. They must have enough free
    space.
    """

    # Todo Error log at the end

    drives_list = []
    error_list = []

    # Save
    local_drive_letter = win32api.GetWindowsDirectory().split("\\")[0] + "\\"

    # Get names of attached drives
    drives = win32api.GetLogicalDriveStrings()
    drives = drives.split('\000')[:-1]
    drives.remove(local_drive_letter)

    # 1. Save on the local drive if there is space
    _, _, local_free = shutil.disk_usage(local_drive_letter)

    if (local_free / 1073741824) < 2:
        error_list.append('Local disk is low on space')
    else:
        drives_list.append(local_drive_letter)

    # 2. Save in external drive, prioritize one named clearsky. Else get one with the largest space
    clearsky_drive = False

    for drive in drives:
        if win32api.GetVolumeInformation(drive)[0].upper() in ["CLEARSKY", "CLEAR_SKY", "CLEAR SKY"]:
            clearsky_drive = drive

    if clearsky_drive != False:
        # check that the clearsky drive has enough space
        _, _, clearsky_drive_free = shutil.disk_usage(clearsky_drive)

        if (clearsky_drive_free / 1073741824) < 2:
            error_list.append("Clear Sky Drive is low on space")
            clearsky_drive = False
        else:
            drives_list.append(clearsky_drive)
    else:
        error_list.append("No drive named ClearSky.")

    # If clearsky drive has not been found or does not have enough free space

    if clearsky_drive == False:
        drive_space = {}
        for drive in drives:
            _, _, free = shutil.disk_usage(drive)

            if (free / 1073741824) >= 2:
                drive_space[drive] = free

        if len(drive_space) != 0:
            external_drive = max(drive_space, key=drive_space.get)
            drives_list.append(external_drive)
        else:
            error_list.append("No external drive found or did not have enough space")

    error_message = '\n'.join(error_list)

    return drives_list, error_message

class TrialWizard(qtw.QWizard):
    Page_TrialInfo = 1
    Page_Demographic = 2
    Page_Recording = 3

    devices_ready = qtc.pyqtSignal(dict)

    start_leap = qtc.pyqtSignal()
    stop_leap = qtc.pyqtSignal()  # No reason to stop app yet
    get_leap_status = qtc.pyqtSignal()

    start_polhemus = qtc.pyqtSignal()
    stop_polhemus = qtc.pyqtSignal()
    get_polhemus_status = qtc.pyqtSignal()

    get_pupil_status = qtc.pyqtSignal()

    def __init__(self, parent=None):
        super(TrialWizard, self).__init__(parent)

        self.setWindowTitle("Reach&Grasp")

        self.local_drive = None
        self.external_drive = None
        self.folder_name = None

        self.drives = []

        self.device_config = {"leap": False,
                               "polhemus": False,
                               "pupil": False,
                               "Allow changes": False}

        self.leap_ready = False
        self.pupil_ready = False
        self.polhemus_ready = False

        self.leap_path = None
        self.polhemus_path = None

        # Read configuration file
        self.read_config()

        # Check that there is space available to store the data.

        self.error_msg = qtw.QMessageBox()
        self.error_msg.setIcon(qtw.QMessageBox.Critical)
        self.error_msg.setWindowTitle("Error")
        self.error_dialog = qtw.QErrorMessage()
        self.update_drives()

        ### Configuring the pages ###

        self.setPage(self.Page_Recording, RecordingPage(self))
        self.setPage(self.Page_Demographic, DemographicPage(self))
        self.setPage(self.Page_TrialInfo, TrialInfoPage(self, self.device_config))

        self.setStartId(self.Page_TrialInfo)

        self.setWizardStyle(qtw.QWizard.ModernStyle)
        self.setPixmap(qtw.QWizard.LogoPixmap, qtg.QPixmap('images/clearsky-logo.png'))

        self.setOption(qtw.QWizard.IndependentPages, True)

        self.startingTime = time.strftime("%Y%m%d-%H%M%S")

        ### LeapRecorder program ###
        if self.device_config['leap']:
            self.leap_handler = LeapHandler(self.leap_path)

            # Using a thread, needs to be done before we connect signals or slots
            self.leap_thread = qtc.QThread()
            self.leap_handler.moveToThread(self.leap_thread)
            self.leap_thread.start()
            logging.info("Leap Handler moved to thread")

        if self.device_config['polhemus']:

            self.polhemus_handler = PolhemusHandler(self.polhemus_path)

            self.polhemus_thread = qtc.QThread()
            self.polhemus_handler.moveToThread(self.polhemus_thread)
            self.polhemus_thread.start()
            logging.info("Polhemus Handler moved to thread")

            # Polhemus Connections


        if self.device_config['leap']:

            # LeapMotion connections

            self.start_leap.connect(self.leap_handler.start_application)

            self.get_leap_status.connect(self.leap_handler.get_status)
            self.leap_handler.device_status.connect(self.leap_update)

            self.page(self.Page_Recording).leapStartRecording.connect(self.leap_handler.start_recording)
            self.page(self.Page_Recording).leapStopRecording.connect(self.leap_handler.stop_recording)

            self.leap_handler.timestamps.connect(self.page(self.Page_Recording).leapTimestamps)

        if self.device_config['polhemus']:

            self.start_polhemus.connect(self.polhemus_handler.start_application)

            self.get_polhemus_status.connect(self.polhemus_handler.get_status)
            self.polhemus_handler.device_status.connect(self.polhemus_update)

            self.page(self.Page_Recording).polhemusStartRecording.connect(self.polhemus_handler.start_recording)
            self.page(self.Page_Recording).polhemusStopRecording.connect(self.polhemus_handler.stop_recording)


        if self.device_config['pupil']:
            self.pupil_handler = PupilHandler()

            # Pupil connections
            self.page(self.Page_Recording).pupilStartCalibration.connect(self.pupil_handler.startCalibration)
            self.page(self.Page_Recording).pupilStopCalibration.connect(self.pupil_handler.stopCalibration)

            self.page(self.Page_Recording).pupilStartRecording.connect(self.pupil_handler.startRecording)
            self.page(self.Page_Recording).pupilStopRecording.connect(self.pupil_handler.stopRecording)

            self.get_pupil_status.connect(self.pupil_handler.get_status)
            self.pupil_handler.device_status.connect(self.pupil_update)

        logging.info("Setup devices")
        # Recording Page

        self.devices_ready.connect(self.page(self.Page_Recording).devicesReady)

        # Initial poll
        self.devicesReady()

        # Then update using an interval timer
        interval_seconds = 5
        self.status_timer = qtc.QTimer()
        self.status_timer.setInterval(interval_seconds * 1000)
        self.status_timer.timeout.connect(self.devicesReady)
        self.status_timer.start()

        logging.info("Here")

    def read_config(self):

        # Todo check if config file exists

        if os.path.exists("config.ini"):

            try:

                logging.info("reading config file")
                config = configparser.ConfigParser()
                config.read("config.ini")

                self.local_drive = config['PATH']['Local']
                self.external_drive = config['PATH']['External']
                self.folder_name = config['PATH']['Folder name']
                # self.leap_path = config['PATH']['leap path']
                # self.polhemus_path = config['PATH']['polhemus path']

                self.leap_path = r"C:\Users\Cam\Desktop\Desktop backup\wexpect-master\wexpect-master\dist\Leap\LeapAquisition.exe"
                self.polhemus_path = r"C:\Users\Cam\Desktop\Desktop backup\wexpect-master\wexpect-master\dist\Polhemus\PDIconsole.exe"

                self.device_config['leap'] = (config['DEVICES']['leap'] == "True")
                self.device_config['polhemus'] = (config['DEVICES']['polhemus'] == "True")
                self.device_config['pupil'] = (config['DEVICES']['pupil'] == "True")
                self.device_config['Allow changes'] = (config['DEVICES']['allow changes'] == "True")

            except :
                logging.error("Error reading congfiguration file")

            else:
                logging.info("Successfully read configuration file")

        else:
            logging.error("Config file could not be found")


        logging.info(str(self.device_config))

    def update_drives(self):
        """
        Deals with the errors generated by the get_drives function. If there is no space available at all then the
        program should quit. Otherwise show the messages so that they can be corrected.
        """
        drives_list = []
        error_list = []

        _, _, local_free = shutil.disk_usage(self.local_drive)

        if (local_free / 1073741824) < 2:
            error_list.append('Local disk is low on space')
        else:
            drives_list.append(self.local_drive)

        _, _, external_free = shutil.disk_usage(self.external_drive)

        if (external_free / 1073741824) < 2:
            error_list.append('External disk is low on space')
        else:
            drives_list.append(self.external_drive)

        error_message = '\n'.join(error_list)

        if bool(error_message) == True:

            if len(drives_list) == 0:  # No drives available
                self.error_msg.finished.connect(qtw.qApp.quit)
                self.error_msg.setText("No drives Available")
                self.error_msg.show()
            else:

                error_message += "\n\nWould you like to continue?"
                reply = qtw.QMessageBox.question(self, 'Message', error_message, qtw.QMessageBox.Yes,
                                                 qtw.QMessageBox.No)

                if reply == qtw.QMessageBox.No:
                    qtw.qApp.quit

        self.drives = drives_list

    def devicesReady(self):

        if self.field("intro.leap"):
            self.get_leap_status.emit()
        else:
            self.leap_ready = False

        if self.field("intro.polhemus"):
            self.get_polhemus_status.emit()
        else:
            self.polhemus_ready = False

        if self.field("intro.pupil"):
            self.get_pupil_status.emit()
        else:
            self.pupil_ready = False

        devices = {'leap': self.leap_ready, 'pupil': self.pupil_ready, 'polhemus': self.polhemus_ready}
        self.devices_ready.emit(devices)

    @qtc.pyqtSlot(dict)
    def leap_update(self, update):
        # The program will send an update dictionary.

        if not update['app_active']:
            logging.error("Leap not active, restarting")
            self.leap_ready = False
            self.start_leap.emit()
        elif update['device_connected'] < 1:
            self.leap_ready = False
        else:
            self.leap_ready = True

    @qtc.pyqtSlot(dict)
    def polhemus_update(self, update):
        # The program will send an update dictionary.

        if not update['app_active']:
            logging.error("Polhemus not active, restarting")

            self.polhemus_ready = False
            self.start_polhemus.emit()
        else:
            self.polhemus_ready = True

    @qtc.pyqtSlot(dict)
    def pupil_update(self, update):
        capture_open = update['capture_open']
        calibrated = update['calibrated']

        if capture_open and calibrated:
            self.pupil_ready = True
        else:
            self.pupil_ready = False

    def nextId(self):
        id = self.currentId()

        if id == self.Page_Recording:
            self.page(self.Page_Recording).updatePage()

        if id == self.Page_TrialInfo:
            return self.Page_Demographic
        if id == self.Page_Demographic:
            return self.Page_Recording
        else:
            return -1

    def accept(self):

        participantId = self.field('intro.participantID')
        filename = self.startingTime + "_" + participantId

        path = self.folder_name + "\\" + filename + "\\"

        for drive in self.drives:
            os.makedirs(drive + path, exist_ok=True)

        # Update csv

        for drive in self.drives:
            csv_path = drive + self.folder_name + "\\"  + "participant_record.csv"
            if not os.path.exists(csv_path):
                participant_record = pd.DataFrame({"ID":[participantId], "starting time":[self.startingTime]})
                participant_record.to_csv(csv_path)
            else:
                new = pd.DataFrame({"ID":[participantId], "starting time":[self.startingTime]})
                participant_record_df = pd.read_csv(csv_path, index_col=0)
                participant_record_df = participant_record_df.append(new, ignore_index=True)
                participant_record_df.to_csv(csv_path)

        # MoCA

        intro = {'group': self.field('intro.group'),
                 'handedness': ('intro.handedness'),                                         
                 'moca': self.field('intro.moca'),
                 'srds': self.field('intro.srds'),
                 'acuity': self.field('intro.acuity'),
                 'acuity': self.field('intro.notes')}

        intro_df = pd.DataFrame.from_dict(intro, orient='index')

        for drive in self.drives:
            intro_df.to_csv(drive + path + filename + '_intro.csv', header=None)

        # Demographic

        demographic = {'age': self.field('demographic.age'),
                       'sex': self.field('demographic.sex'),
                       'education': self.field('demographic.education')}

        demographic_df = pd.DataFrame.from_dict(demographic, orient='index')

        for drive in self.drives:
            demographic_df.to_csv(drive + path + filename + '_demographics.csv', header=None)

        # Timestamps

        tone_timestamps_df = pd.DataFrame.from_dict(self.page(self.Page_Recording).tone_timestamps, orient='index')
        # tone_timestamps_df.columns = ["recording", "timestamp"]

        for drive in self.drives:
            tone_timestamps_df.to_csv(drive + path + filename + '_tone_timestamps.csv', header=None)

        leap_timestamps_df = pd.DataFrame.from_dict(self.page(self.Page_Recording).leap_timestamps, orient='index')

        for drive in self.drives:
            leap_timestamps_df.to_csv(drive + path + filename + '_leap_timestamps.csv', header=None)

        # Leap motion

        try:
            shutil.rmtree("report")
            os.mkdir("report")
        except FileNotFoundError:
            os.mkdir("report")

        timestamps = self.page(self.Page_Recording).tone_timestamps

        leap_timestamps = self.page(self.Page_Recording).leap_timestamps


        for recording in self.page(self.Page_Recording).leap_recordings:

            if os.path.exists(recording):

                leap_df = pd.read_csv(recording)

                for drive in self.drives:
                    shutil.copyfile(recording, drive + path + filename + '_' + recording.split('\\')[-1])

                while (os.path.exists(recording)):
                    try:
                        os.remove(recording)
                    except:
                        logging.warning("Unable to remove {}".format(recording))

                try:
                    name = recording.split('\\')[-1]
                    name = name.split('.')[0]
                    name = name.replace('_leap', "")

                    auditory_tone_obj = timestamps[name]
                    auditory_tone_seconds = (auditory_tone_obj - datetime(1970, 1, 1)).total_seconds()

                    leap, system = leap_timestamps[name]

                    auditory_tone_seconds = auditory_tone_seconds - float(system)

                    leap_seconds = int(leap) / (10 ** 6)

                    leap_df['timestamp'] = (leap_df['timestamp'] / (10 ** 6)) - leap_seconds

                    leap_min = leap_df['timestamp'].min()

                    if leap_min > 0:
                        leap_df['timestamp'] = leap_df['timestamp'] - leap_min
                        auditory_tone_seconds = auditory_tone_seconds - leap_min

                    leap_df['distance'] = leap_df[['palm_position_x', 'palm_position_y', 'palm_position_z']].apply(
                        np.linalg.norm, axis=1)

                    new_filename = "report/" + name

                    plot_distance(name, leap_df, auditory_tone_seconds, new_filename)
                except:
                    logging.error("Could not create graph for {}".format(recording))
                else:
                    logging.info("Created graph for {}".format(recording))

        for drive in self.drives:
            shutil.copytree("report/", drive + path + "report\\")

        shutil.rmtree("report/")

        # Pupil Core

        for recording in self.page(self.Page_Recording).pupil_recordings:

            if os.path.exists(recording):

                logging.info("Found recording {}".format(recording))

                for drive in self.drives:
                    shutil.copytree(recording,
                                    drive + path + filename + '_' + recording.split('\\')[-1])
                    logging.info("Copying {} to drive {}".format(recording, drive))

                try:
                    shutil.rmtree(recording)
                except:
                    logging.warning("Unable to delete recording: {}".format(recording))
            else:
                logging.warning("{} recording not found".format(recording))

        logging.info("Successfully saved all recordings")
        super(TrialWizard, self).accept()


class TrialInfoPage(qtw.QWizardPage):
    drives = []

    def __init__(self, parent, device_config):
        super(TrialInfoPage, self).__init__(parent)

        self.setTitle("  ")
        self.setSubTitle("  ")

        participantIdLineEdit = qtw.QLineEdit()

        groupComboBox = qtw.QComboBox()
        groupComboBox.addItem('Alzheimer\'s Disease', 'Alzheimer\'s Disease')
        groupComboBox.addItem('Parkinson\'s Disease', 'Parkinson\'s Disease')
        groupComboBox.addItem('Vascular Dementia', 'Vascular Dementia')
        groupComboBox.addItem('MCI', 'MCI')
        groupComboBox.addItem('Control', 'Control')
        groupComboBox.addItem('Other', 'Other')

        handednessComboBox = qtw.QComboBox()
        handednessComboBox.addItem('Right', 'Right')
        handednessComboBox.addItem('Left', 'Left')

        mocaSpinBox = qtw.QSpinBox(maximum=30)
        srds = qtw.QSpinBox(maximum=80)

        leapCheckBox = qtw.QCheckBox()

        leapCheckBox.setChecked(device_config['leap'])

        polhemusCheckBox = qtw.QCheckBox()
        polhemusCheckBox.setChecked(device_config['polhemus'])

        pupilCoreCheckBox = qtw.QCheckBox()
        pupilCoreCheckBox.setChecked(device_config['pupil'])
        pupilCoreCheckBox.toggled.connect(parent.page(parent.Page_Recording).updatePage)

        devicesLayout = qtw.QFormLayout()
        devicesLayout.addRow("Leap Motion", leapCheckBox)
        devicesLayout.addRow("Polhemus", polhemusCheckBox)
        devicesLayout.addRow("Pupil Core", pupilCoreCheckBox)

        devicesGroupBox = qtw.QGroupBox("Device configuration")
        devicesGroupBox.setLayout(devicesLayout)

        if device_config['Allow changes'] == False:
            logging.info("Devices Hidden")
            devicesGroupBox.setHidden(True)

        acuityLineEdit = qtw.QLineEdit()
        acuityLineEdit.setValidator(qtg.QRegExpValidator(qtc.QRegExp("[-]*[0-9]*[.][0-9]*")))

        notesPlainTextEdit = qtw.QPlainTextEdit()

        self.registerField('intro.participantID*', participantIdLineEdit)
        self.registerField('intro.group', groupComboBox, "currentText")
        self.registerField('intro.handedness', handednessComboBox, "currentText")
        self.registerField('intro.moca', mocaSpinBox)
        self.registerField('intro.srds', srds)
        self.registerField('intro.leap', leapCheckBox)
        self.registerField('intro.polhemus', polhemusCheckBox)
        self.registerField('intro.pupil', pupilCoreCheckBox)
        self.registerField('intro.acuity', acuityLineEdit)
        self.registerField('intro.notes', notesPlainTextEdit, "plainText")


        layout = qtw.QFormLayout()
        layout.setVerticalSpacing(15)
        layout.addRow("Participant ID", participantIdLineEdit)
        layout.addRow("Group", groupComboBox)
        layout.addRow("Handedness", handednessComboBox)
        layout.addRow("MoCA Total Score", mocaSpinBox)
        layout.addRow("SRDS Total Score", srds)
        layout.addRow("Visual Acuity Score (If applicable)", acuityLineEdit)
        layout.addRow("Notes", notesPlainTextEdit)

        widgetLayout = qtw.QVBoxLayout()

        layoutGroupBox = qtw.QGroupBox()
        layoutGroupBox.setLayout(layout)
        widgetLayout.addWidget(devicesGroupBox)
        widgetLayout.addWidget(layoutGroupBox)
        self.setLayout(widgetLayout)


class DemographicPage(qtw.QWizardPage):
    def __init__(self, parent=None):
        super(DemographicPage, self).__init__(parent)

        self.setTitle("  ")
        self.setSubTitle("  ")

        ageLineEdit = qtw.QLineEdit()
        ageLineEdit.setValidator(qtg.QRegExpValidator(qtc.QRegExp("[0-9]*[.]*[0-9]*")))

        sexComboBox = qtw.QComboBox()
        sexComboBox.addItem('M', 'M')
        sexComboBox.addItem('F', 'F')

        educationLineEdit = qtw.QLineEdit()
        educationLineEdit.setValidator(qtg.QRegExpValidator(qtc.QRegExp("[0-9]*[.][0-9]*")))

        self.registerField('demographic.age', ageLineEdit)
        self.registerField('demographic.sex', sexComboBox, "currentText")
        self.registerField('demographic.education', educationLineEdit)

        layout = qtw.QFormLayout()
        layout.addRow("Age", ageLineEdit)
        layout.addRow("Sex", sexComboBox)
        layout.addRow("Years of education", educationLineEdit)
        self.setLayout(layout)


class RecordingPage(qtw.QWizardPage):

    leap_recordings = []
    leap_timestamps = {}
    tone_timestamps = {}
    pupil_recordings = []
    polhemus_recordings = []

    leapStartRecording = qtc.pyqtSignal(str)
    polhemusStartRecording = qtc.pyqtSignal(str)
    pupilStartRecording = qtc.pyqtSignal(str)

    leapStopRecording = qtc.pyqtSignal()
    polhemusStopRecording = qtc.pyqtSignal()
    pupilStopRecording = qtc.pyqtSignal()

    pupilStartCalibration = qtc.pyqtSignal()
    pupilStopCalibration = qtc.pyqtSignal()

    def __init__(self, parent=None):
        super(RecordingPage, self).__init__(parent)

        self.leap_ready = False
        self.polhemus_ready = False
        self.pupil_ready = False
        self.pupil_calibrated = False

        self.num_vis = 0
        self.num_mem = 0

        self.current_task = None

        self.setTitle("  ")
        self.setSubTitle("   ")

        self.tone_player = qtmm.QSoundEffect()
        self.tone_player.setSource(qtc.QUrl.fromLocalFile("tone.wav"))

        self.error_dialog = qtw.QErrorMessage()

        self.leapTextLabel = qtw.QLabel("Leap Motion")
        self.leapImageLabel = qtw.QLabel()

        self.leapDeviceLayout = qtw.QHBoxLayout()
        self.leapDeviceLayout.addWidget(self.leapTextLabel)
        self.leapDeviceLayout.addWidget(self.leapImageLabel)

        self.polhemusTextLabel = qtw.QLabel("Polhemus")
        self.polhemusImageLabel = qtw.QLabel()

        self.polhemusDeviceLayout = qtw.QHBoxLayout()
        self.polhemusDeviceLayout.addWidget(self.polhemusTextLabel)
        self.polhemusDeviceLayout.addWidget(self.polhemusImageLabel)

        self.pupilTextLabel = qtw.QLabel("Pupil")
        self.pupilImageLabel = qtw.QLabel()

        self.pupilDeviceLayout = qtw.QHBoxLayout()
        self.pupilDeviceLayout.addWidget(self.pupilTextLabel)
        self.pupilDeviceLayout.addWidget(self.pupilImageLabel)

        devicesLayout = qtw.QVBoxLayout()
        devicesLayout.addLayout(self.leapDeviceLayout)
        devicesLayout.addLayout(self.polhemusDeviceLayout)
        devicesLayout.addLayout(self.pupilDeviceLayout)

        devicesGroupBox = qtw.QGroupBox("Devices Ready")
        devicesGroupBox.setLayout(devicesLayout)

        step_oneLabel = qtw.QLabel("Make sure that the following programs are open. Ensure that the participants hand "
                                   "is being tracked with the Leap Motion. ")

        leapVisualizerButton = qtw.QPushButton("Open Leap Visualizer")
        leapVisualizerButton.clicked.connect(self.openleapVisualizer)

        self.pupilCaptureButton = qtw.QPushButton("Open PupilCapture")
        self.pupilCaptureButton.clicked.connect(self.openPupilCapture)
        self.pupilCalibrationButton = qtw.QPushButton("Start Calibration", checkable=True)

        step_oneLayout = qtw.QVBoxLayout()
        step_oneLayout.addWidget(step_oneLabel)
        step_oneLayout.addWidget(leapVisualizerButton)
        step_oneLayout.addWidget(self.pupilCaptureButton)
        step_oneLayout.addWidget(self.pupilCalibrationButton)

        step_oneGroupBox = qtw.QGroupBox("Step 1:")
        step_oneGroupBox.setLayout(step_oneLayout)

        step_one_aLabel = qtw.QLabel("Once the Pupil Capture software is open. Calibrate using the single Marker")

        self.pupilCalibrationButton.clicked.connect(self.calibrationButtonToggle)

        step_one_aLayout = qtw.QVBoxLayout()
        step_one_aLayout.addWidget(step_one_aLabel)
        step_one_aLayout.addWidget(self.pupilCalibrationButton)

        self.step_one_aGroupBox = qtw.QGroupBox("Step 1a:")
        self.step_one_aGroupBox.setLayout(step_one_aLayout)

        testPushButton = qtw.QPushButton("Test Tone")
        testPushButton.clicked.connect(self.tone_player.play)

        step_twoLayout = qtw.QVBoxLayout()

        step_twoLabel = qtw.QLabel("This is the tone that will initiate the reach sequence. The participant will reach"
                                   " immediately when hearing the tone.")
        step_twoLayout.addWidget(step_twoLabel)
        step_twoLayout.addWidget(testPushButton)

        step_twoGroupBox = qtw.QGroupBox("Step 2:")
        step_twoGroupBox.setLayout(step_twoLayout)

        # Radio buttons to select which test is being recorded. These recordings will be added to the arrays. Then the
        # user will be able to add or delete one.

        step_threeLabel = qtw.QLabel("Select which test is going to be recorded, the number next to them will indicate"
                                     " the number of recordings that have been made so far.")

        self.reach_graspRadioButton = qtw.QRadioButton("Reach and grasp")
        self.reach_pointRadioButton = qtw.QRadioButton("Reach and point")

        self.registerField('rec.grasp', self.reach_graspRadioButton)
        self.registerField('rec.point', self.reach_pointRadioButton)

        self.reach_graspRadioButton.setChecked(True)

        reach_buttonLayout = qtw.QHBoxLayout()
        reach_buttonLayout.addWidget(self.reach_graspRadioButton)
        reach_buttonLayout.addWidget(self.reach_pointRadioButton)

        self.visuallyGuidedRadioButton = qtw.QRadioButton("Visually guided : {}".format(self.num_vis))
        self.memoryGuidedRadioButton = qtw.QRadioButton("Memory guided : {}".format(self.num_mem))

        self.visuallyGuidedRadioButton.setChecked(True)

        self.recordingButton = qtw.QPushButton("Start Recording", checkable=True)
        self.recordingButton.clicked.connect(self.recordingButtonToggle)

        self.status_label = qtw.QLabel("Ready")

        self.registerField('rec.visually_guided', self.visuallyGuidedRadioButton)
        self.registerField('rec.memory_guided', self.memoryGuidedRadioButton)

        buttonLayout = qtw.QHBoxLayout()
        buttonLayout.addWidget(self.visuallyGuidedRadioButton)
        buttonLayout.addWidget(self.memoryGuidedRadioButton)

        step_threeGroupBoxLayout = qtw.QVBoxLayout()
        step_threeGroupBoxLayout.addWidget(step_threeLabel)
        #step_threeGroupBoxLayout.addLayout(reach_buttonLayout)
        step_threeGroupBoxLayout.addLayout(buttonLayout)
        step_threeGroupBoxLayout.addWidget(self.recordingButton)
        step_threeGroupBoxLayout.addWidget(self.status_label)

        step_threeGroupBox = qtw.QGroupBox("Step 3:")
        step_threeGroupBox.setLayout(step_threeGroupBoxLayout)

        # LAYOUT
        layout = qtw.QVBoxLayout()
        layout.addWidget(devicesGroupBox)
        layout.addWidget(step_oneGroupBox)
        layout.addWidget(self.step_one_aGroupBox)
        layout.addWidget(step_twoGroupBox)
        layout.addWidget(step_threeGroupBox)
        self.setLayout(layout)

        self.updatePage()

    @qtc.pyqtSlot(dict)
    def devicesReady(self, devices):
        self.leap_ready = devices['leap']
        self.polhemus_ready = devices['polhemus']
        self.pupil_ready = devices['pupil']

        tick = qtg.QPixmap("images/tick.png")
        cross = qtg.QPixmap("images/cross.png")

        if self.leap_ready:
            self.leapImageLabel.setPixmap(tick)
        else:
            self.leapImageLabel.setPixmap(cross)

        if self.polhemus_ready:
            self.polhemusImageLabel.setPixmap(tick)
        else:
            self.polhemusImageLabel.setPixmap(cross)

        if self.pupil_ready:
            self.pupilImageLabel.setPixmap(tick)
        else:
            self.pupilImageLabel.setPixmap(cross)

        if self.field("intro.pupil"):
            self.pupilTextLabel.setHidden(False)
            self.pupilImageLabel.setHidden(False)
        else:
            self.pupilTextLabel.setHidden(True)
            self.pupilImageLabel.setHidden(True)

        if self.field("intro.polhemus"):
            self.polhemusTextLabel.setHidden(False)
            self.polhemusImageLabel.setHidden(False)
        else:
            self.polhemusTextLabel.setHidden(True)
            self.polhemusImageLabel.setHidden(True)

        if self.field("intro.leap"):
            self.leapTextLabel.setHidden(False)
            self.leapImageLabel.setHidden(False)
        else:
            self.leapTextLabel.setHidden(True)
            self.leapImageLabel.setHidden(True)

    def playTone(self):
        """
        This function plays the tone to initiate the reach and grasp task. Before the tone is played a timestamp is
        taken to denote the starting time.
        """

        self.tone_timestamps[self.current_task] = datetime.utcnow()
        self.tone_player.play()
        self.status_label.setText("Recording, stop when task is completed")
        self.recordingButton.setEnabled(True)

    def updatePage(self):

        if self.field('intro.pupil'):
            self.pupilCaptureButton.setHidden(False)
            self.step_one_aGroupBox.setHidden(False)
        else:
            self.pupilCaptureButton.setHidden(True)
            self.step_one_aGroupBox.setHidden(True)

    def openleapVisualizer(self):
        # if the software is already open then focus it, else open it.
        # TODO find program recursively

        filename = "VRVisualizer.exe"
        path = None
        for root, dir, files in os.walk(r"C:\Program Files\Leap Motion"):
            if filename in files:
                path = os.path.join(root, filename)

        if ("VRVisualizer.exe" not in (p.name() for p in psutil.process_iter())):
            if path is None:
                self.error_dialog.showMessage("Can't find Leap Visualizer program")
            else:
                subprocess.Popen(path)
        else:
            self.error_dialog.showMessage('Vizualizer is already open')

    def openPupilCapture(self):

        # Recursively find it instead
        # if the software is already open then focus it, else open it.

        filename = "pupil_capture.exe"
        path = None
        for root, dir, files in os.walk(r"C:\Program Files (x86)\Pupil-Labs"):
            if filename in files:
                path = os.path.join(root, filename)

        if ("pupil_capture.exe" not in (p.name() for p in psutil.process_iter())):
            if path is None:
                self.error_dialog.showMessage("Can't find Pupil Capture program")
            else:
                subprocess.Popen(path)
        else:
            self.error_dialog.showMessage('Pupil  is already open')

    def calibrationButtonToggle(self):
        if self.pupilCalibrationButton.isChecked():
            # name = time.strftime("%Y%m%d-%H%M%S") + "_calibration"
            # self.pupilStartRecording.emit(name)
            # home = os.path.expanduser("~")
            # recordings_path = home + "\\recordings\\" + name
            # self.pupil_recordings.append(recordings_path)


            if self.field('intro.pupil'):
                if not self.pupil_ready:
                    logging.warning("Pupil was not ready to calibrate")
                    self.pupilCalibrationButton.setChecked(False)
                    return

            self.pupil_calibrated = True
            self.pupilCalibrationButton.setText("Stop Calibration")
            self.pupilStartCalibration.emit()
        else:
            self.pupilCalibrationButton.setText("Start Calibration")
            self.pupilStopCalibration.emit()

    def recordingButtonToggle(self):

        if self.recordingButton.isChecked():
            # Check that the devices which are being used are ready to record.

            if self.field('intro.leap'):
                if not self.leap_ready:
                    # dialog leap not ready
                    logging.warning("Leap was not ready to record")
                    self.recordingButton.setChecked(False)
                    return

            if self.field('intro.polhemus'):
                if not self.polhemus_ready:
                    # dialog leap not ready
                    logging.warning("Polhemus was not ready to record")
                    self.recordingButton.setChecked(False)
                    return

            if self.field('intro.pupil'):
                if not self.pupil_ready:
                    logging.warning("Pupil was not ready to record")
                    self.recordingButton.setChecked(False)
                    return
                elif not self.pupil_calibrated:
                    logging.warning("Didn't start recording as Pupil was not calibrated")
                    return

            self.recordingButton.setText("Stop Recording")
            self.recordingButton.setEnabled(False)

            if self.field('rec.visually_guided'):
                task = "_visual"
                self.num_vis = self.num_vis + 1
            elif self.field('rec.memory_guided'):
                task = "_memory"
                self.num_mem = self.num_mem + 1
            else:
                task = "_none"

            if self.field('rec.grasp'):
                type = "_grasp"
            elif self.field('rec.point'):
                type = "_point"
            else:
                type = "_none"

            self.current_task = time.strftime("%Y%m%d-%H%M%S") + task

            if self.field('intro.leap') and self.leap_ready:
                leap_filename = self.current_task + "_leap.csv"
                self.leapStartRecording.emit(leap_filename)
                self.leap_recordings.append(leap_filename)

            if self.field('intro.polhemus') and self.polhemus_ready:
                polhemus_filename = self.current_task + "_polhemus.csv"
                self.polhemusStartRecording.emit(polhemus_filename)
                self.polhemus_recordings.append(polhemus_filename)

            if self.field('intro.pupil') and self.pupil_ready:
                self.pupilStartRecording.emit(self.current_task)
                home = os.path.expanduser("~")
                recordings_path = home + "\\recordings\\" + self.current_task
                self.pupil_recordings.append(recordings_path)

            duration = random.randint(3, 7)
            qtc.QTimer.singleShot(duration * 1000, self.playTone)
            self.status_label.setText("Recording, wait for tone")

        else:
            if self.field('intro.leap') and self.leap_ready:
                self.leapStopRecording.emit()
            
            if self.field('intro.polhemus') and self.polhemus_ready:
                self.polhemusStopRecording.emit()

            if self.field('intro.pupil') and self.pupil_ready:
                self.pupilStopRecording.emit()

            self.recordingButton.setText("Start Recording")
            self.status_label.setText("Ready")
            self.visuallyGuidedRadioButton.setText("Visually guided : {}".format(self.num_vis))
            self.memoryGuidedRadioButton.setText("Memory guided : {}".format(self.num_mem))

    @qtc.pyqtSlot(list)
    def leapTimestamps(self, timestamps):
        self.leap_timestamps[self.current_task] = (timestamps[0], timestamps[1])


class LeapHandler(qtc.QObject):
    """
    The leap handler uses wexpect to interface with 'LeapRecorder.exe', a command line program to record data from the Leap Motion Controller.
    This can be slow at times, so this Object has been adapted to use pysignals and slots so that it can be moved to a separate thread.
    'LeapRecorder.exe' takes the following inputs:

    +--------------------------------+-------+--------+
    | TASK                           | INPUT | RETURN |
    +--------------------------------+-------+--------+
    | Get Leap motion service status | 1     | INT    |
    +--------------------------------+-------+--------+
    | Get Number of devices          | 2     | INT    |
    +--------------------------------+-------+--------+
    | Get Timestamp                  | 3     | STR    |
    +--------------------------------+-------+--------+
    | Set Filename                   | 4     | VOID   |
    +--------------------------------+-------+--------+
    | Start Recording                | 5     | VOID   |
    +--------------------------------+-------+--------+
    | Stop Recording                 | 6     | VOID   |
    +--------------------------------+-------+--------+
    | Exit                           | 7     | VOID   |
    +--------------------------------+-------+--------+

    The Leap handler implements slots for each of these methods, which are called via a pysignals in the main thread. There are two outputs,
    'update' returns the

    """

    device_status = qtc.pyqtSignal(dict)
    timestamps = qtc.pyqtSignal(list)
    finished = qtc.pyqtSignal()

    def __init__(self, app_path):
        super().__init__()
        self.app_path = r'"{}"'.format(app_path)
        self.child = None

    @qtc.pyqtSlot()
    def start_application(self):
        """ Return bool

        Spawn the LeapRecorder.exe and wait until the program is ready for input. Return True if the program has
        successfully started.
        """

        # Ensure that the process has not been started or is currently running
        if (self.child is None):
            try:
                self.child = wexpect.spawn(self.app_path)
                self.child.expect('>')
            except Exception as e:
                logging.error("Leap application could not be opened")
                logging.error("Exception message: " + str(e))
                logging.error("Leap path: " + str(self.app_path))
                logging.error("Path exists?" + str(os.path.exists(self.app_path)))
            else:
                logging.info("Leap application spawned")

    @qtc.pyqtSlot()
    def get_status(self):

        update = {'app_active': False, 'device_connected': 0}

        app_status = "Recorder is not active"

        if self.child is not None:
            if self.child.isalive():
                update['app_active'] = True

                num_devices = self._sendCommand("2")

                if "Number of devices" in num_devices:
                    device = num_devices.split(',')[-1]
                    update['device_connected'] = int(device)

        self.device_status.emit(update)

    def getTimestamps(self):
        logging.info("Getting timestamps from Leap Motion")
        output = self._sendCommand("3")

        if output == None:
            logging.error("Unable to get timestamps")
        else:

            lines = output.split('\r')

            leap_timestamp = None
            system_timestamp = None

            for line in lines:
                if ("Leap timestamp" in line):
                    leap_timestamp = line.split(', ')[1]

                elif ("Current time" in line):

                    system_timestamp_utc = dt.datetime.fromisoformat(line.split(', ')[1])

                    epoch = dt.datetime.utcfromtimestamp(0)

                    system_timestamp = (system_timestamp_utc - epoch).total_seconds()

            if (leap_timestamp is not None) and (system_timestamp is not None):
                self.timestamps.emit([leap_timestamp,system_timestamp])
            else:
                logging.error("Timestamps could not be read")


    @qtc.pyqtSlot()
    def stop_application(self):
        self.sendCommand("Quit")

    @qtc.pyqtSlot(str)
    def start_recording(self, filename):
        try:
            self.getTimestamps()
            logging.info("Setting filename: {}".format(filename))
            self._sendCommand("4")  # Set Filename
            self._sendCommand(filename)
            self._sendCommand("5")  # Start Recording
            logging.info("Sending recording command")
        except:
            logging.error("Unable to start recording on Leap Motion")
        else:
            logging.info("Leap motion started recording successfully")

    @qtc.pyqtSlot()
    def stop_recording(self):
        self._sendCommand("6")  # Stop Recording

    def _sendCommand(self, command):
        try:
            self.child.sendline(command)
            # Wait for prompt
            self.child.expect('>')
        except wexpect.EOF:
            # The program has exited
            logging.error("Leap program has exited")
        else:
            return self.child.before


class PolhemusHandler(qtc.QObject):

    device_status = qtc.pyqtSignal(dict)

    def __init__(self, app_path):
        super().__init__()
        self.app_path = r'"{}"'.format(app_path)
        self.child = None

    @qtc.pyqtSlot()
    def get_status(self):

        update = {'app_active': False, 'device_connected': 0}

        app_status = "Recorder is not active"

        if self.child is not None:
            if self.child.isalive():
                update['app_active'] = True
                update['device_connected'] = 1

        self.device_status.emit(update)

    # @qtc.pyqtSlot()
    # def get_status(self):
    #     success = False
    #
    #     if self.child is not None:
    #         success = True
    #
    #     self.device_status.emit(success)

    # @qtc.pyqtSlot()
    # def start_application(self):
    #     """ Return bool
    #
    #     Spawn the PDIconsole.exe and wait until the program is ready for input. Return True if the program has
    #     successfully started.
    #     """
    #
    #     # Ensure that the process has not been started or is currently running
    #     if (self.child is None):
    #         try:
    #             self.child = wexpect.spawn(self.app_path)
    #
    #             self.child.expect('>')
    #         except:
    #             logging.error("Polhemus application could not be opened")
    #         else:
    #             logging.info("Polhemus application spawned")

    @qtc.pyqtSlot()
    def start_application(self):
        """ Return bool

        Spawn the PDIconsole.exe and wait until the program is ready for input. Return True if the program has
        successfully started.
        """

        if (self.child is None):
            try:
                self.child = wexpect.spawn(self.app_path)
                self.child.expect('>>')
            except Exception as e:
                logging.error(str(e))
                logging.error("Polhemus application could not be opened")
                logging.error("Path exists?" + str(os.path.exists(self.app_path)))
                logging.error("Polhemus path: " + str(self.app_path))
            else:
                if "exit" in self.child.before:
                    logging.error("Polhemus not plugged in")
                    self._sendCommand('q')  # Quit the application
                    self.child = None
                else:
                    logging.info("Polhemus application spawned")



    @qtc.pyqtSlot()
    def start_recording(self, filename):
        try:
            self._sendCommand('r')
            self._sendCommand(filename + "_polhemus.csv")
        except Exception as e:
            logging.error("Polhemus failed to start recording")
            logging.error(str(e))

    @qtc.pyqtSlot()
    def stop_recording(self):
        self._sendCommand('q')

    def _sendCommand(self, command):
        try:
            self.child.sendline(command)
        except wexpect.EOF:
            return "Command unsuccessful"
        else:
            return self.child.before

class PupilHandler(qtc.QObject):

    device_status = qtc.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        # This follows the network api protocol in the Pupil docs.
        self.ctx = zmq.Context()
        self.socket = zmq.Socket(self.ctx, zmq.REQ)
        self.socket.connect('tcp://127.0.0.1:50020')
        self.is_calibrated = False

    def get_status(self):
        # ensure that the pupil capture process is running
        if ("pupil_capture.exe" in (p.name() for p in psutil.process_iter())):
            self.is_ready = True
        else:
            self.is_ready = False

        status = {'capture_open': self.is_ready, 'calibrated': self.is_ready}

        self.device_status.emit(status)

    @qtc.pyqtSlot()
    def startCalibration(self):

        # If the Pupil Capture application is not open then open a dialog asking the user to open up Pupil Capture.

        # Additionally the program will hang if it cannot communicate with the software.

        if self.is_ready:
            try:
                self.socket.connect('tcp://127.0.0.1:50020')
                self.socket.send_string(r'C')
                reply = self.socket.recv_string()
            except:
                logging.error("Unable to start calibration on Pupil Core")
            else:
                logging.info("Start calibration, reply: {}".format(reply))

    @qtc.pyqtSlot()
    def stopCalibration(self):

        # If the Pupil Capture application is not open then open a dialog asking the user to open up Pupil Capture.
        # Additionally the program will hang if it cannot communicate with the software.

        if self.is_ready:
            try:
                self.socket.connect('tcp://127.0.0.1:50020')
                self.socket.send_string(r'c')
                reply = self.socket.recv_string()
            except:
                logging.error("Unable to stop calibration on Pupil Core")
            else:
                logging.info("Stop Calibration, reply: {}".format(reply))

    @qtc.pyqtSlot(str)
    def startRecording(self, filename):

        # If the Pupil Capture application is not open then open a dialog asking the user to open up Pupil Capture.
        # Additionally the program will hang if it cannot communicate with the software.

        if self.is_ready:
            try:
                self.socket.connect('tcp://127.0.0.1:50020')
                self.socket.send_string('R ' + filename)
                reply = self.socket.recv_string()
            except:
                logging.error("Unable to start recording on Pupil Capture")

            else:
                logging.info("Start recording, reply: {}".format(reply))

    @qtc.pyqtSlot()
    def stopRecording(self):

        # If the Pupil Capture application is not open then open a dialog asking the user to open up Pupil Capture.
        # Additionally the program will hang if it cannot communicate with the software.

        if self.is_ready:
            try:
                self.socket.connect('tcp://127.0.0.1:50020')
                self.socket.send_string('r')
                reply = self.socket.recv_string()
            except:
                logging.error("Unable to stop recording on Pupil Capture")

            else:
                logging.info("Stop recording, reply: {}".format(reply))


if __name__ == '__main__':

    import sys

    logging.basicConfig(filename='reach&grasp.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info("Starting application")
    app = qtw.QApplication(sys.argv)
    wizard = TrialWizard()
    wizard.show()
    sys.exit(app.exec_())