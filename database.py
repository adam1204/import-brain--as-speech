import os

# For downloading the files.
import requests, tarfile
from shutil import move, rmtree

# For loading eeg, indecies and features.
import glob, mne
import scipy.io

# For preprocessing.
from scipy import signal as sig

# For saving the EEG trials.
import pandas as pd

class Database(object):

    PARTICIPANTS = ["MM05","MM08","MM09","MM10","MM11","MM12","MM14","MM15","MM16","MM18","MM19","MM20","MM21","P02"]

    def __init__(self, database_dir):
        self.DATABASE_DIR = database_dir
        self.TAR_DIR = os.path.join(database_dir,"TAR")
        self.PARTICIPANTS_DIR = os.path.join(database_dir,"PARTICIPANTS")
        self.PARTICIPANT_DIR = lambda participant: os.path.join(self.PARTICIPANTS_DIR, participant)
        self.PARTICIPANT_DATA_DIR = lambda participant: os.path.join(self.PARTICIPANT_DIR(participant), "DATA")
        self.PARTICIPANT_FIGURE_DIR = lambda participant: os.path.join(self.PARTICIPANT_DIR(participant), "FIGURES")

        if not os.path.exists(database_dir):
            raise FileNotFoundError("Database.__init__: Please give a valid directory for the database.")
        if not os.path.exists(self.TAR_DIR):
            os.mkdir(self.TAR_DIR)
        if not os.path.exists(self.PARTICIPANTS_DIR):
            os.mkdir(self.PARTICIPANTS_DIR)

    def initialize(self, participant):
        self.download(participant)
        self.extract(participant)
        self.preprocess_eeg(participant)
    
    def download(self, participant):
        """ 
        Downloads ONE specific participants files in a .tar.bz2 compressed format from the authors webpage.
        
        Note
        ----
        The downloaded .tar.bz2 file in the self.TAR_DIR directory named as participant. E.g.: MM05.tar.bz2

        Input
        -----
        participant: The participant's ID. E.g.: "MM05"

        Raises
        ------
        request.exceptions.HTTPError: if an error occured while downloading.
        """

        URL = f"http://www.cs.toronto.edu/~complingweb/data/karaOne/{participant}.tar.bz2"
        CHUNK_SIZE = 1024 * 1024 * 1 # 1MB
        OUTPUT_PATH = os.path.join(self.TAR_DIR, f"{participant}.tar.bz2")

        if not os.path.exists(OUTPUT_PATH):
            with requests.get(URL, stream=True, allow_redirects=True) as response:
                response.raise_for_status() # Raises HTTPError in case of failure.
                with open(OUTPUT_PATH, "ab") as file:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        file.write(chunk)

    def extract(self, participant):
        """
        Extract ONE specific participants files from its .tar.bz2 file to a directory in the output_directory.

        Note
        ----
            The extracted .tar.bz2 file will be saved to the self.PARTICIPANTS_DIR directory named as participant. E.g.: MM05

        Input
        -----
            participant: The participant's ID. E.g.: "MM05"

        Raises:
            tarfile.ReadError: if an error occured while reading the .tar.bz2 file.
            FileNotFoundError: if the .tar.bz2 file cannot be found.
        """
        TAR_PATH = os.path.join(self.TAR_DIR, f"{participant}.tar.bz2")
        EXTRACT_PATH = os.path.join(self.PARTICIPANTS_DIR, "tmp")
        EXTRACTED_PARTICIPANTS_DATA_PATH = os.path.join(EXTRACT_PATH,"p","spoclab","users","szhao","EEG","data", participant)

        with tarfile.open(TAR_PATH) as tar:
            tar.extractall(EXTRACT_PATH)

        move(EXTRACTED_PARTICIPANTS_DATA_PATH, self.PARTICIPANTS_DIR)
        rmtree(EXTRACT_PATH)

        
        os.mkdir(self.PARTICIPANT_FIGURE_DIR(participant))
        os.mkdir(self.PARTICIPANT_DATA_DIR(participant))

    def preprocess_eeg(self, participant):
        """
        Preprocesses the EEG signal. 
        
        Preprocess pipeline
        -------------------
        1. Load EEG file and indecies file.
        2. Drop channels.
        3. Filter EEG signal.
        4. Split EEG signal into thinking and speaking trials.
        5. Save the trials.

        Note
        ----
        This function should be called after extract().
        Creates the participants DATA and subdirectories if they are do not exist. 

        Input
        -----
        participant: The participant's ID. E.g.: "MM05"

        """
        owd = os.getcwd()
        os.chdir(os.path.join("Database","PARTICIPANTS",participant))

        CNT_PATH = glob.glob("*.cnt")[0]
        INDECIES_PATH = glob.glob("epoch_inds.mat")[0]

        # Load files.
        eeg = mne.io.read_raw_cnt(CNT_PATH, preload=True)
        indecies = scipy.io.loadmat(INDECIES_PATH, variable_names=('thinking_inds', 'speaking_inds'))

        os.chdir(owd)

        # Drop Channels
        eeg.drop_channels(['VEO', 'HEO', 'EKG', 'EMG', 'Trigger', 'M1', 'M2'])

        # Plot raw EEG signal.
        self._plot_eeg(participant, eeg, "raw_eeg")

        # Butterworth filter the EEG signal.
        eeg = self._butterworth_filter_eeg(eeg)

        # Plot the filtered EEG.
        self._plot_eeg(participant, eeg, "filtered_eeg")
        
        # Split the EEG into trials and save them.
        self._eeg_to_data(participant, eeg, indecies)

    def _eeg_to_data(self, participant, eeg, indecies):
        """
        Split the EEG into trials and save them.

        Note
        ----

        Input
        -----
        participant
        eeg
        indecies


        """
        DATA_DIR = self.PARTICIPANT_DATA_DIR(participant)
        SPEAKING_DATA_DIR = os.path.join(DATA_DIR, "SPEAKING")
        THINKING_DATA_DIR = os.path.join(DATA_DIR, "THINKING")

        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
            os.mkdir(SPEAKING_DATA_DIR)
            os.mkdir(THINKING_DATA_DIR)
        if not os.path.exists(SPEAKING_DATA_DIR):
            os.mkdir(SPEAKING_DATA_DIR)
        if not os.path.exists(THINKING_DATA_DIR):
            os.mkdir(THINKING_DATA_DIR) 

        self._save_eeg_trials(eeg, indecies['speaking_inds'][0][1::2][:-1], SPEAKING_DATA_DIR)
        self._save_eeg_trials(eeg, indecies['thinking_inds'][0][:-1], THINKING_DATA_DIR)
 
    def _save_eeg_trials(self, eeg, indecies, output_dir):
        """
        
        Input:

        """
        i = 0
        for index_pair in indecies:
            sample_start, sample_end = index_pair[0][0], index_pair[0][1]
            eeg_trial = eeg.get_data("all", sample_start, sample_end)
            pd.DataFrame(eeg_trial).to_csv(os.path.join(output_dir, f"{i}.csv"), header=False)
            i+=1

    def _plot_eeg(self, participant, eeg, plot_name):
        """
        Plot the eeg signals in time and in psd and saves it.

        Note
        ----
        The plotted figures are saved in the participant's FIGURE directory. If the directory does not exists, this function also creates it.

        Input
        -----
        participant: The participant's ID. E.g.: "MM05"
        eeg: mne.io.Raw
            The eeg signal.
        plot_name: The filename of the saved plot.
        """
        FIGURE_DIR = self.PARTICIPANT_FIGURE_DIR(participant)
        if not os.path.exists(FIGURE_DIR):
            os.mkdir(FIGURE_DIR)

        eeg.plot(show=False).savefig(os.path.join(FIGURE_DIR, f"{plot_name}_time.png"))

        eeg.compute_psd().plot(show=False).savefig(os.path.join(FIGURE_DIR, f"{plot_name}_psd.png"))

    # Source: https://github.com/wjbladek/SilentSpeechClassifier/blob/master/SSC.py
    # Line 177, filter_data()
    def _butterworth_filter_eeg(self, eeg, lp_freq = 59, hp_freq = 1):
        """
        Executes the butterworth filtering on the eeg signals.

        Input
        -----
        eeg: mne.io.Raw
            The eeg signal.
        lp_freq : Frequency of a low-pass filter.
        hp_freq : Frequency of a high-pass filter.

        Returns
        -------
        eeg: mne.io.Raw
            The filtered eeg signal.
        """
        if hp_freq:
            for idx, eeg_vector in enumerate(eeg[:][0]):
                [b, a] = sig.butter(4, hp_freq/eeg.info['sfreq']/2, btype='highpass')
                eeg[idx] = sig.filtfilt(b, a, eeg_vector)
        if lp_freq:
            for idx, eeg_vector in enumerate(eeg[:][0]):
                [b, a] = sig.butter(4, lp_freq/eeg.info['sfreq']/2, btype='lowpass')
                eeg[idx] = sig.filtfilt(b, a, eeg_vector)
        return eeg

    def load_eeg_trials(self, participant, eeg_type):
        """
        Loads the participant's specific (thinking/speaking) EEG trials into memory and returns it.

        Note
        ----
        Only usable if _save_eeg_trials() has been called without errors so the .csv files are ready to be read.

        Input
        -----
        participant: The participant's ID. E.g.: "MM05"
        eeg_type: "thinking" or "speaking"

        Returns
        -------
        eeg_trials: The loaded eeg_trials.

        Raise
        -----
        ValueError: if the eeg_type is not thinking or speaking.

        """
        if not eeg_type == "thinking" and not eeg_type == "speaking":
            raise ValueError("eeg_type must be  thinking  or  speaking")

        DATA_DIR = self.PARTICIPANT_DATA_DIR(participant)
        DATA_DIR = os.path.join(DATA_DIR, "SPEAKING") if eeg_type == "speaking" else os.path.join(DATA_DIR, "THINKING")

        #Load trials.
        i = 0
        eeg_trials = []
        while os.path.exists( os.path.join(DATA_DIR, f"{i}.csv")):
            eeg_trials.append(pd.read_csv(os.path.join(DATA_DIR, f"{i}.csv")))
            i+=1
        return eeg_trials

    def load_labels(self, participant):
        """
        Loads the participant's label for each trial in order. The last label is not returned because the last trial is not valid.

        Note
        ----
        It uses the labels.txt file from the kinect_data folder.
        Each line contains one label.

        Input
        -----
        participant: The participant's ID. E.g.: "MM05"

        Returns
        -------
        Y: The participant's valid labels.
        """
        # Path to labels.txt.
        LABELS_PATH = os.path.join(self.PARTICIPANT_DIR(participant), "kinect_data", "labels.txt")
        
        # Read the lines and strip() them to get the labels.
        Y = []
        with open(LABELS_PATH, "rt") as file:
            Y = [line.strip() for line in file.readlines()]
        return Y[:-1]