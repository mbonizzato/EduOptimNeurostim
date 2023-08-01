import os
import json
import random
from scipy.io import loadmat
from online import *

allowed_extensions = ['.mat']


def get_system(config):
    if config.input_space and not config.online:
        return System(config=config, input_space=config.input_space), 'offline'
    elif config.input_space and config.online:
        return System(config=config, input_space=config.input_space), 'online'

    return Dataset(dataset_path=config.dataset_path,
                   n_muscles=config.n_muscles,
                   selected_muscles=config.selected_muscles), 'offline'


class Muscle(object):
    def __init__(self, subject, muscle_index):
        self.subject = subject
        self.muscle_index = muscle_index
        self.sorted_isvalid = subject.data[subject.id][0][0][8][:, muscle_index]
        self.sorted_resp = subject.data[subject.id][0][0][9][:, muscle_index]
        self.sorted_resp_mean = subject.data[subject.id][0][0][subject.sorted_resp_mean_index][:, muscle_index]

    def get_response(self, mean=True):
        return self.sorted_resp_mean if mean else np.stack(self.sorted_resp.tolist())


class OnlineMuscle(object):
    def __init__(self, config):
        self.config = config

    def get_response_from_query(self, values, online_api):
        # Send query to online system
        for i, param in enumerate(list(self.config.input_space.keys())):
            online_api.set_param(param, values[i])

        # Trigger stimulation
        online_api.stimulate()

        # Read the response
        response = online_api.read_response()

        return response


class SyntheticMuscle(object):
    def __init__(self, config):
        self.config = config

    def get_response_from_query(self, query):
        raise NotImplementedError()


class Subject(object):
    def __init__(self, subject_path, selected_muscles=None):
        # Check if subject path exists
        assert os.path.exists(subject_path), 'subject_path=%s does not exist!' % subject_path

        # Retrieve subject ID and file extension
        self.id, extension = os.path.splitext(os.path.split(subject_path)[-1])
        # Check if file extension is allowed
        assert extension in allowed_extensions, 'extension=%s is not allowed!' % extension

        # Get data matrix indexing for the specific subject
        self.ch2xy_index, self.sorted_resp_mean_index = get_data_indices(subject_id=self.id)

        # Load subject data according to the extension
        self.data = None
        if extension == '.mat':
            self.data = loadmat(subject_path)

        # Initialize subject-specific attributes from the data
        self.emgs = self.data[self.id][0][0][0][0]
        self.n_channel = self.data[self.id][0][0][2][0][0]
        self.ch2xy = self.data[self.id][0][0][self.ch2xy_index]
        self.n_dim = self.ch2xy.shape[1]

        # Load muscles for each subject
        self.muscles = []
        for i in range(len(self.emgs)):
            muscle_name = self.emgs[i][0]
            if selected_muscles is None or muscle_name in selected_muscles:
                self.muscles.append(Muscle(subject=self, muscle_index=i))


class System(object):
    def __init__(self, config, input_space):
        self.config = config
        self.n_dim = len(input_space)
        self.shape = [len(v) for k, v in input_space.items()]
        self.n_channel = np.prod(self.shape)

        # Create channel to dim. value (`ch2xy`) and channel to dim. index (`ch2idx`) mappings
        # The values here serve as placeholder 
        self.ch2xy = np.empty((self.n_channel, self.n_dim))
        self.ch2idx = np.empty((self.n_channel, self.n_dim), dtype=np.int)
        for i in range(self.n_channel):
            for j, k in enumerate(np.unravel_index(i, shape=self.shape)):
                self.ch2xy[i, j] = input_space[list(input_space.keys())[j]][k]
                self.ch2idx[i, j] = k

        # Initialize system of muscles
        self.muscles = []

        # If toy system, return a synthetic muscle with a randomly generated response
        if (self.config.toy or self.config.input_space) and not self.config.online:
            for i in range(self.config.toy_n):
                muscle = SyntheticMuscle(config=self.config)
                self.muscles.append(muscle)
        else:      
            #If online, load the tdt channel to electrode coordinate ch2xy from json file
            assert os.path.exists(config.mapping_electrodes_online) and config.mapping_electrodes_online.endswith('.json')
            with open(config.mapping_electrodes_online) as f:
                online_mapping = json.load(f)
            #Overwrite values defined above
            self.ch2xy = np.array(online_mapping['ch2xy'],dtype=np.uint8)
            self.ch2idx = self.ch2xy - 1 

            muscle = OnlineMuscle(config=self.config)
            self.muscles.append(muscle)

    def __len__(self):
        return len(self.muscles)

    def __getitem__(self, index):
        return self.muscles[index]


class Dataset(object):
    def __init__(self, dataset_path, n_muscles=None, selected_muscles=None):
        # Check if dataset path exists
        assert os.path.exists(dataset_path), 'dataset_path=%s does not exist!' % dataset_path

        # Get subject paths
        subject_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
        # Sort subject paths
        subject_paths = list(sorted(subject_paths))

        # Initialize subjects
        self.subjects = [Subject(p, selected_muscles) for p in subject_paths]

        # Assert all subjects have the same num. channels and num. dims and set attributes
        assert len(set([subject.n_channel for subject in self.subjects])) == 1
        assert len(set([subject.n_dim for subject in self.subjects])) == 1
        self.n_channel, self.n_dim = self.subjects[0].n_channel, self.subjects[0].n_dim

        # Assume all subjects have the same `ch2xy` and set attributes
        self.ch2xy = self.subjects[0].ch2xy
        self.ch2idx = self.subjects[0].ch2xy - 1  # -1 due to MATLAB indexing
        self.shape = [len(set(self.ch2xy[:, i])) for i in range(self.n_dim)]

        # Get the muscles from subjects
        self.muscles = []
        for subject in self.subjects:
            self.muscles.extend(subject.muscles)

        # Randomly shuffle the dataset  
        random.shuffle(self.muscles)

        # Apply data fraction if applicable
        if n_muscles:
            self.muscles = self.muscles[:n_muscles]

        # Log the dataset with subjects and muscles to the user
        print('Dataset (sub:muscle): ', ['%s:%d' % (m.subject.id, m.muscle_index) for m in self.muscles])

        # Make sure that the dataset has at least one muscle
        assert len(self.muscles) >= 1

    def __len__(self):
        return len(self.muscles)

    def __getitem__(self, index):
        return self.muscles[index]


def get_data_indices(subject_id):
    """
    Function which returns data indices for `ch2xy` and `sorted_respMean` for
    mapping data fields for specific subjects

    :param (str) subject_id: The ID of the subject to-be-used for mapping
    """
    if subject_id in ['Cebus1_M1_190221', 'Cebus2_M1_200123']:
        return 16, 10
    elif subject_id in ['Macaque1_M1_181212', 'Macaque2_M1_190527', 'rat1_M1_190716',
                        'rat2_M1_190617', 'rat3_M1_190728', 'rat4_M1_191109',
                        'rat5_M1_191112', 'rat6_M1_200218']:
        return 14, 15
    else:

        raise ValueError('subject_id=%s is not recognized!' % subject_id)
