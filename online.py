import tdt
import time
import os
import numpy as np
from scipy.signal import butter,filtfilt

import matplotlib.pyplot as plt


def get_online_api(config):
    """Function for returning the desired online API"""
    if config.online_api == 'synapse':
        return SynapseAPI()
    else:
        raise NotImplementedError('API not implemented for %s' % config.online_api)


def butter_filter(type_,data, cutoff, fs, order):
    """Function for applying a parametrized filter on a signal"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    if type_ == 'high':
        b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    elif type_ == 'low':
        b, a = butter(order, normal_cutoff, btype='lowpass', analog=False) 

    y = filtfilt(b, a, data) 
    
    return y

def filter(type_, cutoff=10, sampleRate = 6000, degree= 2 ):
    """Function for returning the parametrized filter in function form for easy calling"""
    def butter(responses):
        return butter_filter(type_,responses, cutoff, sampleRate, degree)

    return butter


def filter_emg():
    """Function for applying the full filtering process on an EMG signal"""
    high = filter('high', cutoff=70, sampleRate = 6032, degree=2 )
    low = filter('low',cutoff=30, sampleRate = 6032, degree=2 )
        
    return lambda x : low(np.abs(high(x)))


class SynapseAPI(object):
    """Provided class for interfacing between Python and Synapse"""
    def __init__(self):
        self.stimulator = 'eStim1'  # name of the stimulator, is configuration-specific
        self.emg_buffers = [f'buffer_{i}' for i in range(1, 4)]  # buffers to read responses from
        self.emg_index = 0  # which buffer to read the final response from
        self.response_index = np.array(np.array([153,193])/600*3600,dtype=np.int) #time window in ms second in which to look for the response, 100 ms baseline, 43 ms stimulation
        self.synapse_api = tdt.SynapseAPI('localhost')  # API for connecting to Synapse
        self.previous_response = None  # stores the previous response from the system

        self.read_response_method = 'delay'  # choose between 'delay' and 'loop'
        self.read_response_delay = 0.7  # delay (seconds) for response reading
        self.read_response_time_limit = 30  # time-limit (seconds) for response reading while loop

        # `params2fn` is a dictionary mapping param keys to functions for setting the param
        self.params2fn = {
            'amplitude': self.set_amplitude,
            'channel': self.set_channel
        }

        #Filtering function for EMGs to be called when processing the content of the Synapse buffer
        self.filter_emg = filter_emg()

        # Decide if to record the data as then there are configuration needs for Synapse circuits
        record = False
        if record:
            self.start_recording()
        else:
            self.start_preview()

    def start_preview(self):
        if self.synapse_api.getMode() < 1:
            self.synapse_api.setMode(2)
            time.sleep(10)

    def start_recording(self):
        if self.synapse_api.getMode() < 1:
            self.synapse_api.setMode(3)
        time.sleep(10)

    def set_param(self, param, value):
        assert param in self.params2fn
        self.params2fn[param](value)

    def set_amplitude(self, amplitude, voice='A'):
        self.synapse_api.setParameterValue(self.stimulator, f'Amp{voice}', amplitude)

    def read_channel(self,voice='A'):
        return self.synapse_api.getParameterValue(self.stimulator, f'Chan{voice}')

    def set_channel(self, channel, voice='A', handshake=False):
        self.synapse_api.setParameterValue(self.stimulator, f'Chan{voice}', int(channel))

        if handshake:
            print(f'Channel sent is :{channel}, channel received is : {self.read_channel(voice=voice)}')

    def delay(self, delay):
        self.synapse_api.setParameterValue(self.stimulator, 'DelayC', delay)

    def stimulate(self):
        self.synapse_api.setParameterValue('Button1', 'Go', 1)
        self.synapse_api.setParameterValue('Button1', 'Go', 0)

    def process_response_arr(self, response_arr):
        # Get responses from the previously indicated buffer only
        response_arr = response_arr[self.emg_index]
        #filter the signal
        filt_response = self.filter_emg(response_arr)
        # Return the maximum recorded response from the resultant array
        return np.max(response_arr[self.response_index[0]:self.response_index[1]])

    def read_response(self):
        def read_buffers():
            """Helper to read system EMG buffers and process the array to get a scalar response"""
            response_arr = np.stack([self.synapse_api.getParameterValues(b, 'array') for b in self.emg_buffers])
            return self.process_response_arr(response_arr=response_arr)

        response = None

        if self.read_response_method == 'delay':
            # Wait for a fixed period of time and then read buffers once
            time.sleep(self.read_response_delay)
            response = read_buffers()
        elif self.read_response_method == 'loop':
            # Keep reading buffers until a a different response value from before is read
            start_time = time.time()
            while response is None or response == self.previous_response:
                # Break if too much time has past
                if time.time() - start_time > self.read_response_time_limit:
                    break
                response = read_buffers()
        else:
            raise NotImplementedError()

        # Update previous response in the online API
        self.previous_response = response

        return response


def plot_online(explore_online, exploit_online,path,y_mu_mapped_arr):
    
        
    x = explore_online.shape[-1]

    fig,axs = plt.subplots(1,2)
    
    axs[0].plot(range(x), explore_online[0,0,0,0], 'k', alpha=0.9)
    axs[0].set_title('Exploration \n (value of best predicted combination)')
    axs[1].plot(range(x), exploit_online[0,0,0,0], 'b', alpha=0.9)
    axs[1].set_title('Exploitation \n (stimulation efficacy)')
    
    for i in range(2):
    
        axs[i].set_xlabel('Queries')
        axs[i].set_ylabel('Objective function value')
    
    plt.tight_layout()

    plt.savefig(os.path.join(path, 'performance_vs_queries_online.png'))
    plt.savefig(os.path.join(path, 'performance_vs_queries_online.svg'), format='svg')
    plt.close('all')
    
    plt.imshow(y_mu_mapped_arr, cmap='OrRd')
    plt.title('Estimated response map')
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    
    plt.savefig(os.path.join(path, 'estimated_map_online.png'))
    plt.savefig(os.path.join(path, 'estimated_map_online.svg'), format='svg')
    