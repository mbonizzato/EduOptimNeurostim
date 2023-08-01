import tdt
import time
import numpy as np


def get_online_api(config):
    """Function for returning the desired online API"""
    if config.online_api == 'synapse':
        return SynapseAPI()
    else:
        raise NotImplementedError('API not implemented for %s' % config.online_api)


class SynapseAPI(object):
    """Provided class for interfacing between Python and Synapse"""
    def __init__(self):
        self.stimulator = 'eStim1'  # name of the stimulator, is configuration-specific
        self.emg_buffers = [f'buffer_{i}' for i in range(1, 4)]  # buffers to read responses from
        self.emg_index = 1  # which buffer to read the final response from
        self.baseline_index = 4  # baseline index for aggregating responses from a single buffer
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
        self.synapse_api.setParameterValue('UIn1', 'Button1', 1)

    def process_response_arr(self, response_arr):
        # Get responses from the previously indicated buffer only
        response_arr = response_arr[self.emg_index]
        # Take the baseline mean via the given baseline index
        baseline_mean = np.mean(response_arr[:self.baseline_index])
        # Normalize the picked out responses using the baseline mean
        response_arr = response_arr[self.baseline_index:] / baseline_mean
        # Return the maximum recorded response from the resultant array
        return np.max(response_arr)

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
