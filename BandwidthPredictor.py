
import math
from scipy.stats import hmean

class BandwidthPredictor:

    def __init__(self):
        self.bwGroundTruth = []
        self.bwPredicted = []
        self.bwError = []
        self.unit = 50 # T in ms
        self.horizon = 5 # use as history

    def set_horizon(self, h):
        self.horizon = h

    def getGroundTruth(self):
        return self.bwGroundTruth
    
    def getUnitInMs(self):
        return self.unit
    
    # input file default unit is 1000 ms per line, bw in Mbps
    # set unit (T) to discretize the file into Mb/(per unit time)
    def read_bandwidth_file(self, file_path):
        # self.bwGroundTruth.append(0)
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    value = float(line.strip())
                    if (self.unit == 1000):
                        self.bwGroundTruth.append(value)
                    else:
                        count = 1000 / self.unit
                        value /= count
                        while (count > 0):
                            self.bwGroundTruth.append(value)
                            count -= 1
                except ValueError:
                    pass

    # called every T = 50 ms, input current timestamp since launch
    def get_predicted_bandwidth(self, timestamp):

        # find current index in ground truth
        t_index = math.floor(timestamp / self.unit)

        # calculate the error
        if self.bwPredicted:
            error = (self.bwPredicted[-1] - self.bwGroundTruth[t_index]) / self.bwGroundTruth[t_index]
            self.bwError.append(abs(error))

        # compute harmonic mean over the horizon or fewer if less than horizon
        horizon_start_index = 0 if t_index < self.horizon else t_index - self.horizon
        harmonic_bw = hmean(self.bwGroundTruth[horizon_start_index:t_index])

        max_error = 0
        error_start_index = 0 if len(self.bwError) < self.horizon else len(self.bwError) - self.horizon
        for i in range(error_start_index, len(self.bwError)):
            max_error = max(max_error, self.bwError[i])

        predicted_bw = harmonic_bw / (1 + max_error)
        if math.isnan(predicted_bw):
            predicted_bw = 0
        if (predicted_bw != 0):
            self.bwPredicted.append(harmonic_bw)

        # throughput (Mb) for next unit
        return predicted_bw if predicted_bw != 0 else 0


'''
Reference: https://dl.acm.org/doi/pdf/10.1145/2785956.2787486
RobustMPC: We assume that the throughput lower bound is
C_t/(1+err), where C_t is obtained using harmonic mean of past 5 chunks, while prediction error err
is the maximum absolute percentage error of the past 5 chunks.
'''