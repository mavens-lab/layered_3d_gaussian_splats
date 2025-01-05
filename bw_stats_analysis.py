import os
import numpy as np

class BandwidthProcessor:
    def __init__(self, unit=1000):
        self.unit = unit
        self.bwGroundTruth = []

    def read_bandwidth_file(self, file_path):
        """Reads bandwidth values from a file and processes them based on the unit."""
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    value = float(line.strip())
                    self.bwGroundTruth.append(value)
                except ValueError:
                    # Ignore lines that cannot be converted to float
                    pass

    def compute_statistics(self):
        """Computes the mean and variance of the bandwidth values."""
        if not self.bwGroundTruth:
            raise ValueError("No bandwidth data to compute statistics.")
        # print(self.bwGroundTruth)
        mean = np.mean(self.bwGroundTruth)
        variance = np.var(self.bwGroundTruth)
        std = np.std(self.bwGroundTruth)
        return mean, variance, std

    def process_directory(self, directory_path):
        """Processes all files in the given directory."""
        results = {}
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                self.bwGroundTruth = []  # Reset for each file
                self.read_bandwidth_file(file_path)
                try:
                    mean, variance, std = self.compute_statistics()
                    results[file_name] = (mean, variance, std)
                except ValueError:
                    results[file_name] = (None, None, None)  # No data in the file
        return results

# Example usage:
if __name__ == "__main__":
    directory_path = "normalized_bw"
    unit = 1000  

    processor = BandwidthProcessor(unit)
    results = processor.process_directory(directory_path)

    for file_name, stats in results.items():
        mean, variance, std = stats
        if mean is not None:
            print(f"File: {file_name}, Mean: {mean}, Variance: {variance}, Std: {std}")
        else:
            print(f"File: {file_name}, No valid data found.")