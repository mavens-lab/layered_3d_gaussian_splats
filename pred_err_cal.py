import os
import numpy as np

class BandwidthProcessor:
    def __init__(self, unit=1000):
        self.unit = unit
        self.bwValues = []

    def read_bandwidth_file(self, file_path):
        """Reads bandwidth values from a file and processes them based on the unit."""
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    value = float(line.strip())
                    self.bwValues.append(value)
                except ValueError:
                    # Ignore lines that cannot be converted to float
                    pass

    def compute_mean_square_error(self, ground_truth, predictions):
        """Computes the Mean Square Error (MSE) between ground truth and predictions."""
        if len(ground_truth) != len(predictions):
            raise ValueError("Ground truth and predictions must have the same length.")
        errors = np.array(ground_truth) - np.array(predictions)
        mse = np.mean(errors ** 2)
        return mse

    def process_file_pairs(self, ground_truth_dir, predictions_dir):
        """Processes file pairs from two directories to compute MSE."""
        results = {}
        ground_truth_files = set(os.listdir(ground_truth_dir))
        prediction_files = set(os.listdir(predictions_dir))
        common_files = ground_truth_files.intersection(prediction_files)

        for file_name in common_files:
            ground_truth_path = os.path.join(ground_truth_dir, file_name)
            predictions_path = os.path.join(predictions_dir, file_name)

            self.bwValues = []
            self.read_bandwidth_file(ground_truth_path)
            ground_truth = self.bwValues[:]

            self.bwValues = []
            self.read_bandwidth_file(predictions_path)
            predictions = self.bwValues[:]

            try:
                mse = self.compute_mean_square_error(ground_truth, predictions)
                results[file_name] = mse
            except ValueError as e:
                results[file_name] = str(e)  # Capture the error message

        return results

# Example usage:
if __name__ == "__main__":
    ground_truth_dir = "bw_traces"  # Replace with your ground truth directory
    predictions_dir = "bw_traces2"  # Replace with your predictions directory
    unit = 1000  # Replace with your unit if different

    processor = BandwidthProcessor(unit)
    results = processor.process_file_pairs(ground_truth_dir, predictions_dir)

    for file_name, mse in results.items():
        if isinstance(mse, str):
            print(f"File: {file_name}, Error: {mse}")
        else:
            print(f"File: {file_name}, Mean Square Error: {mse}")
