import os
import numpy as np

class BandwidthNormalizer:
    def __init__(self, target_mean=11.8):
        self.target_mean = target_mean

    def normalize_bandwidth_file(self, input_file, output_file):
        """Normalizes the bandwidth data to have a mean of `target_mean` and keeps the first 60 data points."""
        try:
            # Read and parse the input file
            with open(input_file, 'r') as file:
                data = [float(line.strip()) for line in file if line.strip().isdigit()]

            # Keep only the first 60 data points
            if len(data) > 60:
                data = data[:60]

            # Calculate current mean and rescale
            current_mean = np.mean(data)
            if current_mean != 0:
                scaling_factor = self.target_mean / current_mean
                data = [x * scaling_factor for x in data]

            # Write the normalized data to the output file
            with open(output_file, 'w') as file:
                file.writelines(f"{value}\n" for value in data)

        except Exception as e:
            print(f"Error processing file {input_file}: {e}")

    def normalize_directory(self, input_dir, output_dir):
        """Normalizes all bandwidth files in the input directory and writes them to the output directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file_name in os.listdir(input_dir):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_path = os.path.join(output_dir, file_name)

            if os.path.isfile(input_file_path):
                self.normalize_bandwidth_file(input_file_path, output_file_path)

# Example usage:
if __name__ == "__main__":
    input_directory = "bw_traces"  # Replace with your input directory
    output_directory = "normalized_bw"  # Replace with your output directory
    target_mean = 11.8  # Target mean in Mbps

    normalizer = BandwidthNormalizer(target_mean)
    normalizer.normalize_directory(input_directory, output_directory)

    print(f"Normalization complete. Files written to {output_directory}.")
