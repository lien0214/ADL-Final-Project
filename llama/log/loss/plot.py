import json
import matplotlib.pyplot as plt

def read_log_file(filename):
    """
    Reads a JSON formatted log file and returns the data.
    """
    with open(filename, 'r') as file:
        log_data = json.load(file)
    return log_data

def plot_combined_loss(log_files, title):
    """
    Plots loss from multiple log files on the same plot.
    """
    plt.figure(figsize=(8, 6))
    for i, file in enumerate(log_files):
        log_data = read_log_file(file)
        # log_data = log_data["log_history"]
        epochs = [(i + 1) * 8 / 5 * (j + 1) for j in range(5)]
        # losses = [log['loss'] for log in log_data]
        plt.plot(epochs, log_data, marker='o', label=f'batch_size={2**(i+1)}')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("loss.jpg")

def main(log_files):
    """
    Main function to handle reading multiple log files and plotting them together.
    """
    plot_combined_loss(log_files, 'To-chinese Loss per Epoch under Different Batch Size')

if __name__ == "__main__":
    # Example log files (replace with actual file paths)
    example_log_files = [
        f"./e2c_mix_2",
        f"./e2c_mix_4",
        f"./e2c_mix_8"
    ]
    main(example_log_files)

