import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_rewards_from_csv(csv_file, save_path="reward_plot.png"):
    """
    Reads the reward data from a CSV file and plots the rewards over episodes.

    Args:
        csv_file (str): Path to the CSV file containing the reward data.
        save_path (str): Path to save the plot image.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Check if the CSV contains a 'reward' column
    if 'Total Reward' not in data.columns:
        print("Error: The CSV file must have a 'Total Reward' column.")
        return

    rewards = data['Total Reward'].values
    episodes = data.index + 1  # Assuming episodes are numbered starting from 1

    # Plotting the rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label="Reward per Episode", color="b", linestyle='-', marker='o')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Reward Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Create the full path by joining the save path with a file name
    full_save_path = os.path.join(save_path, "reward_plot_agent12.png")
    
    # Save the plot
    plt.savefig(full_save_path)
    plt.close()
    print(f"Plot saved to {full_save_path}")

# Example usage:
if __name__ == "__main__":
    # Replace with the path to your saved CSV file
    # csv_file = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\training_results\reward_log.csv"
    csv_file = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/training_results/reward_log_agent12.csv"
    # save_path = r"C:\Users\frodi\Documents\OptimisedHeating\AgentV2\training_results"
    save_path = "/Users/frodibrooks/Desktop/DTU/Thesis/OptimisedHeating/AgentV2/training_results"

    plot_rewards_from_csv(csv_file, save_path)
