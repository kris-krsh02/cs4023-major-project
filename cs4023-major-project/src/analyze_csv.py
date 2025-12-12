import csv
import matplotlib.pyplot as plt

def read_csv_file(filename):
    data = []
    # Open the file in read mode ('r')
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def split_data(data):
    x_vals = []
    y_vals = []
    for i in range(len(data)):
        x_vals.append(int(data[i][0]))
        y_vals.append(float(data[i][1]))
        
    return x_vals, y_vals

def moving_avg(data, window_size):
    #Calculate moving averages
        x_values = range(1,len(data)+1)
        moving_avgs = []
        for i in x_values: 
            if i < window_size:
                # average the first i values
                moving_avgs.append(sum(data[0:i])/(i))
            else:
                # average the most recent window_size values
                moving_avgs.append(sum(data[i-window_size:i])/window_size)
        return moving_avgs


# Window size for moving average
window_size = 100

### GENERATE COMBINED PLOT FOR AVERAGE CUMULATIVE REWARD       
data_0_50 = read_csv_file('cs4023-major-project\\src\\data\\CumulativeReward_0_50.csv')
episodes, rewards_0_50 = split_data(data_0_50)
avgs_0_50 = moving_avg(rewards_0_50, window_size)
data_0_200 = read_csv_file('cs4023-major-project\\src\\data\\CumulativeReward_0_200.csv')
episodes, rewards_0_200 = split_data(data_0_200)
avgs_0_200 = moving_avg(rewards_0_200, window_size)
data_09_50 = read_csv_file('cs4023-major-project\\src\\data\\CumulativeReward_0.9_50.csv')
episodes, rewards_09_50 = split_data(data_09_50)
avgs_09_50 = moving_avg(rewards_09_50, window_size)
data_09_200 = read_csv_file('cs4023-major-project\\src\\data\\CumulativeReward_0.9_200.csv')
episodes, rewards_09_200 = split_data(data_09_200)
avgs_09_200 = moving_avg(rewards_09_200, window_size)

plt.plot(range(1,201), avgs_0_50, label = f"\u03B3 = 0, \u03B5_decay = 50")
plt.plot(range(1,201), avgs_0_200, label = f"\u03B3 = 0, \u03B5_decay = 200")
plt.plot(range(1,201), avgs_09_50, label = f"\u03B3 = 0.9, \u03B5_decay = 50")
plt.plot(range(1,201), avgs_09_200, label = f"\u03B3 = 0.9, \u03B5_decay = 200")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.legend(loc = 'upper left')
plt.title(f"Moving average cumulative reward (n={window_size})")
plt.savefig(f"cs4023-major-project\\src\\plots\\avg_CumulativeReward_Compare.png")
plt.close()

### GENERATE COMBINED PLOT FOR AVERAGE NUMBER OF STEPS       
window_size = 100
data_0_50 = read_csv_file('cs4023-major-project\\src\\data\\NumberofSteps_0_50.csv')
episodes, steps_0_50 = split_data(data_0_50)
avgs_0_50 = moving_avg(steps_0_50, window_size)
data_0_200 = read_csv_file('cs4023-major-project\\src\\data\\NumberofSteps_0_200.csv')
episodes, steps_0_200 = split_data(data_0_200)
avgs_0_200 = moving_avg(steps_0_200, window_size)
data_09_50 = read_csv_file('cs4023-major-project\\src\\data\\NumberofSteps_0.9_50.csv')
episodes, steps_09_50 = split_data(data_09_50)
avgs_09_50 = moving_avg(steps_09_50, window_size)
data_09_200 = read_csv_file('cs4023-major-project\\src\\data\\NumberofSteps_0.9_200.csv')
episodes, steps_09_200 = split_data(data_09_200)
avgs_09_200 = moving_avg(steps_09_200, window_size)

plt.plot(range(1,201), avgs_0_50, label = f"\u03B3 = 0, \u03B5_decay = 50")
plt.plot(range(1,201), avgs_0_200, label = f"\u03B3 = 0, \u03B5_decay = 200")
plt.plot(range(1,201), avgs_09_50, label = f"\u03B3 = 0.9, \u03B5_decay = 50")
plt.plot(range(1,201), avgs_09_200, label = f"\u03B3 = 0.9, \u03B5_decay = 200")
plt.xlabel("Episodes")
plt.ylabel("Number of Steps")
plt.legend(loc = 'upper left')
plt.title(f"Moving average number of steps (n={window_size})")
plt.savefig(f"cs4023-major-project\\src\\plots\\avg_NumberofSteps_Compare.png")
plt.close()
