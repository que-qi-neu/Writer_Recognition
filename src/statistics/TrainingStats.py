import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader

def Display_Train_Loss(csvFile, step=15):
    df = pd.read_csv(csvFile).dropna(subset=['train_loss'])

    # Calculate the rolling average of the last 15 steps
    df['smooth_loss'] = df['train_loss'].rolling(window=step).mean()

    # Plot the smoothed line
    plt.plot(df['step'], df['smooth_loss'])
    plt.xlabel('Step')
    plt.ylabel(f'Triplet Training Loss ({step}-step avg)')
    plt.show()

def Display_Val_Loss(csvFile):
    df = pd.read_csv(csvFile).dropna(subset=['validation_loss'])

    # Plot the smoothed line
    plt.plot(df['epoch'], df['validation_loss'])
    plt.xlabel('epoch')
    plt.ylabel(f'Triplet Validation Loss')
    plt.show()

def Display_Tain_Loss_tf(tfeventDir):
    reader = SummaryReader("lightning_logs/version_0/")
    df = reader.scalars

    # 2. Filter for your loss and plot
    train_loss = df[df['tag'] == 'train_loss']
    plt.plot(train_loss['step'], train_loss['value'])
    plt.title("Training Loss")
    plt.show()

Display_Train_Loss("data/CVL-cropped/stats/training_log/version_22/metrics.csv")
Display_Val_Loss("data/CVL-cropped/stats/training_log/version_22/metrics.csv")
# Display_Tain_Loss_tf("lightning_logs/version_10/")