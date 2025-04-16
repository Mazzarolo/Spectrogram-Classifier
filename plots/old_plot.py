import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

bacteria_files = {
    'E. coli': 'bacteria/ECOL1.csv',
    'P. aeruginosa': 'bacteria/Paeruginosa2.csv',
    'S. agalactiae': 'bacteria/Salgalactiae1.csv',
    'S. aureus': 'bacteria/Saureus2.csv'
}

linestyles = ['-', '--', '-.', ':']

def generate_samples(df, num):
    dic = {}
    for i in range(num):
        dic[i] = df.sample(frac=0.1, random_state=1234)
    return dic

plt.figure(figsize=(14, 4))

plt.axvspan(170, 270, color='grey', alpha=0.3)
plt.axvspan(580, 640, color='grey', alpha=0.3)
plt.axvspan(710, 790, color='grey', alpha=0.3)

for idx, (key, value) in enumerate(bacteria_files.items()):
    df = pd.read_csv(value, quotechar='"', skiprows=3).transpose()
    samples = generate_samples(df, 1000)
    
    means = np.array([samples[i].mean() for i in range(len(samples))])
    mean_sample = means.mean(axis=0)
    std_sample = means.std(axis=0)
    
    x = range(len(mean_sample))
    
    plt.plot(x, mean_sample, label=key, linestyle=linestyles[idx])
    multiplier = 10000000000000
    plt.fill_between(x, mean_sample - std_sample * multiplier, mean_sample + std_sample * multiplier, alpha=0.4)

plt.legend()
plt.xlabel('Sample Points')
plt.ylabel('Mean Intensity')
plt.title('Mean Intensity of Bacteria Samples with Confidence Intervals')
plt.show()
