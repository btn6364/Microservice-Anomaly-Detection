import numpy as np
import statistics
import matplotlib.pyplot as plt 


def gaussian_noise(data):
    # Genearte noise with same size as that of the data, μ = mean, σ = std, size = length of y.
    noise = np.random.normal(statistics.mean(data), statistics.stdev(data), len(data)) 

    # Add the noise to the data. 
    data_noised = data + noise  
    return data_noised

if __name__=="__main__":
    # Test Gaussian Noise on Anomaly Score for Microservice
    y = np.array([
        0.07, 0.0, 0.19, 0.11, 0.1, 0.09, 0.05, 0.03, 0.1, 0.08, 
        0.0, 0.17, 0.1, 0.06, 0.06, 0.07, 0.09, 0.05, 0.06, 0.06, 
        0.06, 0.0, 0.0, 0.0, 0.14, 0.09, 0.18, 0.08, 0.0, 0.05, 
        0.09, 0.06, 0.0, 0.06, 0.05, 0.09, 0.06, 0.17, 0.08, 0.15, 
        0.11, 0.11, 0.1, 0.08, 0.18, 0.14, 0.06, 0.06, 0.06, 0.03, 
        0.05, 0.1, 0.08, 0.09, 0.11, 0.17, 0.07, 0.04, 0.06, 0.06, 
        0.29, 0.19, 0.09, 0.1, 0.12, 0.03, 0.05, 0.07, 0.06, 0.07, 
        0.0, 0.05, 0.06, 0.09, 0.0, 0.07, 0.05, 0.09, 0.08, 0.0, 
        0.07, 0.05, 0.2, 0.22, 0.16, 0.06, 0.07, 0.07, 0.1, 0.06
    ]) 
    y_noised = gaussian_noise(y)
    x = np.arange(len(y)) 
    plt.title("Anomaly Score Before/After Adding Gaussian Noise") 
    plt.xlabel("Time") 
    plt.ylabel("Anomaly Score") 
    plt.plot(x,y,color="blue")  
    plt.plot(x,y_noised, color="red") 
    plt.savefig('gaussian.png')