import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import warnings
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import scipy
# The code is taken from https://github.com/deryckt/TIRE/blob/master/utils.py

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def distance(data, window_size):
    """
    Calculates distance (dissimilarity measure) between features
    
    Args:
        data: array of of learned features of size (nr. of windows) x (number of shared features)
        window_size: window size used for CPD
        
    Returns:
        Array of dissimilarities of size ((nr. of windows)-stride)
    """
    
    nr_windows = np.shape(data)[0]
    
    index_1 = range(window_size,nr_windows,1)
    index_2 = range(0,nr_windows-window_size,1)
    
    return np.sqrt(np.sum(np.square(data[index_1]-data[index_2]),1))

def parameters_to_cps(parameters,window_size):
    """
    Preparation for plotting ground-truth change points
    
    Args:
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        
    Returns:
        Array of which entry is non-zero in the presence of a change point. Higher values correspond to larger parameter changes.
    """
    
    length_ts = np.size(parameters,0)
        
    index1 = range(window_size-1,length_ts-window_size,1) #selects parameter at LAST time stamp of window
    index2 = range(window_size,length_ts-window_size+1,1) #selects parameter at FIRST time stamp of next window

    diff_parameters = np.sqrt(np.sum(np.square(parameters[index1]-parameters[index2]),1))
    
    max_diff = np.max(diff_parameters)
    
    return diff_parameters/max_diff

def cp_to_timestamps(changepoints, tolerance, length_ts):
    """
    Extracts time stamps of change points
    
    Args:
        changepoints:
        tolerance:
        length_ts: length of original time series
        
    Returns:
        list where each entry is a list with the windows affected by a change point
    """
    
    locations_cp = [idx for idx, val in enumerate(changepoints) if val > 0.0]
    
    output = []
    while len(locations_cp)>0:
        k = 0
        for i in range(len(locations_cp)-1):
            if  locations_cp[i]+1 == locations_cp[i+1]:
                k+=1
            else:
                break
        
        output.append(list(range(max(locations_cp[0]-tolerance,0),min(locations_cp[k]+1+tolerance,length_ts),1)))
        del locations_cp[:k+1]
        
    return output

def ts_to_windows(ts, onset, window_size, stride, normalization="timeseries"):
    """Transforms time series into list of windows"""
    windows = []
    len_ts = len(ts)
    onsets = range(onset, len_ts-window_size+1, stride)
    
    if normalization =="timeseries":
        for timestamp in onsets:
            windows.append(ts[timestamp:timestamp+window_size])
    elif normalization=="window":
        for timestamp in onsets:
            windows.append(np.array(ts[timestamp:timestamp+window_size])-np.mean(ts[timestamp:timestamp+window_size]))
            
    return np.array(windows)

def combine_ts(list_of_windows):
    """
    Combines a list of windows from multiple views to one list of windows
    
    Args:
        list_of_windows: list of windows from multiple views
        
    Returns:
        one array with the concatenated windows
    """
    nr_ts, nr_windows, window_size = np.shape(list_of_windows)
    tss = np.array(list_of_windows)
    new_ts = []
    for i in range(nr_windows):
        new_ts.append(tss[:,i,:].flatten())
    return np.array(new_ts)

def new_peak_prominences(distances):
    """
    Adapted calculation of prominence of peaks, based on the original scipy code
    
    Args:
        distances: dissimarity scores
    Returns:
        prominence scores
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        all_peak_prom = peak_prominences(distances,range(len(distances)))
    return all_peak_prom

def tpr_fpr(bps,distances, method="prominence",tol_dist=0):
    """Calculation of TPR and FPR
    
    Args:
        bps: list of breakpoints (change points)
        distances: list of dissimilarity scores
        method: prominence- or height-based change point score
        tol_dist: toleration distance
        
    Returns:
        list of TPRs and FPRs for different values of the detection threshold
    """
    
    peaks = find_peaks(distances)[0]
    peaks_prom = peak_prominences(distances,peaks)[0]
    peaks_prom_all = np.array(new_peak_prominences(distances)[0])
    distances = np.array(distances)
        
    bps = np.array(bps)
        
    if method == "prominence":
        
        nr_bps = len(bps)
        
        index_closest_peak = [0]*nr_bps
        bp_detected = [0]*nr_bps
        
        #determine for each bp the allowed range s.t. alarm is closest to bp
        
        ranges = [0] * nr_bps
        
        for i in range(nr_bps):
            
            if i==0:
                left = 0
            else:
                left = right
            if i==(nr_bps-1):
                right = len(distances)
            else:
                right = int((bps[i][-1]+bps[i+1][0])/2)+1
                
            ranges[i] = [left,right]
        
        quantiles = np.quantile(peaks_prom,np.array(range(51))/50)
        quantiles_set = set(quantiles)
        quantiles_set.add(0.)
        quantiles = list(quantiles_set)
        quantiles.sort()
        
        nr_quant = len(quantiles)
        
        ncr = [0.]*nr_quant
        nal = [0.]*nr_quant
        
        for i in range(nr_quant):
            for j in range(nr_bps):
                
                bp_nbhd = peaks_prom_all[max(bps[j][0]-tol_dist,ranges[j][0]):min(bps[j][-1]+tol_dist+1,ranges[j][1])]
                
                if max(bp_nbhd) >= quantiles[i]:
                    ncr[i] += 1
                            
            indices_all = np.array(range(len(distances)))
            heights_alarms = distances[peaks_prom_all>=quantiles[i]]
            nal[i] = len(heights_alarms)
                    
        ncr = np.array(ncr)
        nal = np.array(nal)
        
        ngt = nr_bps
        
        tpr = ncr/ngt
        fpr = (nal-ncr)/nal
        
        tpr = list(tpr[nal!=0])
        fpr = list(fpr[nal!=0])
        
        tpr.insert(0,1.0)
        fpr.insert(0,1.0)
        tpr.insert(len(tpr),0.0)
        fpr.insert(len(fpr),0.0)
    return tpr, fpr

def precision_recall_f1(bps,distances, method="prominence",tol_dist=0):
    """Calculation of TPR and FPR
    
    Args:
        bps: list of breakpoints (change points)
        distances: list of dissimilarity scores
        method: prominence- or height-based change point score
        tol_dist: toleration distance
        
    Returns:
        list of TPRs and FPRs for different values of the detection threshold
    """
    
    peaks = find_peaks(distances)[0]
    peaks_prom = peak_prominences(distances,peaks)[0]
    peaks_prom_all = np.array(new_peak_prominences(distances)[0])
    distances = np.array(distances)
        
    bps = np.array(bps)
        
    if method == "prominence":
        
        nr_bps = len(bps)
        
        index_closest_peak = [0]*nr_bps
        bp_detected = [0]*nr_bps
        
        #determine for each bp the allowed range s.t. alarm is closest to bp
        
        ranges = [0] * nr_bps
        
        for i in range(nr_bps):
            
            if i==0:
                left = 0
            else:
                left = right
            if i==(nr_bps-1):
                right = len(distances)
            else:
                right = int((bps[i][-1]+bps[i+1][0])/2)+1
                
            ranges[i] = [left,right]
        
        quantiles = np.quantile(peaks_prom,np.array(range(51))/50)
        quantiles_set = set(quantiles)
        quantiles_set.add(0.)
        quantiles = list(quantiles_set)
        quantiles.sort()
        
        # we will be as flexible as possible as to the classifications - i.e. possible over-counting of false positives, but ce la vie.
        min_quantile = quantiles[1]
                
        n_true_positives = 0.0
        n_false_negatives = 0.0
        n_preds = 0.0
        
        for j in range(nr_bps):

            bp_nbhd = peaks_prom_all[max(bps[j][0]-tol_dist,ranges[j][0]):min(bps[j][-1]+tol_dist+1,ranges[j][1])]

            if max(bp_nbhd) >= min_quantile:
                n_true_positives += 1
            else:
                n_false_negatives += 1

        indices_all = np.array(range(len(distances)))
        
        total_P_N = len(distances)
        total_N = len(distances) - nr_bps
        
        total_P = nr_bps
        
        heights_alarms = distances[peaks_prom_all >= min_quantile]
        n_preds = len(heights_alarms)
                    
        n_true_negatives = total_N - n_false_negatives
        
        tpr = n_true_positives/total_P
        
        fpr = (n_preds-n_true_positives)/((n_preds-n_true_positives) + n_true_negatives)
        
        tp = n_true_positives
        
        fp = n_preds-n_true_positives
        
        tn = n_true_negatives
        
        fn = n_false_negatives
        
        precision = n_true_positives/n_preds
        
        recall = n_true_positives/total_P
        
        f1 = 2 * precision * recall/(precision + recall)
       
        
    return precision, recall, f1


def matched_filter(signal, window_size):
    """
    Matched filter for dissimilarity measure smoothing (and zero-delay weighted moving average filter for shared feature smoothing)
    
    Args:
        signal: input signal
        window_size: window size used for CPD
    Returns:
        filtered signal
    """
    mask = np.ones((2*window_size+1,))
    for i in range(window_size):
        mask[i] = i/(window_size**2)
        mask[-(i+1)] = i/(window_size**2)
    mask[window_size] = window_size/(window_size**2)
        
    signal_out = np.zeros(np.shape(signal))
    
    if len(np.shape(signal)) >1:
        for i in range(np.shape(signal)[1]):
            signal_extended = np.concatenate((signal[0,i]*np.ones(window_size), signal[:,i], signal[-1,i]*np.ones(window_size)))
            signal_out[:,i] = np.convolve(signal_extended, mask, 'valid')
    else:
        signal = np.concatenate((signal[0]*np.ones(window_size), signal, signal[-1]*np.ones(window_size)))
        signal_out = np.convolve(signal, mask, 'valid')
    
    return signal_out

def minmaxscale(data, a, b):
    """
    Scales data to the interval [a,b]
    """
    data_min = np.amin(data)
    data_max = np.amax(data)
        
    return (b-a)*(data-data_min)/(data_max-data_min)+a

def calc_fft(windows, nfft=30, norm_mode="timeseries"):
    """
    Calculates the DFT for each window and transforms its length
    
    Args:
        windows: time series windows
        nfft: number of points used for the calculation of the DFT
        norm_mode: ensure that the timeseries / each window has zero mean
        
    Returns:
        frequency domain windows, each window having size nfft//2 (+1 for timeseries normalization)
    """
    mean_per_segment = np.mean(windows, axis=-1)
    mean_all = np.mean(mean_per_segment, axis=0)
    
    if norm_mode == "window":
        windows = windows-mean_per_segment[:,None]
        windows_fft = abs(np.fft.fft(windows, nfft))[...,1:nfft//2+1]
    elif norm_mode == "timeseries":
        windows = windows-mean_all
        windows_fft = abs(np.fft.fft(windows, nfft))[...,:nfft//2+1]
        
    
    fft_max = np.amax(windows_fft)
    fft_min = np.amin(windows_fft)
        
    windows_fft = 2*(windows_fft-fft_min)/(fft_max-fft_min)-1
    
    return windows_fft


def plot_cp(distances, parameters, window_size, time_start, time_stop, plot_prominences=False):
    """
    Plots dissimilarity measure with ground-truth changepoints
    
    Args:
        distances: dissimilarity measures
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        time_start: first time stamp of plot
        time_stop: last time stamp of plot
        plot_prominences: True/False
        
    Returns:
        plot of dissimilarity measure with ground-truth changepoints
    
    """
    
    if parameters.shape[1] > 1:
        cps = []
        parameters_postprocess = []
        for i in range(parameters.shape[1]):
            parameters_i = np.array([a.item() if type(a) == np.ndarray else a for a in parameters[:, i]])
            parameters_i = np.expand_dims(parameters_i, axis=1)
            cps_i = parameters_to_cps(parameters_i, window_size)

            parameters_postprocess.append(parameters_i)
            cps.append(cps_i)

        parameters_postprocess = np.array(parameters_postprocess)
        cps = np.array(cps)

        breakpoints = np.amax(cps, axis=0)
    else:
        breakpoints = parameters_to_cps(parameters, window_size)


    length_ts = np.size(breakpoints)
    t = range(len(distances))


    x = t
    z = distances
    y = breakpoints#[:,0]

    peaks = find_peaks(distances)[0]
    peaks_prom = peak_prominences(distances,peaks)[0]
    peaks_prom_all = np.array(new_peak_prominences(distances)[0])

    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(x,z,color="black")
    
    if plot_prominences:
        ax.plot(x,peaks_prom_all, color="blue")

    ax.set_xlim(time_start,time_stop)
    ax.set_ylim(0,1.5*max(z))
    plt.xlabel("time")
    plt.ylabel("dissimilarity")

    ax.plot(peaks,distances[peaks],'ko')

    height_line = 1

    ax.fill_between(x, 0, height_line, where=y > 0.0001,
                color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x, 0, height_line, where=y > 0.25,
                color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x, 0, height_line, where=y > 0.5,
                color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x, 0, height_line, where=y > 0.75,
                color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x, 0, height_line, where=y > 0.9,
                color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    plt.show()
    
def get_auc(distances, tol_distances, parameters, window_size):
    """
    Calculation of AUC for toleration distances in range(TD_start, TD_stop, TD_step) + plot of corresponding ROC curves
    
    Args:
        distances: dissimilarity measures
        tol_distances: list of different toleration distances
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        
    Returns:
        list of AUCs for every toleration distance
    """
    

    if parameters.shape[1] > 1:
        cps = []
        parameters_postprocess = []
        for i in range(parameters.shape[1]):
            parameters_i = np.array([a.item() if type(a) == np.ndarray else a for a in parameters[:, i]])
            parameters_i = np.expand_dims(parameters_i, axis=1)
            cps_i = parameters_to_cps(parameters_i, window_size)

            parameters_postprocess.append(parameters_i)
            cps.append(cps_i)

        parameters_postprocess = np.array(parameters_postprocess)
        cps = np.array(cps)

        breakpoints = np.amax(cps, axis=0)
    else:
        breakpoints = parameters_to_cps(parameters, window_size)

    legend = []
    list_of_lists = cp_to_timestamps(breakpoints,0,np.size(breakpoints))
    auc = []

    for i in tol_distances:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tpr, fpr = tpr_fpr(list_of_lists,distances, "prominence",i)
        plt.plot(fpr,tpr)
        legend.append("tol. dist. = "+str(i))
        auc.append(np.abs(np.trapz(tpr,x=fpr)))

    print(auc)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.plot([0,1],[0,1],'--')
    legend.append("TPR=FPR")
    plt.legend(legend)
    plt.show()

    plt.plot(tol_distances,auc)
    plt.xlabel("toleration distance")
    plt.ylabel("AUC")
    plt.title("AUC")
    plt.show()
    
    return auc

def get_f1(distances, tol_distances, parameters, window_size):
    """
    Calculation of AUC for toleration distances in range(TD_start, TD_stop, TD_step) + plot of corresponding ROC curves
    
    Args:
        distances: dissimilarity measures
        tol_distances: list of different toleration distances
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        
    Returns:
        list of AUCs for every toleration distance
    """
    
    if parameters.shape[1] > 1:
        cps = []
        parameters_postprocess = []
        for i in range(parameters.shape[1]):
            parameters_i = np.array([a.item() if type(a) == np.ndarray else a for a in parameters[:, i]])
            parameters_i = np.expand_dims(parameters_i, axis=1)
            cps_i = parameters_to_cps(parameters_i, window_size)

            parameters_postprocess.append(parameters_i)
            cps.append(cps_i)

        parameters_postprocess = np.array(parameters_postprocess)
        cps = np.array(cps)

        breakpoints = np.amax(cps, axis=0)
    else:
        breakpoints = parameters_to_cps(parameters, window_size)

    legend = []
    list_of_lists = cp_to_timestamps(breakpoints,0,np.size(breakpoints))
    f1 = []

    for i in tol_distances:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            precision_single, recall_single, f1_single = precision_recall_f1(list_of_lists, distances, "prominence", i)
        f1.append(f1_single)

    print(f1)
    
    return f1
    
    
def combine_and_format_mat_datasets(dataset_name, dataset_path, scaling_type = 'minmax'):
    data, parameters = combine_mat_datasets(dataset_name, dataset_path)

    if scaling_type == 'standard':
        scaler_seq = StandardScaler()
    elif scaling_type == 'minmax':
        scaler_seq = MinMaxScaler()

    data = scaler_seq.fit_transform(data.reshape(-1,data.shape[1])).reshape(data.shape)

    return data, parameters, parameters

def combine_mat_datasets(dataset_name, dataset_path):
    data = []
    parameters = []
    for dataset_file in glob.glob(dataset_path + "{}*.mat".format(dataset_name)):
        temp_mat = scipy.io.loadmat(dataset_file)
        data.append(temp_mat['Y'])
        parameters.append(temp_mat['L'])
    data = np.concatenate(data, axis=0)
    parameters = np.concatenate(parameters, axis=0)
    return data, parameters

def smoothened_dissimilarity_measures(encoded_windows, encoded_windows_fft, domain, window_size):
    """
    Calculation of smoothened dissimilarity measures

    Args:
        encoded_windows: TD latent representation of windows
        encoded_windows_fft:  FD latent representation of windows
        domain: TD/FD/both
        parameters: array with used parameters
        window_size: window size used
        par_smooth

    Returns:
        smoothened dissimilarity measures
    """

    if domain == "TD":
        encoded_windows_both = encoded_windows
    elif domain == "FD":
        encoded_windows_both = encoded_windows_fft
    elif domain == "both":
        beta = np.quantile(distance(encoded_windows, window_size), 0.95)
        alpha = np.quantile(distance(encoded_windows_fft, window_size), 0.95)
        encoded_windows_both = np.concatenate((encoded_windows * alpha, encoded_windows_fft * beta), axis=1)

    encoded_windows_both = matched_filter(encoded_windows_both, window_size)
    distances = distance(encoded_windows_both, window_size)
    distances = matched_filter(distances, window_size)

    return distances


def change_point_score(distances, window_size):
    """
    Gives the change point score for each time stamp. A change point score > 0 indicates that a new segment starts at that time stamp.

    Args:
    distances: postprocessed dissimilarity measure for all time stamps
    window_size: window size used in TD for CPD

    Returns:
    change point scores for every time stamp (i.e. zero-padded such that length is same as length time series)
    """

    prominences = np.array(new_peak_prominences(distances)[0])
    prominences = prominences / np.amax(prominences)
    return np.concatenate((np.zeros((window_size,)), prominences, np.zeros((window_size - 1,))))
