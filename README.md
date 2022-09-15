# Graph Neural Networks for Multi-Channel Time-Series Change Pointe Detection

This repository contains source code for the work done as part of research into whether Graph Neural Networks (GNNs) can be used to tackle the problem of multi-channel time-series change point detection.

Change point detection is an active area of research, where many methods ranging from the purely statistical to advanced deep learning methods have been developed and attempted at various use-cases and sub-problems within the domain of change point detection. For example, Zhang et al \[1\], authors of Correlation-Aware Change Point Detection (CORD-CPD), use Graph Neural Networks to create a detection method which aims to be able to detect whether a changepoint not only is present in a specific window of a time-series, but also whether this window contains a specific type of changepoint, namely a correlation change across the channels in the time-series. 

We aim to see if we can also com up with a general method for changepoint detection, where we use another very successful method for change point detection called TIRE, developed by De Ryck et al \[2\], which is a lot more efficient than CORD-CPD, but doesn't take into account spatial information, i.e. cross-channel information in the multi-channel time-series. TIRE follows a simple autoencoder architecture, with encoder and decoder comprised of simple fully-connected layers. The magic in TIRE happens in that the time-series is duplicated and lagged a number of times corresponding to a number of parallel autoencoders. Specifically, the autoencoder has weights trained for each of these lagged windowed time-series of the original time-series, and a proportion of the weights of the latent space are shared among these autoencoders, and differences in these shared, so-called time-invariant, features are incorporated into the loss function of TIRE. This loss function makes sure the autoencoder not only reconstructs the time-series, but also that it maintains smoothness in its representations of it. The change point detection method of TIRE, namely a smoothed dissimilarity metric, tries to detect abrupt changes in encoded windows of the time-series through the encoder of the autoencoder, i.e. to measure changes in the latent space representations, such that any change point would be detected ideally.

## Developing Extensions to TIRE

To tackle the extension of TIRE into a multi-channel setting which exploits cross-channel spatial information, we first begin by exploring the complexity of the TIRE model itself, namely, we explore whether TIRE as a base autoencoder with a simple single full-connected layer in the encoder and decoder steps of its autoencoder structure are sufficient in achieving state-of-the-art performance on all the datasets considered. 

### SeqTIRE

While these simplifying architectural choices aim to abstract a model as general as possible, we can still experiment with more advanced and perhaps appropriate architectures for the time-series case, such as sequence modelling layers like LSTMs or GRUs. These sequence modelling layers and architectures have seen much success in time-series forecasting tasks, anomaly detection tasks in time-series, and many others across domains such as NLP. Hence, they are at least worth trying here too.

To this effect, we develop and test the model architecture shown below in Figure 2.

In our experiments, we find that LSTMs perform best in the running of SeqTIRE, and notice a good increase of 2 to 3 AUC and F1 points on average across the datasets used in experiments.

### GNTIRE

Our Graph Network TIRE (GNTIRE) model is what we propose as a graph neural network extension to TIRE which ought to be able to tackle spatial cross-channel information extraction. A visual of the model is included in Figure 3.
