# L3 Lab: Time Series and Sequences

1. Submit all results in a directory under `sessions/L3_timeseries/lab/results/`. Name the directory using your student ID, or join both team members' IDs with an underscore. See the root `README.md` for the pull request rules.
1. Read Miebs et al. (2020), "Efficient Strategies of Static Features Incorporation into the Recurrent Neural Network." The paper is available at `sessions/L3_timeseries/lecture/resources/s11063-020-10195-x.pdf`.
1. The `data` directory contains four files with long-term energy-consumption data:
   * `TS4011.csv.gz`
   * `TS0277.csv.gz`
   * `TS3449.csv.gz`
   * `TS3423.csv.gz`
1. Decompose each time series into seasonal, trend, and residual components (for example, using `statsmodels.tsa.seasonal`).
1. Examine and describe the components at different time scales (daily, weekly, monthly, and yearly).
1. Normalize each time series by its mean consumption. Calculate the mean from the training data only to avoid using information from the test period.
1. Split each data set chronologically, reserving its most recent year for testing. Ensure that no input sequence or forecast window overlaps the training and test periods.
1. Using what you learned, particularly the session on RNNs, design an architecture that takes an input sequence of your chosen length and predicts future values. Evaluate forecast horizons from 1 to *n* time steps, and build a separate model for each time series. How does forecast quality change as the horizon increases? Quantify your findings.
1. Encode calendar features (for example, hour of day and day of week) using one-hot encoding or sine/cosine functions, and provide them to the model as additional dynamic inputs. Does this improve forecast quality?
1. Design static features that may help the model, for example:
   * Calendar features for the first time step in the forecast horizon.
   * Monthly aggregates (such as sums, means, or volumes).
   * Percentiles (for example, the 0th, 20th, 40th, and other selected percentiles).
   * Trend information.
   * Other features you devise that may improve forecasting.

   Calculate features using only information that would be available at forecast time.
1. Design an RNN-based architecture that can use both dynamic (sequence) and static features. Use the architectures discussed in the lecture as inspiration. Consider how the methods in Miebs et al. (2020) could be extended to other approaches. Train a generic model using the training data from all four time series, then evaluate it on each time series' held-out test period.
1. If you have questions or would like to discuss your approach, please contact me by email.
1. Submission deadline: 20.11.2026.
