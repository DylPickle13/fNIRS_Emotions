import mne
import mne_nirs
from itertools import compress
from mne.preprocessing.nirs import (
    beer_lambert_law,
    optical_density,
    temporal_derivative_distribution_repair,
    source_detector_distances
)
from mne_nirs.signal_enhancement import enhance_negative_correlation
from mne_nirs.channels import get_long_channels

def preprocess_data(raw):
    """
    Preprocess the raw data
    :param raw: Raw object
    :return: raw_od_unprocessed, raw_haemo
    """ 

    # This is the sampling frequency of the data
    sampling_frequency = 6.105006105006105

    # if sampling frequency is greater then sampling_frequency, resample the data (there were 2 datasets at 10Hz)
    if raw.info['sfreq'] > sampling_frequency:
        raw.resample(sampling_frequency)
    
    # Drop short channels
    raw = drop_bad_short_channels(raw)

    # crop the data to the first and last annotation
    raw = raw.crop(tmin=raw.annotations[0]['onset'], tmax=raw.annotations[-1]['onset'])

    # convert the raw data to optical density
    raw_od = optical_density(raw, verbose=False)

    # Apply temporal derivative distribution repair
    raw_od = temporal_derivative_distribution_repair(raw_od, verbose=False)

    # Store the unprocessed optical density data
    raw_od_unprocessed = get_long_channels(raw_od.copy())

    # Apply short channel regression
    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)

    # Apply the Beer-Lambert law to convert OD data to hemoglobin concentrations.
    raw_haemo = beer_lambert_law(raw_od, ppf=0.1)

    # Retain only long channels (source-detector distance > 0.015 meters (15 mm))
    raw_haemo = get_long_channels(raw_haemo)

    # Filter the data
    raw_haemo.filter(
        l_freq=0.01,
        h_freq=0.5,
        filter_length=2015,        # Increase to meet the minimum for a 0.01 Hz transition band
        l_trans_bandwidth='auto',  # automatically set transition bandwidth (low)
        h_trans_bandwidth='auto',  # automatically set transition bandwidth (high)
        n_jobs=-1,                 # parallelization (useful for large data)
        method='fir',              # FIR filtering
        verbose=False,             # suppress output
    )

    # Enhance negative correlation
    raw_haemo = enhance_negative_correlation(raw_haemo)

    # remove the last annotation if it is <= 2000, for some reason the last annotation triggers on experiment end
    if int(raw_haemo.annotations.description[-1]) <= 2000:
        raw_haemo.annotations.delete(-1)

    # add hbt channels
    raw_haemo = add_hbt_channels(raw_haemo)

    return raw_od_unprocessed, raw_haemo

def drop_bad_short_channels(raw):
    """
    Drop bad short channels from the data
    :param raw: Raw object
    :return: Raw object
    """
    # get the source detector distances
    distances = source_detector_distances(raw.info)

    # remove channels with a distance greater than 0.05
    raw = raw.drop_channels(list(compress(raw.ch_names, distances > 0.05)))

    return raw

def add_hbt_channels(raw_haemo):
    """
    Add hbt channels to the data
    :param raw_haemo: Raw object
    :return: Raw object
    """
    raw_haemo_with_hbt = raw_haemo.copy()

    # get list of channel names for the hbo and hbr channels, and make a list of channel names for the hbt channels
    hbt_ch_names = [ch_name for ch_name in raw_haemo_with_hbt.ch_names[:len(raw_haemo_with_hbt.ch_names) // 2]]

    # replace the hbo in hb_ch_names with hbt
    hbt_ch_names = [ch_name.replace('hbo', 'hbt') for ch_name in hbt_ch_names]

    # create an info object for the hbt channels
    hbt_info = mne.create_info(hbt_ch_names, raw_haemo_with_hbt.info['sfreq'], ['fnirs_cw_amplitude'] * len(hbt_ch_names))

    # add a list of channels to raw_haemo_with_hbt, the channel type is 'hbt' where hbt = hbo + hbr
    raw_haemo_with_hbt.add_channels([mne.io.RawArray(raw_haemo_with_hbt.get_data('hbo') + raw_haemo_with_hbt.get_data('hbr'), hbt_info, verbose=False)], force_update_info=True)

    return raw_haemo_with_hbt