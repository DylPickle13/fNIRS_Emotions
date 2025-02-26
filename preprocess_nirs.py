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
    # https://doi.org/10.1016/j.neuroimage.2018.09.025
    raw_od = temporal_derivative_distribution_repair(raw_od, verbose=False)

    # Store the unprocessed optical density data
    raw_od_unprocessed = get_long_channels(raw_od.copy())

    # interpolate bad channels
    # https://doi.org/10.1016/0013-4694(89)90180-6
    # raw_od.interpolate_bads(mode='accurate', method=dict(fnirs="nearest"), verbose=False)

    # Apply short channel regression
    # https://mne.tools/mne-nirs/stable/generated/mne_nirs.signal_enhancement.short_channel_regression.html
    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)

    # Apply the Beer-Lambert law to convert OD data to hemoglobin concentrations.
    # https://mne.discourse.group/t/why-ppf-and-not-dpf-in-mne-preprocessing-nirs-beer-lambert-law/4373
    raw_haemo = beer_lambert_law(raw_od, ppf=0.1)

    # Retain only long channels (source-detector distance > 0.015 meters (15 mm))
    raw_haemo = get_long_channels(raw_haemo)

    # Filter the data
    # https://doi.org/10.3389/fnhum.2018.00505
    raw_haemo.filter(
        l_freq=0.01,
        h_freq=0.5,
        filter_length=2015,        # # Increase to meet the minimum for a 0.01 Hz transition band
        l_trans_bandwidth='auto',  # automatically set transition bandwidth (low)
        h_trans_bandwidth='auto',  # automatically set transition bandwidth (high)
        n_jobs=-1,                 # parallelization (useful for large data)
        method='fir',              # FIR filtering
        verbose=False,             # suppress output
    )

    # Enhance negative correlation
    # https://doi.org/10.1016/j.neuroimage.2009.11.050
    raw_haemo = enhance_negative_correlation(raw_haemo)

    # remove the last annotation if it is <= 2000
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

def relabel_annotations(raw_haemo, mode):
    """
    Relabel the annotations in the data
    :param raw_haemo: Raw object
    :param mode: str
    :return: Epochs object
    """
    tmin = 0
    tmax = 16
    if mode == 'face_type':
        # Relabel the annotations to 'Base', 'Real', and 'Virt'
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if int(raw_haemo.annotations.description[i]) >= 1000 and int(raw_haemo.annotations.description[i]) < 2000:
                raw_haemo.annotations.description[i] = 'Base'
            elif int(raw_haemo.annotations.description[i]) >= 2000:
                indices.append(i)
        raw_haemo.annotations.delete(indices)
        
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if raw_haemo.annotations.description[i] == 'Base':
                if int(raw_haemo.annotations.description[i + 1]) >= 100:
                    raw_haemo.annotations.description[i + 1] = 'Real'
                elif int(raw_haemo.annotations.description[i + 1]) < 100:
                    raw_haemo.annotations.description[i + 1] = 'Virt'

            # if raw_haemo.annotations.description[i] is a number, add it to the indices list
            if raw_haemo.annotations.description[i].isdigit():
                indices.append(i)
        raw_haemo.annotations.delete(indices)
    elif mode == 'emotion':
        # Relabel the annotations to 'Base', 'Real', and 'Virt'
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if int(raw_haemo.annotations.description[i]) >= 1000 and int(raw_haemo.annotations.description[i]) < 2000:
                raw_haemo.annotations.description[i] = 'Base'
            elif int(raw_haemo.annotations.description[i]) >= 2000:
                indices.append(i)
        raw_haemo.annotations.delete(indices)
        
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if raw_haemo.annotations.description[i] == 'Base':
                # get the tens digit of the next number
                next_num = int(raw_haemo.annotations.description[i + 1]) // 10 % 10
                if next_num == 0:
                    raw_haemo.annotations.description[i + 1] = 'Joy'
                elif next_num == 1:
                    raw_haemo.annotations.description[i + 1] = 'Fear'
                elif next_num == 2:
                    raw_haemo.annotations.description[i + 1] = 'Anger'
                elif next_num == 3:
                    raw_haemo.annotations.description[i + 1] = 'Disgust'
                elif next_num == 4:
                    raw_haemo.annotations.description[i + 1] = 'Sadness'
                elif next_num == 5:
                    raw_haemo.annotations.description[i + 1] = 'Neutral'
                elif next_num == 6:
                    raw_haemo.annotations.description[i + 1] = 'Surprise'

            # if raw_haemo.annotations.description[i] is a number, add it to the indices list
            if raw_haemo.annotations.description[i].isdigit():
                indices.append(i)
        raw_haemo.annotations.delete(indices)
    elif mode == 'neutral_vs_emotion':
        # Relabel the annotations to 'Base', 'Real', and 'Virt'
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if int(raw_haemo.annotations.description[i]) >= 1000 and int(raw_haemo.annotations.description[i]) < 2000:
                raw_haemo.annotations.description[i] = 'Base'
            elif int(raw_haemo.annotations.description[i]) >= 2000:
                indices.append(i)
        raw_haemo.annotations.delete(indices)
        
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if raw_haemo.annotations.description[i] == 'Base':
                # get the tens digit of the next number
                next_num = int(raw_haemo.annotations.description[i + 1]) // 10 % 10
                if next_num == 0:
                    raw_haemo.annotations.description[i + 1] = 'Emotion'
                elif next_num == 1:
                    raw_haemo.annotations.description[i + 1] = 'Emotion'
                elif next_num == 2:
                    raw_haemo.annotations.description[i + 1] = 'Emotion'
                elif next_num == 3:
                    raw_haemo.annotations.description[i + 1] = 'Emotion'
                elif next_num == 4:
                    raw_haemo.annotations.description[i + 1] = 'Emotion'
                elif next_num == 5:
                    raw_haemo.annotations.description[i + 1] = 'Neutral'
                elif next_num == 6:
                    raw_haemo.annotations.description[i + 1] = 'Emotion'

            # if raw_haemo.annotations.description[i] is a number, add it to the indices list
            if raw_haemo.annotations.description[i].isdigit():
                indices.append(i)
        raw_haemo.annotations.delete(indices)
    elif mode == 'all':
        # Relabel the annotations
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if int(raw_haemo.annotations.description[i]) >= 1000 and int(raw_haemo.annotations.description[i]) < 2000:
                raw_haemo.annotations.description[i] = 'Base'
            elif int(raw_haemo.annotations.description[i]) >= 2000:
                indices.append(i)
        raw_haemo.annotations.delete(indices)
        
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if raw_haemo.annotations.description[i] == 'Base':
                if int(raw_haemo.annotations.description[i + 1]) >= 100:
                    raw_haemo.annotations.description[i + 1] = 'Blck'
                elif int(raw_haemo.annotations.description[i + 1]) < 100:
                    raw_haemo.annotations.description[i + 1] = 'Blck'

            # if raw_haemo.annotations.description[i] is a number, add it to the indices list
            if raw_haemo.annotations.description[i].isdigit():
                indices.append(i)
        raw_haemo.annotations.delete(indices)

        # remove the 'Base' annotations
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if raw_haemo.annotations.description[i] == 'Base':
                indices.append(i)
        raw_haemo.annotations.delete(indices)

    # Extract annotations and create events
    events, event_dict = mne.events_from_annotations(raw_haemo, verbose=False)
    
    # Create epochs
    epochs = mne.Epochs(
        raw_haemo,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        reject_by_annotation=False,
        proj=False,
        baseline=(0, 0),
        detrend=None,
        preload=True,
        verbose=False,
    )

    return epochs

def get_info(raw_haemo):
    # the raw_haemo.info['description'] is a string in the form a dictionary, convert it to a dictionary
    return eval(raw_haemo.info['description'])

def pick_channels(data, types):
    # make a copy of the raw data
    new_data = data.copy()

    if isinstance(types, list):
        # get the indices of the channels that contain the type
        indices = [i for i, s in enumerate(new_data.ch_names) if any(xs in s for xs in types)]
    else:
        # get the indices of the channels that contain the type
        indices = [i for i, s in enumerate(new_data.ch_names) if types in s]

    # pick the channel names that contain the type
    return new_data.pick([new_data.ch_names[i] for i in indices])