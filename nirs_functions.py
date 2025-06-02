import warnings
import os
import mne
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(type: str, filter: str = 'none') -> list:
    """
    Load the data
    :param type: str
    :param good: bool
    :return: list
    """
    if type != 'raws' and type != 'raw_ods' and type != 'raw_haemos':
        raise Exception(f"Invalid type: {type}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Get the count of datasets in the folder
        processed_data_count = len([f for f in os.listdir(f'processed_data/{type}') if f.endswith('.fif')])    

        # Load the processed data
        data = []
        for i in range(1, processed_data_count + 1):
            data.append(mne.io.read_raw_fif(f'processed_data/{type}/{type[:-1]}{i}.fif', preload=True, verbose=False))

        if type == 'raw_haemos' and filter == 'good':
            all_channels = json.load(open("processed_data/mappings/mappings.json"))['all_channels']

            # load the percentage_good_windows_df and assert that it is not empty
            percentage_good_windows_df = pd.read_csv('processed_data/windows/percentage_good_windows.csv')
            if percentage_good_windows_df.empty:
                raise Exception("percentage_good_windows_df is empty")

            raw_haemo_good_recordings = []
            for i, raw_haemo in enumerate(data, 1):
                if len(percentage_good_windows_df) >= i:
                    if percentage_good_windows_df.iloc[i - 1]['Good Recording']:
                        raw_haemo_good = raw_haemo.copy().reorder_channels(all_channels)
                        raw_haemo_good_recordings.append(raw_haemo_good)
            
            return raw_haemo_good_recordings

    return data

def get_epochs(raw_haemos, mode, max_times, return_mode):
    epochs = {}
    raw_haemos_annotated = []
    dataframes = []
    for i, raw_haemo in enumerate(raw_haemos):
        raw_haemo = raw_haemo.copy()
        data_points, times = raw_haemo.get_data(return_times=True)
        data = pd.DataFrame(data_points.T, columns=raw_haemo.ch_names)
        data.insert(0, 'time', times)

        annots = raw_haemo.annotations.to_data_frame(time_format=None)

        # subtract the value of the first onset from the rest of the onsets
        annots['onset'] = annots['onset'] - annots['onset'][0]

        annots['previous_description'] = annots['description'].shift(1)

        # Convert the 'previous_description' column to numeric, coercing errors to NaN
        annots['previous_description'] = pd.to_numeric(annots['previous_description'], errors='coerce')

        # Convert the 'description' column to numeric, coercing errors to NaN
        annots['description'] = pd.to_numeric(annots['description'], errors='coerce')

        # In annots, remove the row if both description and previous_description are < 200
        annots = annots[~((annots['description'] < 200) & (annots['previous_description'] < 200))]

        # drop the 'previous_description' column
        annots = annots.drop(columns=['previous_description'])

        if mode == 'face_type':
            annots['description'] = annots['description'].apply(lambda x: trigger_decoder(x)[0])
        elif mode == 'emotion':
            annots['description'] = annots['description'].apply(lambda x: trigger_decoder(x)[1])
        elif mode == 'face_type_emotion':
            annots['description'] = annots['description'].apply(lambda x: trigger_decoder(x)[0] + '_' + trigger_decoder(x)[1] if trigger_decoder(x)[0] != 'Task' and trigger_decoder(x)[1] != 'Base' else trigger_decoder(x)[0])
        elif mode == 'all':
            annots['description'] = annots['description'].apply(lambda x: 'Blck' if trigger_decoder(x)[0] != 'Task' and trigger_decoder(x)[1] != 'Base' else trigger_decoder(x)[0])

        # Merge data and annots on 'time' column
        data.insert(1, 'description', '')
        closest_indices = np.searchsorted(data['time'].values, annots['onset'].values)
        # Ensure the indices are within the bounds of the data array
        closest_indices = np.clip(closest_indices, 0, len(data) - 1)
        # Assign the description values to the data DataFrame at the closest indices
        data.loc[closest_indices, 'description'] = annots['description'].values

        # Set the duration column of annots to the difference between the next and current onset
        annots['duration'] = annots['onset'].shift(-1) - annots['onset']

        # remove the 'Task' description from the annots dataframe
        annots = annots[~((annots['description'] == 'Task'))]

        if return_mode == 'annotated':
            raw_haemo.set_annotations(mne.Annotations(onset=annots['onset'].values, duration=annots['duration'].values, description=annots['description'].astype('str')))
            raw_haemos_annotated.append(raw_haemo)
            continue

        # 2) Identify the 103 channel prefixes (e.g. "S1_D1") via the " hbo" suffix
        prefixes = [col[:-4] for col in data.columns if col.endswith(' hbo')]

        # The three chromophores, in the order we want them
        channel_types = ['hbo', 'hbr', 'hbt']

        # Build the column list so that all hbo channels come first, then hbr, then hbt
        channel_columns = []
        for ct in channel_types:
            for prefix in prefixes:
                channel_columns.append(f"{prefix} {ct}")

        if return_mode == 'data':
            # Add the gender column to the data DataFrame
            data['Sex'] = get_info(raw_haemo)['gender']
            dataframes.append(data)
            continue

        # Find every index where a new description appears (non-empty string) → that's an epoch start
        epoch_starts = data.index[data['description'] != ''].tolist()
        # Add the end‐of‐file as final boundary
        boundaries = epoch_starts + [len(data)]

        # 3) Loop through each start→end, grab the three × 103 channels, reshape & pad
        ind_epochs = {}
        for start, end in zip(epoch_starts, boundaries[1:]):
            cond = data.at[start, 'description']
            
            # Skip processing if the condition is 'Task'
            if cond == 'Task':
                continue
            
            block = data.iloc[start:end][channel_columns].values  # shape (t, 309)
            t = block.shape[0]
            
            # reshape → (t, 3, 103)
            block = block.reshape(t, len(channel_types), len(prefixes))
            # transpose → (3, 103, t)
            block = block.transpose(1, 2, 0)
            
            # pad to (3, 103, max_times)
            padded = np.full((3, 103, max_times), np.nan)
            padded[:, :, :t] = block
            
            ind_epochs.setdefault(cond, []).append(padded)

        # remove the 'Task' condition from the ind_epochs dictionary
        ind_epochs = {k: v for k, v in ind_epochs.items() if k != 'Task'}

        if mode == 'face_type_emotion':
            # get the shape[0] for each key in the ind_epochs dictionary
            for condition, condition_data in ind_epochs.items():
                if condition == 'Base':
                    continue
                # pad the data with NaN to shape[0] = 8
                condition_data = np.array(condition_data)
                padded = np.full((8, condition_data.shape[1], condition_data.shape[2], max_times), np.nan)
                
                # fill the padded array with the data
                for i in range(condition_data.shape[0]):
                    padded[i, :, :, :] = condition_data[i]

                ind_epochs[condition] = padded
        
        # add the ind_epochs to the epochs dictionary
        for condition, condition_data in ind_epochs.items():
            condition_data = np.array(condition_data)

            # Rescale the epochs to baseline
            condition_data = mne.baseline.rescale(
                condition_data, times=np.linspace(0, max_times / raw_haemo.info['sfreq'], max_times), baseline=(0, 0), verbose=False
            )
            if condition not in epochs:
                epochs[condition] = []
            epochs[condition].append(condition_data)

    if return_mode == 'annotated':
        return raw_haemos_annotated
    elif return_mode == 'data':
        return dataframes
    return epochs

def trigger_decoder(trigger: int) -> tuple:
    """
    Decode the trigger
    :param trigger: int
    :return: tuple
    """
    if trigger >= 1000 and trigger < 2000:
        return ('Base', 'Base')
    elif trigger >= 2000:
        return ('Task', 'Task')

    face_type = None
    if trigger >= 100:
        face_type = 'Real'
    elif trigger < 100:
        face_type = 'Virt'

    emotion = None
    if trigger // 10 % 10 == 0:
        emotion = 'Joy'
    elif trigger // 10 % 10 == 1:
        emotion = 'Fear'
    elif trigger // 10 % 10 == 2:
        emotion = 'Anger'
    elif trigger // 10 % 10 == 3:
        emotion = 'Disgust'
    elif trigger // 10 % 10 == 4:
        emotion = 'Sadness'
    elif trigger // 10 % 10 == 5:
        emotion = 'Neutral'
    elif trigger // 10 % 10 == 6:
        emotion = 'Surprise'

    return (face_type, emotion)

def get_info(raw_haemo) -> dict:
    """
    Get the info from the raw data
    :param raw_haemo: Raw object
    :return dict
    """

    # the raw_haemo.info['description'] is a string in the form a dictionary, convert it to a dictionary
    return eval(raw_haemo.info['description'])

def pick_channels(data, types):
    """
    Pick the channels that contain the specified type
    :param types: str or list
    """
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

def plot_history(history, mode, mean_score):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Loss on the primary y-axis
    ax1.plot(history.history['loss'], label='Training Loss', color='blue', linestyle='dashed')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='blue', linestyle='solid')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for Accuracy
    ax2 = ax1.twinx()
    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Training Accuracy', color='red', linestyle='dashed')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linestyle='solid')
        ax2.set_ylabel('Accuracy', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    # Combine Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # Title and Layout
    plt.title(f'Training Loss & Accuracy for Mode: {mode}\nEpochs run: {len(history.history["loss"])}, Mean Score: {mean_score:.4f}')
    plt.show()