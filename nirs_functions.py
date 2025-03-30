import warnings
import os
import mne
import json
import pandas as pd
import matplotlib.pyplot as plt

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

def trigger_decoder(trigger: int) -> tuple:
    """
    Decode the trigger
    :param trigger: int
    :return: tuple
    """
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

def relabel_annotations(raw_haemo, mode) -> mne.Epochs:
    """
    Relabel the annotations in the data
    :param raw_haemo: Raw object
    :param mode: str
    :return: Epochs object
    """
    tmin = 0
    tmax = 16

    # Get the 'Base' annotations
    indices = []
    for i in range(0, len(raw_haemo.annotations.onset)):
        if int(raw_haemo.annotations.description[i]) >= 1000 and int(raw_haemo.annotations.description[i]) < 2000:
            raw_haemo.annotations.description[i] = 'Base'
        elif int(raw_haemo.annotations.description[i]) >= 2000:
            indices.append(i)
    raw_haemo.annotations.delete(indices)

    if mode == 'face_type':
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
    elif mode == 'face_type_emotion':
        indices = []
        for i in range(0, len(raw_haemo.annotations.onset)):
            if raw_haemo.annotations.description[i] == 'Base':
                face_type = ""
                if int(raw_haemo.annotations.description[i + 1]) >= 100:
                    face_type = 'Real'
                elif int(raw_haemo.annotations.description[i + 1]) < 100:
                    face_type = 'Virt'

                # get the tens digit of the next number
                next_num = int(raw_haemo.annotations.description[i + 1]) // 10 % 10
                if next_num == 0:
                    raw_haemo.annotations.description[i + 1] = face_type + 'Joy'
                elif next_num == 1:
                    raw_haemo.annotations.description[i + 1] = face_type + 'Fear'
                elif next_num == 2:
                    raw_haemo.annotations.description[i + 1] = face_type + 'Anger'
                elif next_num == 3:
                    raw_haemo.annotations.description[i + 1] = face_type + 'Disgust'
                elif next_num == 4:
                    raw_haemo.annotations.description[i + 1] = face_type + 'Sadness'
                elif next_num == 5:
                    raw_haemo.annotations.description[i + 1] = face_type + 'Neutral'
                elif next_num == 6:
                    raw_haemo.annotations.description[i + 1] = face_type + 'Surprise'

            # if raw_haemo.annotations.description[i] is a number, add it to the indices list
            if raw_haemo.annotations.description[i].isdigit():
                indices.append(i)
        raw_haemo.annotations.delete(indices)
    elif mode == 'all':        
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