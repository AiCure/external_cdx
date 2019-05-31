import parselmouth
import numpy as np

#Audio Features defination

###############################################
#1. Formant Frequency Starts
###############################################

def formant_list(formant,snd):
    """
    Getting formant frequency per second
    Args:
        foramnt: Formant object for sound wave
        snd: Parselmouth sound object
    Returns:
        List of first and second formant for each frame
    """
    f1_list = []
    f2_list = []
    dur = snd.duration-0.02
    dur_round = round(dur, 2)
    time_list = np.arange(0.01, dur_round, 0.01)
    for time in time_list:
        f1 = formant.get_value_at_time(1,time)
        f2 = formant.get_value_at_time(2,time)
        if (str(f1) != '' and str(f1) != 'nan') and (str(f2) != '' and str(f2) != 'nan'):
            f1_list.append(f1)
            f2_list.append(f2)
    f_diff = [f2 - f1 for f2, f1 in zip(f2_list,f1_list)]
    return f1_list,f2_list,f_diff

def formant_score(path):
    """
    Using parselmouth library fetching Formant Frequency
    Args:
        path: (.wav) audio file location
    Returns:
        (list) list of Formant freq for each voice frame
    """
    sound_pat = parselmouth.Sound(path)
    formant = sound_pat.to_formant_burg()
    f_score = formant_list(formant,sound_pat)
    return f_score

###############################################
# Formant Frequency Ends
###############################################

###############################################
# Fundamental Frequency starts
###############################################

def audio_pitch(path):
    """
    Using parselmouth library fetching fundamental frequency
    Args:
        path: (.wav) audio file location
    Returns:
        (list) list of fundamental frequency for each voice frame
    """
    sound_pat = parselmouth.Sound(path)
    pitch = sound_pat.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    return list(pitch_values)

###############################################
# Fundamental Frequency ends
###############################################

###############################################
# Glottal Noise Excitation Ratio starts
###############################################

def gne_ratio(filepath):
    """
        Using parselmouth library fetching glottal noise excitation ratio
        Args:
            path: (.wav) audio file location
        Returns:
            (list) list of gne ratio for each voice frame, min,max and mean gne
    """
    sound = parselmouth.Sound(filepath)
    harmonicity_gne = sound.to_harmonicity_gne()
    gne_mean = harmonicity_gne.values[harmonicity_gne.values != -200].mean()
    gne_all_frames = harmonicity_gne.values[harmonicity_gne.values != -200]
    text_harmonicity = str(harmonicity_gne)
    text_harmonicity_split = text_harmonicity.split('\n')
    text_harmonicity_list = text_harmonicity_split[len(text_harmonicity_split)-3:]
    min_list, max_list = min_max_value(text_harmonicity_list)
    gne_min = min_list[0]
    gne_max = max_list[0]
    return gne_all_frames, gne_mean, gne_min, gne_max

def min_max_value(text_harmonicity_list):
    """
        Computing minimum and maximum gne
        Args:
            text_harmonicity_list: (list) gne ratio score
        Returns:
            (list) list of min,max gne
    """
    max_list = []
    min_list = []
    for i in range(len(text_harmonicity_list)-1):
        if i ==0:
            split_key = text_harmonicity_list[i].split(':')[1]
            min_list.append(split_key.strip())
        else:
            split_key = text_harmonicity_list[i].split(':')[1]
            max_list.append(split_key.strip())
    return min_list, max_list

###############################################
# Glottal Noise Excitation Ratio ends
###############################################

###############################################
# Harmonic Noise Ratio starts
###############################################

def hnr_ratio(filepath):
    """
        Using parselmouth library fetching harmonic noise ratio ratio
        Args:
            path: (.wav) audio file location
        Returns:
            (list) list of hnr ratio for each voice frame, min,max and mean hnr
    """
    sound = parselmouth.Sound(filepath)
    harmonicity = sound.to_harmonicity_ac()
    hnr_mean = harmonicity.values[harmonicity.values != -200].mean()
    hnr_all_frames = harmonicity.values[harmonicity.values != -200]
    hnr_min,hnr_max = min_max_hnr_value(hnr_all_frames)
    return hnr_all_frames, hnr_mean, hnr_min, hnr_max

def min_max_hnr_value(patient_hnr_list):
    """
        Computing minimum and maximum hnr
        Args:
            text_harmonicity_list: (list) hnr ratio score
        Returns:
            (list) list of min,max hnr
    """
    min_val = ''
    max_val = ''
    if len(patient_hnr_list) > 0:
        min_val = min(patient_hnr_list)
        max_val = max(patient_hnr_list)
    return min_val, max_val

###############################################
# Harmonic Noise Ratio ends
###############################################

###############################################
# Audio Intensity starts
###############################################

def intensity_score(path):
    """
    Using parselmouth library fetching Intensity
    Args:
        path: (.wav) audio file location
    Returns:
        (list) list of Intensity for each voice frame
    """
    sound_pat = parselmouth.Sound(path)
    intensity = sound_pat.to_intensity()
    return intensity.values[0]

###############################################
# Audio Intensity ends
###############################################

###############################################
# Normalized Amplitude Quotient starts
###############################################

def get_intensity(path):
    sound = parselmouth.Sound(path)
    intensity = sound.to_intensity()
    len_int = len(intensity.values[0])
    time_val = [i for i in np.linspace(sound.xmin,sound.xmax,len_int)]
    return list(intensity.values[0]),time_val

def glottal_pulse(path):
    sound = parselmouth.Sound(path)
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    n_pulses = parselmouth.praat.call(pulses, "Get number of points")
    periods = [parselmouth.praat.call(pulses, "Get time from index", i+1) -
           parselmouth.praat.call(pulses, "Get time from index", i)
           for i in range(1, n_pulses)]
    periods_val = [parselmouth.praat.call(pulses, "Get time from index", i)
           for i in range(1, n_pulses)]
    return periods_val,periods

def glottal_peak(pd_list):
    count = 0
    peak_time = {}
    for i in range(0,len(pd_list)-1):
        diff = pd_list[i+1] - pd_list[i]
        if diff>=1:
            peak_time[count] = [pd_list[i],pd_list[i+1]]
            count +=1
    return peak_time

def intensity_peak(pd_dict,intensity_val):
    intsty_list = []
    for key, value in pd_dict.items():
        max_int = max(intensity_val[value[0]:value[1]])
        intsty_list.append(max_int)
    return intsty_list

def naq_compute(intensity_list,der_int,period):
    #(np.mean(pd_list[0])/np.mean([abs(number) for number in pd_list[1]]))/(np.mean(pd_list[2])*1000)
    naq = (np.mean(intensity_list)/np.mean([abs(number) for number in der_int]))/(np.mean(period)*1000)
    return naq

def get_pulse_amplitude(path):
    intsty_feat = get_intensity(path)
    pulse_amp = glottal_pulse(path)
    pulse_period = pulse_amp[0]
    time_stamp = intsty_feat[1]
    intensity_val = intsty_feat[0]
    pd_index_list = []
    for period in pulse_period:
        pd_index = next(x[0] for x in enumerate(time_stamp) if x[1] > float(period))
        pd_index_list.append(pd_index)
    pd_dict = glottal_peak(pd_index_list)
    intensity_list = intensity_peak(pd_dict,intensity_val)
    if len(intensity_list)<=1:
        return ""
    intensity_der = np.gradient(intensity_list)
    naq_val = naq_compute(intensity_list,list(intensity_der),pulse_period)
    return naq_val

###############################################
# Normalized Amplitude Quotient ends
###############################################

###############################################
# Voice Frame Score starts
###############################################

def audio_pitch_frame(pitch):
    """
        Computing total number of speech and participant voiced frames
        Args:
            pitch: speech pitch
        Returns:
            (float) total voice frames and participant voiced frames
    """
    total_frames = pitch.get_number_of_frames()
    voiced_frames = pitch.count_voiced_frames()
    return total_frames, voiced_frames

def audio_vfs_val(path):
    """
        Using parselmouth library fetching fundamental frequency
        Args:
            path: (.wav) audio file location
        Returns:
            (float) total voice frames, participant voiced frames and voiced frames percentage
    """
    sound_pat = parselmouth.Sound(path)
    pitch = sound_pat.to_pitch()
    total_frames,voiced_frames = audio_pitch_frame(pitch)
    voiced_percentage = (voiced_frames/total_frames)*100
    return voiced_percentage, voiced_frames, total_frames

###############################################
# Voice Frame Score ends
###############################################