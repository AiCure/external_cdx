import pandas as pd
import numpy as np

def print_full(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2):-int(window_len/2)]

def filter_by_confidence_and_thresh(x, fea, thresh):
    if x['s_confidence'] > 0.2 and np.fabs(x[fea]) < thresh:
        return x[fea]
    else:
        return np.NaN
    
def add_au(x, emotion):
    """
    computing composite emotion expressivity matrix
    Args:
        emotion: Action Unit
    """
    if x['s_confidence'] > 0.2: #if using smooth, no need for 'success'
        sum_r = 0
        cnt = 0
        for au in emotion:
            au_c_label = " AU{:02d}_c".format(au)
            au_r_label = " AU{:02d}_r".format(au)
            if x[au_c_label]==1 and (not np.isnan(x[au_r_label])): #there are data with face in, but au_c=0
                sum_r += x[au_r_label]*100. 
                cnt += 5
        if cnt > 0: 
            sum_r /= cnt
        else:
            sum_r = 0
        return x['Composite_Expressivity'] + sum_r 
    else:
        return np.NaN
    
#Composite emotion expressivity
def add_au_emotion(x, emotion,emotion_type):
    """
    computing individula emotion expressivity matrix
    Args:
        emotion: Action Unit
    """
    if x['s_confidence'] > 0.2: #if using smooth, no need for 'success'
        sum_r = 0
        cnt = 0
        for au in emotion:
            au_c_label = " AU{:02d}_c".format(au)
            au_r_label = " AU{:02d}_r".format(au)
            if x[au_c_label]==1 and (not np.isnan(x[au_r_label])): #there are data with face in, but au_c=0
                sum_r += x[au_r_label]*100. 
                cnt += 5
        if cnt > 0: 
            sum_r /= cnt
        else:
            sum_r = 0
        return x[emotion_type+'_Composite_Expressivity'] + sum_r 
    else:
        return np.NaN
    
def calc_of_for_video(of,ACTION_UNITS,POS_ACTION_UNITS,NEG_ACTION_UNITS,NET_ACTION_UNITS,happiness,
                     sadness,surprise,fear,anger,disgust,contempt,CAI):
    """
        Creating dataframe for emotion expressivity
    """
    of['Happiness_Composite_Expressivity'] = [0]*of.shape[0]
    of['Sadness_Composite_Expressivity'] = [0]*of.shape[0]
    of['Surprise_Composite_Expressivity'] = [0]*of.shape[0]
    of['Fear_Composite_Expressivity'] = [0]*of.shape[0]
    of['Anger_Composite_Expressivity'] = [0]*of.shape[0]
    of['Disgust_Composite_Expressivity'] = [0]*of.shape[0]
    of['Contempt_Composite_Expressivity'] = [0]*of.shape[0]
    of['Negative_Composite_Expressivity'] = [0]*of.shape[0]
    of['Positive_Composite_Expressivity'] = [0]*of.shape[0]
    of['Neutral_Composite_Expressivity'] = [0]*of.shape[0]
    of['CAI_Composite_Expressivity'] = [0]*of.shape[0]
    of['Composite_Expressivity'] = [0]*of.shape[0]
    
    for emotion in happiness:
        of['Happiness_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Happiness',), axis=1)
    for emotion in sadness:
        of['Sadness_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Sadness',), axis=1)    
    for emotion in surprise:
        of['Surprise_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Surprise',), axis=1)    
    for emotion in fear:
        of['Fear_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Fear',), axis=1)    
    for emotion in anger:
        of['Anger_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Anger',), axis=1)    
    for emotion in disgust:
        of['Disgust_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Disgust',), axis=1)    
    for emotion in contempt:
        of['Contempt_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Contempt',), axis=1)
    #Composite Positive Expressivity
    for emotion in POS_ACTION_UNITS:
        of['Positive_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Positive',), axis=1)
    #Composite Negative Expressivity
    for emotion in NEG_ACTION_UNITS:
        of['Negative_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Negative',), axis=1)
    #Composite Neutral Expressivity
    for emotion in NET_ACTION_UNITS:
        of['Neutral_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'Neutral',), axis=1)
    #Composite Activation Expressivity
    for emotion in CAI:
        of['CAI_Composite_Expressivity']=of.apply(add_au_emotion, args=(emotion,'CAI',), axis=1)
    #Composite Expressivity
    for emotion in ACTION_UNITS:
        of['Composite_Expressivity']=of.apply(add_au, args=(emotion,), axis=1)