import subprocess
import speech_recognition as sr
import urllib
import urllib.request, urllib.parse, urllib.error
import json
import numpy as np
import pandas as pd
import boto3
import time
import requests

import sys
sys.path
sys.path.append('./././aicurelib')
from aicurelib.ailogging.logger import create_logger

logger = create_logger(__name__)

intent_score_column =['neg_prob', 'nut_prob', 'pos_prob','label']
comp_intent_score_column =['neg_prob','nut_prob','pos_prob','label','negative_epressivity','neutral_epressivity',
                           'pos_epressivity','composite_epressivity']

###########################################################
#Speech to Text using Deep Speech and google API starts
###########################################################

#Fetch content from audio using AWS STT API
def aws_transcribe(job_uri):
    """
        Using amazon aws transcribe speech to text api, extracting text from audio file.
        Returns: 
              Dataframe with final transcript(text) and confidence score for each word from transcript
    """
    transcribe = boto3.client('transcribe', region_name='us-west-2')
    job_name = "RandallTest2"
    text_df = ""
    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': job_uri},
            MediaFormat='wav',
            LanguageCode='en-US'
        )
        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            logger.info("Not ready yet...")
            time.sleep(5)
        if (status):
            if status['TranscriptionJob']['Transcript']:
                if status['TranscriptionJob']['Transcript']['TranscriptFileUri']:
                    text_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    text_json = requests.get(text_url, headers={'Accept': 'application/json'})
                    text_df = pd.DataFrame(text_json.json())
    except Exception as e:
        logger.error('Exception to convert audio to text {} for {}'.format(e,job_uri))
    return text_df


def aws_response_parse(job_uri):
    """
        Parsing AWS api response to fetch text
        Returns:
            Final transcript(Text) and Rate of Speech based on confidence score for each word from Transcript
    """
    transcript_text = ""
    ros = 0
    try:
        text_df = aws_transcribe(job_uri)
        if isinstance(text_df, pd.DataFrame):
            if 'results' in text_df.columns:
                if 'transcripts' in text_df['results']:
                    res_trans = text_df['results']['transcripts'] 
                    if len(res_trans)>0:
                        transcript_text = res_trans[0]['transcript']

            if 'results' in text_df.columns:
                if 'items' in text_df['results']:
                    item_trans = text_df['results']['items']
                    ros = ros_speech(item_trans)
    except Exception as e:
        logger.error('Exception while parsing ROS & Text {} '.format(e))
    return transcript_text, ros
                             
#Processing content for Intent
def intent_output(text,sentiment_api):
    """
        Computing intent score using nlp seniment api 
        Returns:
            (json)Probability score for content emotion
    """
    data = urllib.parse.urlencode({"text": text}).encode("utf-8")
    url = sentiment_api
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req,data=data) as f:
        resp = f.read()
        return resp
    
def intent_text_dict(text,sentiment_api):
    """
        Call to sentiment api
    """
    prob_output = ''
    #Minimum character should be > 2 for sentiment
    if len(text)>2:
        prob_output = intent_output(text,sentiment_api)
    return prob_output

def list_2_df(list_val, column):
    """
        Converting list into dataframe
        Returns:
            Dataframe
    """
    df = pd.DataFrame([list_val],columns=column)
    return df

def proba_pred(prob_intent):
    """
        Parsing json response of emotion probablity score
        Returns:
            List of probability score
    """
    prob_list = []
    if prob_intent:
        intent_score = json.loads(prob_intent.decode('utf-8'))
        prob_list.append(intent_score["probability"]["neg"])
        prob_list.append(intent_score["probability"]["neutral"])
        prob_list.append(intent_score["probability"]["pos"])
        prob_list.append(intent_score["label"])
    else:
        prob_list.extend([""]*4)
    return prob_list

def compute_composite_score(comp_lst):
    """
        Computing compsite score for content emotion
        Returns:
            Compsite score for negative/positive and neutral emotion
    """
    final_comp_list = [''*4]
    try:
        if len(comp_lst)==0 or comp_lst[0] == '':
            return final_comp_list
        comp_lst_sum = sum(comp_lst)
        new_comp_list = [(x / comp_lst_sum)*100 for x in comp_lst]
        score_list = []
        for i in range(len(new_comp_list)):
            if i ==0:
                score = new_comp_list[i]* 0.1
                score_list.append(score)
            elif i == 1:
                score = new_comp_list[i]* 0.5
                score_list.append(score)
            else:
                score = new_comp_list[i]* .9
                score_list.append(score)
        final_score = sum(score_list)
        final_comp_list = [new_comp_list[0],new_comp_list[1],new_comp_list[2],final_score]
    except Exception as e:
        logger.error('Exception while computing composite score {} '.format(e))
    return final_comp_list

def compute_label(df):
    """
        Preparing dataframe labels for using combined analysis of speech recognition and Deepspeech
        Args:
            df: Dataframe
        Returns:
            Composite list with all attributes values as element
    """
    new_dict = {}
    for index, row in df.iterrows():
        composite_intent_list =[]
        col_name = list(row.index)
        composite_intent_list.append(list(row[0:4]))
        flat_comp_list = [item for sublist in composite_intent_list for item in sublist]
        comp_score_lst = [row['neg_prob'],row['nut_prob'],row['pos_prob']]
        composite_score = compute_composite_score(comp_score_lst)  
        flat_comp_list.extend(composite_score)
    return flat_comp_list

def collect_content_result(audio_file,output_wav,stage_bucket_url,sentiment_api,pid,qid):
    """
        Preparing final dataframe with google/deepspeech content, Intent score
        Args:
            audio_file: Audio file location
            deep_path: Deep Speech pre-trained model location
            sentiment_api: NLP sentiment api
            pid: patient id
            qid: question id
        Returns:
            updated_intent_score_df: Dataframe with google/deepspeech content, Intent score 
    """
    job_uri = stage_bucket_url + output_wav
    aws_text,ros = aws_response_parse(job_uri)
    prob_intent = intent_text_dict(aws_text,sentiment_api)
    intent_score_list = proba_pred(prob_intent)
    intent_score_df = list_2_df(intent_score_list, intent_score_column)
    updated_intent_score_list = compute_label(intent_score_df)
    updated_intent_score_df = list_2_df(updated_intent_score_list, comp_intent_score_column)
    updated_intent_score_df['text_aws']= aws_text
    updated_intent_score_df['ros_speech']= ros
    updated_intent_score_df['pid']= pid
    updated_intent_score_df['qid']= qid
    return updated_intent_score_df

###########################################################
#Speech to Text using Deep Speech and google API ends
###########################################################

###########################################################
#Rate of Speech starts
###########################################################

def ros_speech(item_trans):
    """
        Computing Rate of speech using start and end time for each word(getting start and end time from AWS api response)
        where confidence score should be greater than .60
        
        Returns:
            Rate of speech: No of words per second
    """
    time_count = []
    confidence = 0
    ros_speech = 0
    for words in item_trans:
        try:
            if 'start_time' in words and 'end_time' in words:
                time_diff = float(words['end_time']) - float(words['start_time'])
                if 'alternatives' in words:
                    confidence = float(words['alternatives'][0]['confidence'])
                #Confidence threshold 0.6
                if confidence > 0.6:
                    time_count.append(time_diff)
        except:
            logger.error('Exception while computing ROS {} '.format(e))
            continue
    ros_speech = len(time_count)/np.sum(time_count)
    return ros_speech

###########################################################
#Rate of Speech ends
###########################################################

###########################################################
#Word Repetition starts
###########################################################

def word_count(df):
    """
        Computing each word count
        Returns:
            wordcount dictionary, Patient id and Question id
    """
    for index, row in df.iterrows():
        wordcount={}
        pid = row['pid']
        qid = row['qid']
        for word in str(row['text_aws_punc']).split():
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
    return wordcount,pid,qid

def word_percent(intent_df):
    """
        Computing percentage value for word repetition
        Returns:
            percentage value for word repetition
    """
    #Removing Punctuation from Text
    intent_df['text_aws_punc'] = intent_df['text_aws'].str.replace('[^\w\s]','')
    word_dict,pid,qid = word_count(intent_df)
    percent_val = 0
    if len(word_dict)>1:
        #if frequency of word is greater than 2
        count = sum(1 for i in word_dict.values() if i > 2)
        percent_val = count/len(word_dict)
    return percent_val,pid,qid

###########################################################
#Word Repetition ends
###########################################################
