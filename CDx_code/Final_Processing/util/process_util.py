import pandas as pd
import numpy as np
import glob
import requests
import json
import configparser
from scipy.spatial import distance

FACE_OUT_LOC = '/video/face_expressivity'
VFS_OUT_LOC = '/audio/voice_frame_score'
FF_OUT_LOC = '/audio/fundamental_freq'
MOTION_OUT_LOC = '/video/head_movement'
PROMPT_OUT_LOC = '/audio/prompt_count'
FORMANT_OUT_LOC = '/audio/formant_freq'
AI_OUT_LOC = '/audio/intensity'
NAQ_OUT_LOC = '/audio/normalized_amp'
#Accessing data from processing confriguration file
config_proc_Parser = configparser.RawConfigParser()
config_proc_filePath = r'./data_processing/process_conf.txt'
config_proc_Parser.read(config_proc_filePath)
ANSA_Q8 = config_proc_Parser.get('proc_config', 'ansa_q8').split(',')
ANSA_Q34 = config_proc_Parser.get('proc_config', 'ansa_q34').split(',')
ANSA_Q41 = config_proc_Parser.get('proc_config', 'ansa_q41').split(',')
ANSA_44_51 = config_proc_Parser.get('proc_config', 'ansa_44_51').split(',')

def excel_parsing(master_csv):
    df = pd.read_excel(master_csv,header=None,sheet_name='Questionnaire')
    participant_df=df.reindex(df.index.drop(0))
    new_header = participant_df.iloc[0]
    participant_df = participant_df[1:]
    participant_df.columns = new_header
    new_col = participant_df["SUBJUUID"].str.split("_", n = 2, expand = True) 
    participant_df['PID'] = new_col[2]
    participant_df = participant_df.reset_index(drop=True)
    participant_df = participant_df[participant_df['AIQSTSCD'].isnull() == False]
    participant_df = participant_df[participant_df['VIDEOURI'].isnull() == False]
    participant_df = participant_df.reset_index(drop=True)
    return participant_df

def api_res_parse(response,qid):
    if len(response['questions']) > 0:
        question_type = ''
        question_status = ''
        for i in range(len(response['questions'])):
            ques_res = response['questions'][i]
            if ques_res['question_id']:
                ques_id = ques_res['question_id']
                if int(qid) == int(ques_id):
                    if 'files' in ques_res['question']:
                        if len(ques_res['question']['files'])>0:
                            for j in range(len(ques_res['question']['files'])):
                                if ques_res['question']['files'][j]['file_id']:
                                    if ques_res['question']['files'][j]['file_type'] == 'audio':
                                        question_status = 'audio'
                    audio_count = False
                    video_count = False
                    if str(question_status) == 'audio':
                        if 'answer_model' in ques_res:
                            if 'answer_displays' in ques_res['answer_model']:
                                ans_disp = ques_res['answer_model']['answer_displays']
                                if len(ans_disp) == 2:
                                    for k in range(len(ans_disp)):
                                        if ans_disp[k]['type']:
                                            if ans_disp[k]['type'] == 'free_video' and str(ans_disp[k]['count']) == '1':
                                                video_count = True
                                            if ans_disp[k]['type'] == 'free_audio' and str(ans_disp[k]['count']) == '1':
                                                audio_count = True
                    if audio_count and video_count:
                        question_type = 'AV'
                    else:
                        question_type = 'V'
                    return question_type
        return question_type

def av_check(qid_df,qnn_list,QNN_API):
    qnn_id = qnn_list.split('/')[-1]
    url = QNN_API + qnn_id
    question_type = []
    response = requests.get(url)
    j_response = json.loads(response.text)
    for index, row in qid_df.iterrows():
        ques_type = api_res_parse(j_response,row['QNID'])
        question_type.append(ques_type)
    qid_df['Question_Type'] = question_type
    return qid_df
        

def instrument_qnn(particpt_df, inst_type):
    qnn_df_list = []
    base_qnn_list = []
    try:
        col_val = [i for i in range(8)]
        inttype_df = particpt_df[particpt_df['AIQSTSCD'].str.match(inst_type)]
        inst_particpt_df = inttype_df.sort_values(by=['AIQSDAT','SUBJID'], ascending=[True,True])
        qnn_split_df = pd.DataFrame(inst_particpt_df.VIDEOURI.str.split('/',7).tolist(),columns = col_val)
        qnn_split_df['qnn_vid'] = qnn_split_df[col_val[:-1]].apply(lambda x: '/'.join(x), axis=1)
        #baseline entires
        group = qnn_split_df.groupby(3)
        base_qnn_list = list(group.apply(lambda x: x['qnn_vid'].unique()[0])) 
        qnn_df = qnn_split_df[['qnn_vid']]
        qnn_df_list = list(qnn_df.groupby(['qnn_vid']).size().reset_index()['qnn_vid'])
    except:
        return qnn_df_list,base_qnn_list
    return qnn_df_list,base_qnn_list

def df_combine(of_list):
    if len(of_list) == 1:
        df_csv = of_list[0]
    else:
        df_csv = pd.concat(of_list, ignore_index=True)
    return df_csv

def img_emotion_list(df_img):
    emotion_list = []
    emotion_list.append(df_img['Negative_Composite_Expressivity'].mean(axis = 0, skipna = True))
    emotion_list.append(df_img['Positive_Composite_Expressivity'].mean(axis = 0, skipna = True))
    emotion_list.append(df_img['Composite_Expressivity'].mean(axis = 0, skipna = True))
    f_emotion_list = [ '%.2f' % elem for elem in emotion_list ]
    return f_emotion_list

def face_attribute(df_pos_csv,df_neg_csv,df_all_csv):
    expr_list = []
    #Negative image list
    if isinstance(df_neg_csv, pd.DataFrame):
        #Validation check: All frames above .80
        df_neg_csv = df_neg_csv[df_neg_csv[' confidence']>=0.8]
        neg_em_list = img_emotion_list(df_neg_csv)
        expr_list.extend(neg_em_list)
    else:
        expr_list.extend(['']*3)
    
    #Positive image list
    if isinstance(df_pos_csv, pd.DataFrame):
        #Validation check: All frames above .80
        df_pos_csv = df_pos_csv[df_pos_csv[' confidence']>=0.8]
        pos_em_list = img_emotion_list(df_pos_csv)
        expr_list.extend(pos_em_list)
    else:
        expr_list.extend(['']*3)
    
    #All image list
    #Validation check: All frames above .80
    df_all_csv = df_all_csv[df_all_csv[' confidence']>=0.8]
    comp_em_list = img_emotion_list(df_all_csv)
    expr_list.extend(comp_em_list)
    return expr_list

def audio_attribute(df_all_csv,aud_att,variability,len_qid):
    expr_list = []
    if variability == 'var':
        ff_val = [float(i) for i in list(df_all_csv[aud_att])]
        expr_list.append(round(np.nanstd(ff_val),2))
    elif variability == 'silence':
        sil_list = []
        for i in range(len(len_qid)):
            sil_val = (len(len_qid[i][len_qid[i][aud_att] == 'no'])/105)*1000
            sil_list.append(sil_val)
        silence_mean = sum(sil_list)/len(len_qid)
        expr_list.append(round(silence_mean,2))
    elif variability == 'pause':
        pause_list = []
        for i in range(len(len_qid)):
            pause_val = (len(len_qid[i][len_qid[i][aud_att] == 'no'])/105)*1000
            pause_list.append(pause_val)
        pause_var = np.std(pause_list)
        expr_list.append(round(pause_var,2))
    else:
        expr_mean_val = df_all_csv[aud_att].mean(axis = 0, skipna = True)
        expr_list.append(round(expr_mean_val,4))
    return expr_list

def face_feature(of_list,face_em_list):
    expr_list = []
    if len(of_list[2])>0:
        if len(of_list[0])>0:
            df_pos_csv = df_combine(of_list[0])
        else:
            df_pos_csv = 'empty'
        if len(of_list[1])>0:
            df_neg_csv = df_combine(of_list[1])
        else:
            df_neg_csv = 'empty'
            
        df_all_csv = df_combine(of_list[2])
        expr_list = face_attribute(df_pos_csv,df_neg_csv,df_all_csv)
    else:
        expr_list = ['']*len(face_em_list)
    return expr_list

def audio_feature(aud_list,aud_att,variability,item_type):
    expr_list = []
    if len(aud_list[2])>0:
        df_all_csv = df_combine(aud_list[2])
        expr_list = audio_attribute(df_all_csv,aud_att,variability,aud_list[2])
        if item_type == 'VST':
            if len(aud_list[0]) > 0:
                df_pos_csv = df_combine(aud_list[0])
                pos_exp = audio_attribute(df_pos_csv,aud_att,variability,aud_list[2])
            else:
                pos_exp = ['']
            expr_list.extend(pos_exp)
            if len(aud_list[1]) > 0:
                df_neg_csv = df_combine(aud_list[1])
                neg_exp = audio_attribute(df_neg_csv,aud_att,variability,aud_list[2])
            else:
                neg_exp = ['']
            expr_list.extend(neg_exp)
    else:
        if item_type == 'VST':
            expr_list = ['']*3
        else:
            expr_list = ['']
    return expr_list

def ansa_feature(of_list,df_item):
    expr_list = []
    if len(of_list[2])>0:
        df_all_csv = df_combine(of_list[2])
        if df_item == 'prompt_count':
            exp_val = df_all_csv[df_item].sum(axis = 0, skipna = True)
        else:
            exp_val = df_all_csv[df_item].mean(axis = 0, skipna = True)
        expr_list = [round(exp_val,2)]
    else:
        expr_list = ['']
    return expr_list

def combine_feature_frames(out_dir, qnn_list, out_loc):
    of_all_list = []
    of_pos_list = []
    of_neg_list = []
    pos_list = ['pos','hap','ansa_q','_ahh']
    neg_list = ['neg','sad','ang','disg','surp','fear','ansa_q']
    for index, row in qnn_list.iterrows():
        if row['Question_Type'] == 'AV':
            qnn_path = row['VIDEOURI'].split('/')
            if len(qnn_path)>1:
                new_out_dir = out_dir + '/'.join(qnn_path[3:-1]) + out_loc
                qid_exp = glob.glob(new_out_dir)
                if len(qid_exp) > 0:
                    new_of_dir = glob.glob(new_out_dir + '/*.csv')
                    if len(new_of_dir)>0:
                        read_of_csv = pd.read_csv(new_of_dir[0])
                        if any(pos in row['AIQSTSCD'].lower() for pos in pos_list):
                            of_pos_list.append(read_of_csv)
                            of_all_list.append(read_of_csv)
                        elif any(neg in row['AIQSTSCD'].lower() for neg in neg_list):
                            of_neg_list.append(read_of_csv)
                            of_all_list.append(read_of_csv)

    return of_pos_list,of_neg_list,of_all_list

def ques_df(particpt_df,qnn_item,inst_type):
    new_particpt_df = particpt_df[particpt_df.VIDEOURI.str.match(qnn_item)]
    new_particpt_df = new_particpt_df[new_particpt_df.AIQSTSCD.str.match(inst_type)].reset_index(drop=True)
    return new_particpt_df

def ansa_rating(ansa_ans):
    ansa_score = 0
    if ansa_ans.lower() == 'often':
        ansa_score = 0
    elif ansa_ans.lower() == 'sometimes':
        ansa_score = 0.25
    elif ansa_ans.lower() == 'rarely':
        ansa_score = 0.50
    elif ansa_ans.lower() == 'almost never':
        ansa_score = 0.75
    elif ansa_ans.lower() == 'never':
        ansa_score = 1
    return ansa_score
        

def ansa_ques_score(qid_df,qid_code,score_type):
    q_score = ''
    qn_score = list(qid_df[qid_df['AIQSTSCD']==qid_code][score_type])
    if len(qn_score)>0:
        q_score = qn_score[0]
    if score_type == 'AIQSNUM' and (str(q_score) == '' or str(q_score) == 'nan'):
        q_score = 0
    return q_score

def ansa_item(qid_df):
    ansa_q9 = ansa_ques_score(qid_df,'ANSA_Q9','AIQSNUM')
    ansa_q8 = ansa_ques_score(qid_df,'ANSA_Q8','AIQSORES')
    ansa_q15 = ansa_ques_score(qid_df,'ANSA_Q15','AIQSNUM')
    ansa_q23 = ansa_ques_score(qid_df,'ANSA_Q23','AIQSNUM')
    ansa_q29 = ansa_ques_score(qid_df,'ANSA_Q29','AIQSNUM')
    ansa_q32 = ansa_ques_score(qid_df,'ANSA_Q32','AIQSNUM')
    ansa_q34 = ansa_ques_score(qid_df,'ANSA_Q34','AIQSORES')
    ansa_q40 = ansa_ques_score(qid_df,'ANSA_Q40','AIQSORES')
    ansa_q49 = ansa_ques_score(qid_df,'ANSA_Q49','AIQSORES')
    ansa_q41 = ansa_ques_score(qid_df,'ANSA_Q41','AIQSORES')
    ansa_q46 = ansa_ques_score(qid_df,'ANSA_Q46','AIQSNUM')
    ansa_q44 = ansa_ques_score(qid_df,'ANSA_Q44','AIQSORES')
    ansa_q51 = ansa_ques_score(qid_df,'ANSA_Q51','AIQSORES')
    item_1r = 10 - int(ansa_q9)
    item_2p = 0 if str(ansa_q8) in ANSA_Q8 else 1
    item_3r = ansa_q15
    item_4r = (int(ansa_q29)/10) + (int(ansa_q32)/10) + 0 if str(ansa_q34) in ANSA_Q34 else 1
    item_5r = ansa_rating(str(ansa_q40)) + ansa_rating(str(ansa_q49))
    item_6r = (0 if str(ansa_q41) in ANSA_Q41 else 1) + ((10-int(ansa_q46))/10)
    item_7p = (0 if str(ansa_q44) in ANSA_44_51 else 1) + (0 if str(ansa_q51) in ANSA_44_51 else 1)
    ansa_score = [item_1r,item_2p,item_3r,item_4r,item_5r,item_6r,item_7p]
    return ansa_score

def head_vel(of_results):
    distance_list = []
    final_dist_list = []
    movement_velocity = 0
    for index, row in of_results.iterrows():
        if index > 0:
            point_x = (of_results[' pose_Tx'][index-1], of_results[' pose_Ty'][index-1], of_results[' pose_Tz'][index-1])
            point_y = (row[' pose_Tx'],row[' pose_Ty'],row[' pose_Tz'])
            try:
                dst = distance.euclidean(point_x, point_y)
            except:
                dst = 0
            distance_list.append(dst)
    for i in range(len(distance_list)):
        if abs(distance_list[i])<200:
            final_dist_list.append(distance_list[i])
    dist_mm = sum(final_dist_list)
    if len(of_results)>0:
        movement_velocity = dist_mm/len(of_results)
    return movement_velocity

def head_movement(of_list):
    motion_list = []
    if len(of_list[2])>0:
        df_all_csv = df_combine(of_list[2])
        motion_val = head_vel(df_all_csv)
        
        motion_list = [round(motion_val,2)]
    else:
        motion_list = ['']
    return motion_list

def audio_f_value(out_dir, qid_df,item_type):
    
    #Audio feature list : VFS
    vst_vfs_list = combine_feature_frames(out_dir, qid_df, VFS_OUT_LOC)
    vfs_list = audio_feature(vst_vfs_list,'voice_percentage','',item_type)
    
    #Audio feature list : Audio Intensity
    vst_ai_list = combine_feature_frames(out_dir, qid_df, AI_OUT_LOC)
    ai_list = audio_feature(vst_ai_list,'Intensity','var',item_type)
    vfs_list.extend(ai_list)

    #Audio feature list : silence mean
    vst_pitch_list = combine_feature_frames(out_dir, qid_df, FF_OUT_LOC)
    sil_list = audio_feature(vst_pitch_list,'voice_label','silence',item_type)
    vfs_list.extend(sil_list)

    #Audio feature list : Variable Pause Length
    pause_list = audio_feature(vst_pitch_list,'voice_label','pause',item_type)
    vfs_list.extend(pause_list)
    
    #Audio feature list : Pitch Variability 
    pitch_list = audio_feature(vst_pitch_list,'fundamental_freq','var',item_type)
    vfs_list.extend(pitch_list)
    
    #Audio feature list : NAQ
    vst_naq_list = combine_feature_frames(out_dir, qid_df, NAQ_OUT_LOC)
    naq_list = audio_feature(vst_naq_list,'NAQ_Amp','',item_type)
    vfs_list.extend(naq_list)
    
    return vfs_list

def cssp_f_values(out_dir, qid_df,item_type):
    #Audio feature list : Formant Variability
    vst_formant_list = combine_feature_frames(out_dir, qid_df, FORMANT_OUT_LOC)
    formant_list = audio_feature(vst_formant_list,'F_diff','var',item_type)
    
    return formant_list

def ansa_f_values(out_dir, qid_df,item_type):
    #ANSA Score
    ansa_score = ansa_item(qid_df)

    #Face feature list
    ansa_face_list = combine_feature_frames(out_dir, qid_df, FACE_OUT_LOC)
    em_list = ansa_feature(ansa_face_list,'Composite_Expressivity')
    ansa_score.extend(em_list)

    #Audio Expressivity pending
    #VFS Score 
    ansa_vfs_list = combine_feature_frames(out_dir, qid_df, VFS_OUT_LOC)
    vfs_list = audio_feature(ansa_vfs_list,'voice_percentage','',item_type)
    ansa_score.extend(vfs_list)

    #Motion
    ansa_motion_list = combine_feature_frames(out_dir, qid_df, MOTION_OUT_LOC)
    motion_list = head_movement(ansa_motion_list)
    ansa_score.extend(motion_list)

    #Prompt Count 
    ansa_prompt_list = combine_feature_frames(out_dir, qid_df, PROMPT_OUT_LOC)
    prompt_num = ansa_feature(ansa_prompt_list,'prompt_count')
    ansa_score.extend(prompt_num)
    
    return ansa_score

def gen_data_non_ema(qid_df):
    info_list = []
    for index, row in qid_df.iterrows():
        info_list.append(row['STUDYID'])
        info_list.append(row['SITEID'])
        info_list.append(row['SUBJID'])
        info_list.append(row['AIQSDAT'])
        info_list.append(row['AIQSTIM'])
        info_list.append(row['AIQTZ'])
        info_list.append(row['AIQSLAN'])
        info_list.append(row['QNNUUID'])
        return info_list
    
def mean_list(base_val):
    #removing string from list
    while("" in base_val) : 
        base_val.remove("") 
    mean_item = np.mean(base_val)
    std_item = np.std(base_val)
    if str(mean_item) == 'nan' or str(mean_item) == '':
        mean_item = 0
    if str(std_item) == 'nan' or str(std_item) == '':
        std_item = 1
    return mean_item,std_item

def vei_intensity(qid_df,base_list,comm_item_list):
    mean_plv = mean_list(base_list[0])
    mean_pv = mean_list(base_list[1])
    mean_naq = mean_list(base_list[2])
    mean_fm = mean_list(base_list[3])
    PLV_zscore =  (pd.to_numeric(qid_df[comm_item_list[0]], errors='coerce') - mean_plv[0])/mean_plv[1]
    PV_zscore = (pd.to_numeric(qid_df[comm_item_list[1]], errors='coerce') - mean_pv[0])/mean_pv[1]
    NAQ_zscore = (pd.to_numeric(qid_df[comm_item_list[2]], errors='coerce') - mean_naq[0])/mean_naq[1]
    FM_zscore = (pd.to_numeric(qid_df[comm_item_list[3]], errors='coerce') - float(mean_fm[0]))/float(mean_fm[1])
    new_plv = PLV_zscore.fillna(0)
    new_pv = PV_zscore.fillna(0)
    new_naq = NAQ_zscore.fillna(0)
    new_fm = FM_zscore.fillna(0)
    VEI = new_plv + new_pv + new_naq + new_fm
    return VEI
