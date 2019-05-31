import configparser
import pandas as pd
import numpy as np

import sys
sys.path

sys.path.append('./././aicurelib')
from aicurelib.ailogging.logger import create_logger
from data_processing.Takeda.util.process_util import combine_feature_frames,face_feature,audio_f_value,cssp_f_values,excel_parsing
from data_processing.Takeda.util.process_util import instrument_qnn,ques_df,av_check,gen_data_non_ema,ansa_f_values,df_combine,vei_intensity

#Accessing data from processing confriguration file
config_proc_Parser = configparser.RawConfigParser()
config_proc_filePath = r'./data_processing/process_conf.txt'
config_proc_Parser.read(config_proc_filePath)
DATA_EXPORT_COL = config_proc_Parser.get('proc_config', 'data_export_col').split(',')
FACE_EM_LIST = config_proc_Parser.get('proc_config', 'face_emotion_feature').split(',')
AUD_EM_LIST = config_proc_Parser.get('proc_config', 'audio_feature').split(',')
VST_FINAL_COL = config_proc_Parser.get('proc_config', 'vst_final_col').split(',')
FEE_FINAL_COL = config_proc_Parser.get('proc_config', 'fee_final_col').split(',')
ANSA_FINAL_COL = config_proc_Parser.get('proc_config', 'ansa_final_col').split(',')
EMA_FINAL_COL = config_proc_Parser.get('proc_config', 'ema_final_col').split(',')
VFS_DROP_LIST = config_proc_Parser.get('proc_config', 'vfs_drop_list').split(',')
ANSA_DROP_LIST = config_proc_Parser.get('proc_config', 'ansa_drop_list').split(',')

logger = create_logger(__name__)

#Instrument Indicators
VST_IMG = "VST_IMG"
CSSP_IMG = 'CSSP_AHH'
FEE_IMG = "FEE_"
ANSA_IMG = "ANSA_Q"
EMA_IMG = "EMA_Q"
FACE_OUT_LOC = '/video/face_expressivity'

#Methods for instruments
#1. VST
def vst_processing(particpt_df,out_dir,qnn_api):
    #preparing qnn dataframe
    inst_list,base_qnn_list = instrument_qnn(particpt_df,VST_IMG)
    feature_list = []
    base_plv = []
    base_pv = []
    base_naq = []
    base_fm = []
    for qnn in inst_list:
        try:
            logger.info('Processing {} for VST Instrument'.format(qnn))
            #Preparing dataframe
            q_qid_df = ques_df(particpt_df,qnn,VST_IMG)
            qid_df = av_check(q_qid_df,qnn,qnn_api)
            #CSSP dataframe
            cssp_qid_df = ques_df(particpt_df,qnn,CSSP_IMG)
            cssp_df = av_check(cssp_qid_df,qnn,qnn_api)
            cssp_df['Question_Type'] = 'AV'

            #general export
            gen_export = gen_data_non_ema(qid_df)

            #Face feature list
            vst_face_list = combine_feature_frames(out_dir, qid_df, FACE_OUT_LOC)
            em_list = face_feature(vst_face_list,FACE_EM_LIST)
            gen_export.extend(em_list)

            #Audio feature list
            aud_feature = audio_f_value(out_dir, qid_df,'VST')
            gen_export.extend(aud_feature)

            #CSSP feature list
            cssp_feature = cssp_f_values(out_dir, cssp_df,'VST')
            gen_export.extend(cssp_feature)

            #base value
            if qnn in base_qnn_list:
                base_plv.append(aud_feature[9])
                base_pv.append(aud_feature[12])
                base_naq.append(aud_feature[15])
                base_fm.append(cssp_feature[0])

            #final list
            feature_list.append(gen_export)
        except Exception as e:
            logger.error('Failed to process VST instrument {} for {}'.format(e,qnn))
            continue
    base_list = [base_plv,base_pv,base_naq,base_fm]
    return feature_list,base_list

#2. FEE
def fee_processing(particpt_df,out_dir,qnn_api):
    #preparing qnn dataframe
    inst_list,base_qnn_list = instrument_qnn(particpt_df,FEE_IMG)
    feature_list = []
    for qnn in inst_list:
        try:
            logger.info('Processing {} for FEE Instrument'.format(qnn))
            #Preparing dataframe
            q_qid_df = ques_df(particpt_df,qnn,FEE_IMG)
            qid_df = av_check(q_qid_df,qnn,qnn_api)
            #Modifying Question type to use common method:As FEE question types are only 'V'
            qid_df['Question_Type'] = 'AV'

            #general export
            gen_export = gen_data_non_ema(qid_df)

            #Face feature list
            fee_face_list = combine_feature_frames(out_dir, qid_df, FACE_OUT_LOC)
            em_list = face_feature(fee_face_list,FACE_EM_LIST)
            gen_export.extend(em_list)

            #final list
            feature_list.append(gen_export)
        except Exception as e:
            logger.error('Failed to process FEE instrument {} for {}'.format(e,qnn))
            continue
    return feature_list

#3. ANSA
def ansa_processing(particpt_df,out_dir,qnn_api):
    #preparing qnn dataframe
    inst_list,base_qnn_list = instrument_qnn(particpt_df,ANSA_IMG)
    feature_list = []
    base_plv = []
    base_pv = []
    base_naq = []
    base_fm = []
    for qnn in inst_list:
        try:
            logger.info('Processing {} for ANSA Instrument'.format(qnn))
            #Preparing dataframe
            q_qid_df = ques_df(particpt_df,qnn,ANSA_IMG)
            qid_df = av_check(q_qid_df,qnn,qnn_api)

            #general export
            gen_export = gen_data_non_ema(qid_df)

            #ANSA Features
            ansa_score = ansa_f_values(out_dir, qid_df,'ANSA')
            gen_export.extend(ansa_score)

            #Audio feature list
            aud_feature = audio_f_value(out_dir, qid_df,'ANSA')
            gen_export.extend(aud_feature)
            cssp_feature = cssp_f_values(out_dir, qid_df,'ANSA')
            gen_export.extend(cssp_feature)

            #base value
            if qnn in base_qnn_list:
                base_plv.append(aud_feature[3])
                base_pv.append(aud_feature[4])
                base_naq.append(aud_feature[5])
                base_fm.append(cssp_feature[0])
            #final list
            feature_list.append(gen_export)
        except Exception as e:
            logger.error('Failed to process ANSA instrument {} for {}'.format(e,qnn))
            continue
    base_list = [base_plv,base_pv,base_naq,base_fm]
    return feature_list,base_list

#4. EMA
def ema_processing(particpt_df,out_dir,qnn_api):
    #preparing qnn dataframe
    inst_list,base_qnn_list = instrument_qnn(particpt_df,EMA_IMG)
    df_list = []
    for qnn in inst_list:
        try:
            logger.info('Processing {} for EMA Instrument'.format(qnn))
            #Preparing dataframe
            q_qid_df = ques_df(particpt_df,qnn,EMA_IMG)
            qid_df = av_check(q_qid_df,qnn,qnn_api)
            df_list.append(qid_df)
        except Exception as e:
            logger.error('Failed to process EMA instrument {} for {}'.format(e,qnn))
            continue
    if len(df_list)>0:
        new_particpt_df = df_combine(df_list)
        ema_particpt_df = new_particpt_df[['STUDYID', 'SITEID', 'SUBJID','AIQSDAT', 'AIQSTIM', 'AIQTZ',
                                      'AIQSLAN','AIQSTSCD','QNNUUID','QNID','AIQSTEST','AIQSORES',
                                      'AIQSNUM','Question_Type']]
        ema_particpt_df['AIQSNUM'] = np.where((ema_particpt_df.AIQSTSCD=="EMA_Q4") & (ema_particpt_df.AIQSORES!='None of these'), 1,ema_particpt_df.AIQSNUM)
        ema_particpt_df['AIQSNUM'] = np.where((ema_particpt_df.AIQSTSCD=="EMA_Q4") & (ema_particpt_df.AIQSORES=='None of these'), 0,ema_particpt_df.AIQSNUM)
        ema_particpt_df['AIQSNUM'] = np.where((ema_particpt_df.AIQSTSCD=="EMA_Q5") & (ema_particpt_df.AIQSORES!='No'), 1,ema_particpt_df.AIQSNUM)
        ema_particpt_df['AIQSNUM'] = np.where((ema_particpt_df.AIQSTSCD=="EMA_Q5") & (ema_particpt_df.AIQSORES=='No'), 0,ema_particpt_df.AIQSNUM)
    else:
        return ""
    return ema_particpt_df

def process_data_export(out_dir,excel_path,qnn_api,final_out_base_dir):
    data_expt_col = DATA_EXPORT_COL
    logger.info('Processing started.......')
    #calling participant dataframe
    participant_df = excel_parsing(excel_path)
    participant_df = participant_df[data_expt_col]
    comm_item_list = ['PAUSE_LENGTH_VAR','PITCH_VAR','NORMALIZED_AMPLITUDE_QUOT','FORMANT_VARIABILITY']
    #VST Dataframe
    vst_list,base_list_vst = vst_processing(participant_df,out_dir,qnn_api)
    vst_df = pd.DataFrame(vst_list, columns = VST_FINAL_COL)
    neg_item_list = ['PL_NEG','PV_NEG','NAQ_NEG','FM_NEG']
    pos_item_list = ['PL_POS','PV_POS','NAQ_POS','FM_POS']
    vst_df['VERBAL_EXP_POSIMG'] = vei_intensity(vst_df,base_list_vst,pos_item_list)
    vst_df['VERBAL_EXP_NEGIMG'] = vei_intensity(vst_df,base_list_vst,neg_item_list)
    vst_df['VERBAL_EXP_ALLIMG'] = vei_intensity(vst_df,base_list_vst,comm_item_list)
    vst_df.drop(VFS_DROP_LIST, axis=1, inplace=True)
    #vst_df.head(20).to_csv('VST_Takeda.csv', index=False)

    #FEE Dataframe
    fee_list = fee_processing(participant_df,out_dir,qnn_api)
    fee_df = pd.DataFrame(fee_list, columns = FEE_FINAL_COL)
    #fee_df.to_csv('FEE_Takeda.csv', index=False)
    
    #ANSA Dataframe
    ansa_list,base_list = ansa_processing(participant_df,out_dir,qnn_api)
    ansa_df = pd.DataFrame(ansa_list, columns = ANSA_FINAL_COL)
    ansa_df['VEI'] = vei_intensity(ansa_df,base_list,comm_item_list)
    ansa_df.drop(ANSA_DROP_LIST, axis=1, inplace=True)
    #ansa_df.head(20).to_csv('ANSA_Takeda.csv', index=False)
    
    #EMA Dataframes
    ema_list = ema_processing(participant_df,out_dir,qnn_api)
    #ema_list.head(20).to_csv('EMA_Takeda.csv', index=False)
    
    #Storing data to excel
    writer = pd.ExcelWriter(final_out_base_dir+'Takeda_final_data.xlsx', engine='xlsxwriter')
    vst_df.to_excel(writer, sheet_name='VST',index=False)
    fee_df.to_excel(writer, sheet_name='FEE',index=False)
    ansa_df.to_excel(writer, sheet_name='ANSA',index=False)
    if isinstance(ema_df, pd.DataFrame):
        ema_list.to_excel(writer, sheet_name='EMA',index=False)
    writer.save()
    logger.info("Data processed successfully......")
