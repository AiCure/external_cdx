# CDx Takeda Final Feature Extraction

Driver Class: **takeda_processing.py**
Driver Controller: **process_data_export()**

1. VST:<br />
*		Controller Method: *vst_processing()*<br />
				POS_COMP_EXP_POSIMG, NEG_COMP_EXP_POSIMG, NET_COMP_EXP_POSIMG, COMP_EXP_POSIMG, POS_COMP_EXP_NEGIMG, NEG_COMP_EXP_NEGIMG, NET_COMP_EXP_NEGIMG, COMP_EXP_NEGIMG,	POS_COMP_EXP_NETIMG, NEG_COMP_EXP_NETIMG, NET_COMP_EXP_NETIMG, COMP_EXP_NETIMG,	
				POS_COMP_EXP_ALLIMG, NEG_COMP_EXP_ALLIMG, NEU_COMP_EXP_ALLIMG, COMP_EXP_ALLIMG
				VERBAL_EXP_POSIMG, VERBAL_EXP_NEGIMG, VERBAL_EXP_NEUIMG, VERBAL_EXP_ALLIMG,
				VFS_SCORE, SILENCE_LENGTH_MEAN, PAUSE_LENGTH_VAR, NORMALIZED_AMPLITUDE_QUOT,
				FORMANT_VARIABILITY, INTENSITY_VARIABILITY


2. ANSA:<br />
*		Controller Method: *fee_processing()*<br />
				ANSA Score, FEI, VEI, VFS_SCORE, MOTION, NO_PROMPTS
      
3. FEE : <br />
*		Controller Method: *ansa_processing()*<br />
				NEG_COMP_EXP_POSEMO, POS_COMP_EXP_POSEMO, COMP_EXP_POSEMO, NEG_COMP_EXP_NEGEMO,
				POS_COMP_EXP_NEGEMO, COMP_EXP_NEGEMO, NEG_COMP_EXP_ALLEMO, POS_COMP_EXP_ALLEMO
				COMP_EXP_ALLEMO

4. EMA : <br />
*		Controller Method: *ema_processing()*<br />
				All the instruction type questions answered by participants.
