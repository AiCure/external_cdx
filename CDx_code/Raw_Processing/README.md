# CDx Feature Extraction Code

1. Audio:<br />
*		audio_features.py : Method defination for Audio features<br />
				a.) Formant Freq: <br />
						Controller Method: *formant_score()*<br />
				b.) Fundamental Freq:<br />
						Controller Method: *audio_pitch()*<br />
				c.) Glottal to Noise Ratio:<br />
						Controller Method: *gne_ratio()*<br />
				d.) Harmonic Noise Ratio:<br />
						Controller Method: *hnr_ratio()*<br />
				e.) Audio Intensity:<br />
						Controller Method: *intensity_score()*<br />
				f.) Normalized Amplitude Quotient:<br />
						Controller Method: *get_pulse_amplitude()*<br />
				g.) Voice Frame Score:<br />
						Controller Method: *audio_vfs_val()*<br />

2. Facial:<br />
*		video_util.py : Computing Facial expressivity based on Action Unit at frame level<br />
						Controller Method: *calc_of_for_video()*
      
3. Content/NLP : Using AWS Speech to Text Transcribe API to extract content from Audio. Based on answer, calculating Audio Intent.<br />
*				a.) Speech to Text: <br />
				 		Controller Method: *collect_content_result()*<br />
		 		b.) Rate of Speech:<br />
		 		 		Controller Method: *ros_speech()*<br />
 		 		c.) Word Repetition:<br />
 		 		 		Controller Method: *word_percent()*
