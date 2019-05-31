# CDx Feature Extraction Code

1. Audio:
		audio_features.py : Method defination for Audio features
				a.) Formant Freq: 
						Controller Method: *formant_score()*
				b.) Fundamental Freq:
						Controller Method: *audio_pitch()*
				c.) Glottal to Noise Ratio:
						Controller Method: *gne_ratio()*
				d.) Harmonic Noise Ratio:
						Controller Method: *hnr_ratio()*
				e.) Audio Intensity:
						Controller Method: *intensity_score()*
				f.) Normalized Amplitude Quotient:
						Controller Method: *get_pulse_amplitude()*
				g.) Voice Frame Score:
						Controller Method: *audio_vfs_val()*

2. Facial:
		video_util.py : Computing Facial expressivity based on Action Unit at frame level
						Controller Method: *calc_of_for_video()*
      
3. Content/NLP : Using AWS Speech to Text Transcribe API to extract content from Audio. Based on answer, calculating Audio Intent.
				a.) Speech to Text: 
				 		Controller Method: *collect_content_result()*
		 		b.) Rate of Speech:
		 		 		Controller Method: *ros_speech()*
 		 		c.) Word Repetition:
 		 		 		Controller Method: *word_percent()*