CUDA_VISIBLE_DEVICES=0,3 python vidsitu_code/feat_extractor.py --mdl_resume_path='tmp/models/timesformer_event_contrastive_3e-5.pth' \
	--mdl_name_used='timesformer_event_contrastive_3e-5' --mdl.mdl_name='timesformer_event_contrastive' --is_cu=False
