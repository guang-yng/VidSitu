# SLOWFAST
# CUDA_VISIBLE_DEVICES=4,5 python main_dist.py "slowfast" --mdl.mdl_name="sf_base" \
#  --train.bs=8 --address=9999 

# SLOWFAST vb_arg
# CUDA_VISIBLE_DEVICES=2,3 python feat_extractor.py "tmp/models/vb.pth" "vb" --task_type="vb" --mdl.mdl_name="sf_base" --train.bs=8

# SLOWFAST Pretrained
# CUDA_VISIBLE_DEVICES=0,1 python main_dist.py "slowfast_pretrained" --mdl.mdl_name="sf_base" --train.bs=4 --only_val=True\
#	--train.gradient_accumulation=2 --train.bsv=8 --train.resume=True --train.resume_path="tmp/models/slowfast_pretrained.pth"

# SLOWFAST Pretrained DEBUG
# CUDA_VISIBLE_DEVICES=6 python main_dist.py "vb" --mdl.mdl_name="sf_base" --train.bs=4 --debug_mode=True --mdl.load_sf_pretrained=True

# SLOWFAST vb eval
# CUDA_VISIBLE_DEVICES=7 python vidsitu_code/evl_fns.py --pred_file='./tmp/predictions/vb/valid_0.pkl' \
#  --split_file_path="./data/vidsitu_annotations/split_files/vseg_split_valid_lb.json" \
#  --vinfo_file_path="./data/vidsitu_annotations/vinfo_files/vinfo_valid_lb.json" \
#  --vsitu_ann_file_path="./data/vidsitu_annotations/vseg_ann_files/vsann_valid_lb.json" \
#  --split_type='valid' --task_type="vb"

# TimeSformer Pretrained
CUDA_VISIBLE_DEVICES=0,1 python main_dist.py "timesformer" --mdl.mdl_name="timesformer" --train.bs=4 --only_val=True\
	--train.gradient_accumulation=2 --train.bsv=8 --train.resume=True --train.resume_path="tmp/models/timesformer.pth"

# SLOWFAST Contrastive
# CUDA_VISIBLE_DEVICES=6,3 python main_dist.py "slowfast_contrastive" --mdl.mdl_name="sf_base_contrastive" \
#  --train.bs=8 --train.gradient_accumulation=2

# SLOWFAST Pretrained Contrastive
# CUDA_VISIBLE_DEVICES=6,7 python main_dist.py "slowfast_pretrained_contrastive" \
#    --mdl.mdl_name="sf_base" --train.bs=8 --mdl.load_sf_pretrained=True  \
#    --train.gradien_accumulation=2
