# SLOWFAST
# CUDA_VISIBLE_DEVICES=6 python main_dist.py "vb" --mdl.mdl_name="sf_base" --train.bs=4 --debug_mode=False

# SLOWFAST vb_arg
CUDA_VISIBLE_DEVICES=2,3 python feat_extractor.py "tmp/models/vb.pth" "vb" --task_type="vb" --mdl.mdl_name="sf_base" --train.bs=8

# SLOWFAST Pretrained
# CUDA_VISIBLE_DEVICES=6,7 python main_dist.py "vb" --mdl.mdl_name="sf_base" --train.bs=8 --mdl.load_sf_pretrained=True --misc.tmp_path="./tmp-pretrained"

# SLOWFAST Pretrained DEBUG
# CUDA_VISIBLE_DEVICES=6 python main_dist.py "vb" --mdl.mdl_name="sf_base" --train.bs=4 --debug_mode=True --mdl.load_sf_pretrained=True

# SLOWFAST vb eval
# CUDA_VISIBLE_DEVICES=7 python vidsitu_code/evl_fns.py --pred_file='./tmp/predictions/vb/valid_0.pkl' \
#  --split_file_path="./data/vidsitu_annotations/split_files/vseg_split_valid_lb.json" \
#  --vinfo_file_path="./data/vidsitu_annotations/vinfo_files/vinfo_valid_lb.json" \
#  --vsitu_ann_file_path="./data/vidsitu_annotations/vseg_ann_files/vsann_valid_lb.json" \
#  --split_type='valid' --task_type="vb"
