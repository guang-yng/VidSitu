#!/bin/bash
# $1 yt_id['vid_id']
# $2 out_file
# $3 yt_id['start']
# $4 yt_id["vid_seg_id"]


if [ ! -f /data/private/yangguang/VidSitu/all_data/vsitu_video_trimmed_dir/$4.mp4 ]; then
	echo $4 >> missing_vids.txt
fi

