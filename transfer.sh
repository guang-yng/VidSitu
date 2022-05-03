#!/bin/bash
# $1 yt_id['vid_id']
# $2 out_file
# $3 yt_id['start']


if [ -f ./tmp_data/$1.mp4 ]; then
    ffmpeg -ss $3 -i ./tmp_data/$1.mp4 -to 10 $2
fi

