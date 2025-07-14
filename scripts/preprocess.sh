python preprocess/preprocess.py \
    --data_dir 'greatest_hits' \
    --video_sample_rate 15 \
    --audio_sample_rate 48000 \
    --clip_duration 5.0 \
    --image_size 112 112 \
    --rms_nframes 512 \
    --rms_hop 128 \
    --rms_scale_factor 2.0 \
    --prompt_mode 'list'