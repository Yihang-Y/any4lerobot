export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0
python agibot_fast.py \
    --src-path /data/nfs2/yyh/gr00t_dataset/AgiBot-World/OpenDriveLab___AgiBot-World/raw/main \
    --output-path /data/nfs2/yyh/gr00t_dataset/AgiBot-World-alpha-lerobot \
    --eef-type gripper \
    --cpus-per-task 8 \
    --staging-dir /tmp \
    --max-tar-readers 4
