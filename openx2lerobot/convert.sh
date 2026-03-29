openx_rlds.py \
    --raw-dir /mnt/data/data/open/RT-X/tensorflow_datasets/droid/1.0.0 \
    --local-dir /mnt/data/data/yyh/droid_1.0.0_lerobot \
    --robot-type franka \
    --use-videos \
    --lang-ann ../info/droid_language_annotations.json \
    --keep-ranges ../info/keep_ranges_1_0_1.json

# CUDA_VISIBLE_DEVICES="" python openx_rlds.py \
#     --raw-dir /mnt/data/data/open/RT-X/tensorflow_datasets/language_table/0.1.0 \
#     --local-dir /data/nfs2/yyh/gr00t_dataset/language_table_0.1.0_lerobot \
#     --use-videos \
#     --robot-type xarm \

# CUDA_VISIBLE_DEVICES="" python openx_rlds.py \
#     --raw-dir /mnt/data/data/open/RT-X/tensorflow_datasets/fractal20220817_data/0.1.0/ \
#     --local-dir /tmp/fractal20220817_data_0.1.0_lerobot \
#     --use-videos \
#     --robot-type google_robot \

# CUDA_VISIBLE_DEVICES="" python openx_rlds.py \
#     --raw-dir /mnt/data/data/open/RT-X/tensorflow_datasets/bridge_dataset/0.1.0 \
#     --local-dir /tmp/bridge_dataset_0.1.0_lerobot \
#     --use-videos \
#     --robot-type widowx \
