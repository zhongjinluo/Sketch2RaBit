CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
--exp_dir=/program/huawei/something/pSp/out/render_masked \
--checkpoint_path=/program/huawei/something/pSp/out/render_masked/checkpoints/best_model_render_masked.pt  \
--data_path=/program/huawei/something/datasets/render_masked_test \
--test_batch_size=2 \
--test_workers=2 \
--couple_outputs 

# CUDA_VISIBLE_DEVICES=1 python demo.py \
# --exp_dir=/home/jinguodong/data/back_uv/aug/pSp/test/render2back_512_show_1/ \
# --checkpoint_path=/home/jinguodong/data/back_uv/aug/pSp/render2back_512_4592_2/checkpoints/best_model.pt  \
# --data_path=/home/jinguodong/data/final_data/render/test_show \
# --test_mesh_path=/home/jinguodong/data/raw/obj_1650/  \
# --test_batch_size=1 \
# --test_workers=1 \
# --couple_outputs 
