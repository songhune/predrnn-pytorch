export CUDA_VISIBLE_DEVICES=0
cd ..
python3 -u run.py \
    --is_training 1 \
    --device cuda:0 \
    --dataset_name mnist \
    --train_data_paths /workspace/data/data/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths /workspace/data/data/moving-mnist-example/moving-mnist-valid.npz \
    --save_dir /workspace/checkpoints/mnist_predrnn \
    --gen_frm_dir /workspace/results/mnist_predrnn \
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 64,64,64,64 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.0003 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000