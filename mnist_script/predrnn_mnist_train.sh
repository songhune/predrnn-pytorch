export CUDA_VISIBLE_DEVICES=2
cd ..
python3 -u run.py \
    --is_training 1 \
    --device cuda:0 \
    --dataset_name mnist \
<<<<<<< HEAD
    --train_data_paths /data/songhune/data/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths /data/songhune/data/moving-mnist-example/moving-mnist-valid.npz \
    --save_dir /data/songhune/checkpoints/mnist_predrnn256 \
    --gen_frm_dir /data/songhune/results/mnist_predrnn256 \
=======
    --train_data_paths /workspace/data/data/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths /workspace/data/data/moving-mnist-example/moving-mnist-valid.npz \
    --save_dir /workspace/checkpoints/mnist_predrnn \
    --gen_frm_dir /workspace/results/mnist_predrnn \
>>>>>>> 0ba085fd97d81dd32261db69465a76c63ec22770
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 256,256,256,256 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 1 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 0 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 5000 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.001 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    --num_save_samples 10 \
    --n_gpu 1 \
    --visual 0 \
    --visual_path ./decoupling_visual \
    --injection_action concat \
    --conv_on_input 0 \
    --res_on_conv 0 \
    --num_action_ch 4