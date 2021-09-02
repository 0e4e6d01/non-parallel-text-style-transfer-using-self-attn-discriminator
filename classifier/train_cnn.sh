DATA_DIR=data/yelp


CUDA_VISIBLE_DEVICES=0 python classifier/run.py \
    --model_name CNNClassifier \
    --data_dir $DATA_DIR \
    --output_dir classifier/exp \
    --vocab_file_name vocab \
    --emb_size 300 \
    --hidden_size 512 \
    --lr 1e-3 \
    --batch_size 128 \
    --num_train_epochs 10 \
    --decay_epoch 0 \
    --shuffle \
    --log_interval 100 \
    --eval_interval 1000 \
    --max_grad_norm 1.0 \
    --dropout 0.5 \
    --optim adam \
    --mode train \
    --num_workers 0 \
    --overwrite_output_dir \
    --enc_bidirectional \
    --use_sos \
    --use_eos \
    --num_kernels_each_size 128 \
    --kernel_sizes 2 3 4 \
    
    # --init kaiming_normal \

