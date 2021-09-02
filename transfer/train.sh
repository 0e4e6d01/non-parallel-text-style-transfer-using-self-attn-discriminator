DATASET=yelp
CLS_MODEL_PATH=path

CUDA_VISIBLE_DEVICES=0 python transfer/run.py \
    --model_name TransferModel \
    --data_dir data/$DATASET \
    --output_dir transfer/exp \
    --cls_weight 0.1 \
    --ca_weight 0.01 \
    --bt_weight 1.0 \
    --vocab_file_name vocab \
    --emb_size 300 \
    --cell LSTM \
    --enc_bidirectional \
    --enc_num_layers 1 \
    --dec_num_layers 1 \
    --hidden_size 512 \
    --lr 1e-4 \
    --batch_size 32 \
    --grad_accum_interval 1 \
    --num_train_epochs 10 \
    --decay_epoch 0 \
    --shuffle \
    --log_interval 100 \
    --eval_interval 1500 \
    --max_grad_norm 5.0 \
    --dropout 0.3 \
    --optim adam \
    --mode train \
    --num_workers 0 \
    --overwrite_output_dir \
    --max_decoding_len 20 \
    --beam_size 1 \
    --use_sos \
    --use_eos \
    --drop_prob 0.1 \
    --shuffle_weight 2.0 \
    --num_kernels_each_size 128 \
    --test_ref_dir references/$DATASET/test \
    --cnn_clf_path classifier/pretrained/cnn/classifier.pt \
    --cls_model_path $CLS_MODEL_PATH \