DATA_DIR=yelp

MODEL_PATH=path


CUDA_VISIBLE_DEVICES=0 python3 classifier/save_model.py \
    --model_name CNNClassifier \
    --data_dir data/$DATA_DIR \
    --output_dir classifier/pretrained/cnn \
    --model_path $MODEL_PATH \
    --vocab_file_name vocab \
    --emb_size 300 \
    --enc_num_layers 1 \
    --hidden_size 512 \
    --optim adam \
    --mode test \
    --num_workers 0 \
    --overwrite_output_dir \
    --enc_bidirectional \
    --num_kernels_each_size 128 \
