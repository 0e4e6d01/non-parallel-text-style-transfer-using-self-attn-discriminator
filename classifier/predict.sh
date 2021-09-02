DATA_DIR=data/yelp

MODEL_PATH=path

CUDA_VISIBLE_DEVICES=0 python classifier/run.py \
    --model_name CNNClassifier \
    --data_dir $DATA_DIR \
    --output_dir $MODEL_PATH \
    --model_path $MODEL_PATH \
    --vocab_file_name vocab \
    --emb_size 300 \
    --hidden_size 512 \
    --mode test \
    --num_workers 2 \
    --overwrite_output_dir \
    --enc_bidirectional \
    --use_sos \
    --use_eos \
    --batch_size 128 \
    --num_kernels_each_size 128 \
    
