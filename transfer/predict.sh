DATASET=yelp

MODEL_PATH=path

CUDA_VISIBLE_DEVICES=0 python transfer/run.py \
    --model_name TransferModel \
    --data_dir data/$DATASET \
    --test_ref_dir references/$DATASET/test \
    --output_dir $MODEL_PATH \
    --model_path $MODEL_PATH \
    --vocab_file_name vocab \
    --emb_size 300 \
    --cell LSTM \
    --enc_bidirectional \
    --enc_num_layers 1 \
    --dec_num_layers 1 \
    --hidden_size 512 \
    --mode test \
    --num_workers 2 \
    --overwrite_output_dir \
    --max_decoding_len 20 \
    --beam_size 1 \
    --cnn_clf_path classifier/pretrained/cnn/classifier.pt \


    # --lower_case \
