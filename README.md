# Non-Parallel Text Style Transfer
These are codes for our method.

## Requirements
- Python3
- Pytorch >= 1.4

## Usage
1. **train self-attn classifier**
   run `./classifier/train_self_attn.sh`

2. **train cnn classifier**
   run `./classifier/train_cnn.sh`

3. **save cnn classifier**
   change `MODEL_PATH` in *classifier\save_model.sh* to cnn model dir (containing `model_state_dict.pt`), e.g. `classifier/exp`
   run `./classifier/save_model.sh`

4. **train transfer model**
   change `CLS_MODEL_PATH` in *transfer\train.sh* to self-attn model dir (containing `model_state_dict.pt`), e.g. `classifier/exp`
   run `transfer\train.sh`

5. **test transfer model**
   change `MODEL_PATH` in *transfer\predict.sh* to transfer model dir (containing `model_state_dict.pt`), e.g. `classifier/exp`

