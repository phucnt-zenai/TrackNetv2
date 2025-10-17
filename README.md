A copy of repo: https://github.com/wolfyeva/TrackNetV2. Also, I used library ```torch_pruning``` to prune in a structuring way.

### My Model weight
My weights are saved in ```models/```, containing ```full_architecture``` (full-architecture checkpoints) and ```pruning``` (pruned models).
All are trained in batchsize `16` in Kaggle. The pruned models come from pruning and finetuning `model_cur_24epochs_v2.pt`.
### Dataset
Dataset description: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw. You can use `python3 preprocess.py` to preprocess them.

Alternatively, you can use my work at:  
 - https://www.kaggle.com/datasets/phuc25111/shuttlecock-tracknetv2
 - https://www.kaggle.com/datasets/phuc25111/shuttlecock-tracknetv2-2

### Requirements
`pip install -r requirements.txt`

### Train TrackNet
`python3 train.py --num_frame 3 --epochs 30 --batch_size 10 --learning_rate 0.001 --save_dir exp`

There are also parameters for resuming or pruning. Please read carefully `train.py` to adapt properly.

### Evaluate TrackNet
I use our self-annotated testset. They stems from public tournaments on Youtube. 

The dataset url: https://www.kaggle.com/datasets/phuc25111/badminton-testset.

`python /kaggle/working/TrackNetv2/eval_on_Hau_checked.py --prune --model_file <checkpoint_url> --tolerance 5 --batch_size 8`

Some results:
| Checkpoint    | Params (M)      | Acc     | Precision     | Recall     | Latency (ms)     |
|---------------|-----------------|---------|---------------|------------|-------------|
| model_cur_8epochs_v2.pt     |<div align="center">10.15</div>| Dữ liệu 3 | Dữ liệu 4     | Dữ liệu 5  | Dữ liệu 6   |
| model_cur_16epochs_v2.pt    |<div align="center">10.15</div>| Dữ liệu 9 | Dữ liệu 10    | Dữ liệu 11 | Dữ liệu 12  |
| model_cur_24epochs_v2.pt    |<div align="center">10.15</div>| Dữ liệu 15| Dữ liệu 16    | Dữ liệu 17 | Dữ liệu 18  |
| pruning_model_step_1.pt   | <div align="center">8.19</div>     | | Dữ liệu 22    | Dữ liệu 23 | Dữ liệu 24  |
| pruning_model_step_2.pt    | <div align="center">6.47</div>    | Dữ liệu 27| Dữ liệu 28    | Dữ liệu 29 | Dữ liệu 30  |
| pruning_model_step_3.pt    | <div align="center">4.96</div>     | Dữ liệu 33| Dữ liệu 34    | Dữ liệu 35 | Dữ liệu 36  |
| pruning_model_step_4.pt   | <div align="center">3.64</div>      | 0.83 | 0.94    | 0.86 | 5.2  |



 *Note: I measured in batchsize `1`, if increase batchsize, you will gain a better mean latency. The latency herein is just the average per frame, meanwwhile we forward 3 frames per batch as model inputs.*


### Example code
- Training: https://www.kaggle.com/code/nguyenthangphuc/tracknetv2-test-shuttlecock-finetune-fec3c6
- Evaluation: https://www.kaggle.com/code/nguyenthangphuc/test-tracknetv2

*This repo is as a part of VNPT_Media project*
