# DSDocRE
Source code for "Denoising Relation Extraction from Document-level Distant Supervision".

### Requirements:
* Python3
* Pytorch>=1.5.0
* apex>=0.1
* numpy
* transformers>=2.5.1


### Prepare Data
Data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw?usp=sharing).
Please put the data file in the `data/`.

### Pre-denoise
Run the following command to train the pre-denoising model:
```
python3 train.py -c config/PreDenoising.config -g 0
```

After training the pre-denoising model, run the following commands to generate rank score for all data:
```
mkdir data/rank_result

# Generate the rank_result for test data.
python3 test.py -g 0 \
    --config config/PreDenoising.config \
    --test_file data/test.json \
    --checkpoint checkpoint/PreDenoise/3.pkl \
    --result_score data/rank_result/test_score.npy \
    --result_title data/rank_result/test_title.json 

# Generate the rank_result for valid data.
python3 test.py -g 0 \
    --config config/PreDenoising.config \
    --test_file data/dev.json \
    --checkpoint checkpoint/PreDenoise/3.pkl \
    --result_score data/rank_result/valid_score.npy \
    --result_title data/rank_result/valid_title.json 

# Generate the rank_result for train_annotated data.
python3 test.py -g 0 \
    --config config/PreDenoising.config \
    --test_file data/train_annotated.json \
    --checkpoint checkpoint/PreDenoise/3.pkl \
    --result_score data/rank_result/train_annotated_score.npy \
    --result_title data/rank_result/train_annotated_title.json 

# Generate the rank_result for train_distant data.
python3 test.py -g 0 \
    --config config/PreDenoising.config \
    --test_file data/train_distant.json \
    --checkpoint checkpoint/PreDenoise/3.pkl \
    --result_score data/rank_result/train_distant_score.npy \
    --result_title data/rank_result/train_distant_title.json 
```

### Pre-train
```
python3 train.py -c config/PreTraining.config -g 0
```

### Fine-tune
```
# Choose an appropriate model to fine-tune
python3 train.py -c config/FineTuning.config -g 0 \
    --pretrained_bert_path checkpoint/PreTrain/epoch_15/
```

### Cite
If you use the code, please cite this paper:
```
@inproceedings{xiao2020denoise,
  title={Denoising Relation Extraction from Document-level Distant Supervision},
  author={Xiao, Chaojun and Yao, Yuan and Xie, Ruobing and Han, Xu and Liu, Zhiyuan and Sun, Maosong and Lin, Fen and Lin, Leyu},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
```
