# KOREAN_NLP_Classification

## Enviroments
-[Ubuntu 20.04]

-[Python>=3.8]

## Installation
Please go ahead and install the Python package to run.

```sh
$pip3 install -r requirements.txt
```

## Data Preparation
The "KOREAN_NLP_Classification" uses data in the format of "example_data.csv" in the data folder.

If you use NLP data in a different format, modifications to the "dataload.py" module will be necessary.


### example_data.csv
|text|class|
|--------|--------|
|sample 1. text content|sample 1 label|
|...|...|
|sample n. text content|sample n label|

`text`: The content, which is comprised of text, corresponds to the input.

`class`: The class represents the label for each sample(text). Each class item must be expressed as a number. For example, in a binary classification problem, 'False' and 'True' should be represented as 0 and 1, respectively.

## Korea NLP models
|Model|url||
|----------|-----------|-----------|
|KLUE-BERT|https://github.com/KLUE-benchmark/KLUE.git|Use and Support on Hugging Face|
|KLUE-RoBERTa|https://github.com/KLUE-benchmark/KLUE.git|Use and Support on Hugging Face|
|KoBERT|https://github.com/SKTBrain/KoBERT.git|-|
|KorBERT|https://aiopen.etri.re.kr/bertModel|KorBERT can be downloaded and used with the permission of ETRI|
|KoBigBird|https://github.com/monologg/KoBigBird.git|Use and Support on Hugging Face|

* KoBigBird can handle more than 512 tokens, with a maximum of 4096 tokens.
* If you are using KorBERT, you need to download the model through ETRI and then store KorBERT in the 'module' folder. Among the models supported by ETRI, we used kor_tensorflow. Upload the required files in a blank state.

## Model Train
Each NLP model can be trained using the `train.py` script.
```sh
$python3 train.py
```

### Adjustable Parameters
`batch size`: Default is 1.

`lr`: Maximum learning rate, default is 2e-5.

`weight_decay`: A parameter related to overfitting. 0 indicates potential overfitting, and increasing this value leads to greater generalization. The default is 0.02.

`pct_start`: Refers to the position of peak learning rate when using the OneCycleLR learning scheduler. It is 0 at the start and one at the end. The default is 0.3.

`epoch`: The number of times the entire dataset is used for training. The default is 10.

`seq_len`: The length of input tokens. Typical NLP models have a maximum of 512 tokens. (KoBigBird can handle up to 4096 tokens) The default is 512.

`NLP_model`: Default is KLUE-RoBERTa

`random_seed`
```sh
$python3 train.py --batch_size 1 --lr 2e-5 --weight_decay 0.02 --pct_start 0.3 --epoch 10 --NLP_model KLUE-RoBERTa --seq_len 512 --random_seed 102 --data_path ./data
```

### Considerations and Modifications
* You need to define the Data path. If necessary, add functions to split the data into train, validation, and test sets.
  ```sh
  train_txt, train_class = Data_Load(data_path+'/train.csv')
  val_txt, val_class = Data_Load(data_path+'/val.csv')
  test_txt, test_class = Data_Load(data_path+'/test.csv')
  ```
* Modify the num_worker parameter of DataLoader to suit your work environment. The default is 4.
  ```sh
  train_batch = data.DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  collate_fn=train_data.pad)
  ```
* If you want to handle the results, you can use `train_result()`. The available results include `f1-score`, `report`(classification_report), `loss_list`, `model_file`(trained model), `predicted_label`(predicted result for each sample), and `logits_list`(prediction scores for each sample).
  ```sh
  f1_list, report, loss_list, model_file, predicted_labels, true_labels, logits_list = train.train_result()
  F1_LIST, REPORT, PREDICTED_LABELS, TRUE_LABELS, LOGIT_LIST = train.test_result() 
  ```
  (Each prediction result covers both validation and test data.)
  
* If you have predefined label names, modify the `label_list`. When creating a `report`, it uses the names from the `label list`. Note that you should define them in order, starting from class 0.
  ```sh
  label_list = ['False', 'True']
  ```
  ||precision|recall|f1-score|support|
  |---|---|---|---|---|
  |False|-|-|-|-|
  |True|-|-|-|-|
  |accuracy|||-|-|
  |macro avg|-|-|-|-|
  |weighted avg|-|-|-|-|

## Model Evaluation
You can perform an evaluation using the `eval.py` script.

Evaluation is similar to training. The important point is to modify the parameters and details defined during training, such as the `batch_size`, the `seq_len`, the `label_list`, etc.
```sh
$python3 eval.py --batch_size 1 --NLP_model KLUE-RoBERTa --seq_len 512 --data_path ./data
```
