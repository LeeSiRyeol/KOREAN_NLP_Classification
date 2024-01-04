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
|text content|number of class|

`text`: The content, which is comprised of text, corresponds to the input.

`class`: The class represents the label for each sample(text). Each class item must be expressed as a number. For example, in a binary classification problem, 'False' and 'True' should be represented as 0 and 1, respectively.

## Korea NLP models
|Model|url||
|----------|-----------|-----------|
|KLUE-BERT|https://github.com/KLUE-benchmark/KLUE.git|Use and Support on Hugging Face|
|KLUE-RoBERTa|https://github.com/KLUE-benchmark/KLUE.git|Use and Support on Hugging Face|
|KoBERT|https://github.com/SKTBrain/KoBERT.git|-|
|KorBERT|https://aiopen.etri.re.kr/bertModel|KorBERT can be downloaded and used with the permission of ETRI.|
|KoBigBird|https://github.com/monologg/KoBigBird.git|Use and Support on Hugging Face|

* `KoBigBird` can use over the 512 tokens; maxinum input tokens is 4096.

## Model Train

