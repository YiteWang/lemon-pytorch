# :lemon: BERT on the English Wikipedia

Our code is based on the [VLM](https://github.com/airsplay/vokenization) repo.

## Installation

To run our code, first install the required pacakages:

```shell
pip install -r requirements.txt
```

## Preparation of the Dataset

Follow the instructions in the [VLM](https://github.com/airsplay/vokenization) repo to download and pre-process language data. Modify the `WIKI_PATH` if necessary.

```shell
bash data/wiki/get_data_cased.bash en
```

Then tokenize the data:

```shell
bash tokenization/tokenize_wiki_bert.bash 
```

## Model Expansion

In this repo, we provide code for testing Pre-LN BERT and Post-LN BERT.

### Pre-train the Small Model

First pre-train the small model using the bash files in `scripts_inflate` using `prelnbert_wiki_udf` or `postlnbert_base_wiki_udf` depending on which BERT model you are testing. Also find proper config file of the small model in the folder `vlm/configs`. An example could be:

```shell
 bash scripts_inflate/prelnbert_wiki_udf.bash 0,1,2,3 path_to_model_s --config_name=vlm/configs/bert-6L-384H.json 
--max_steps 220000 --warmup_steps 5000 --shuffle --sched udfcosine --lr-decay 0.1
```

### Apply :lemon: LEMON to Expand the Small Model

Perform LEMON to expand the small model. Find proper config file of the expanded model in the folder `vlm/configs`.

For Pre-LN BERT, use:

```shell
python3 vlm/inflate_bert.py --src_path path_to_model_f --dest_path path_to_model_s --dest_config  path_to_config_f
```

For Post-LN BERT, use:

```shell
python3 vlm/inflate_postlnbert_lemon.py --src_path path_to_model_s --dest_path path_to_model_f --dest_config  path_to_config_f
```

### Train the Expanded Models

Resume training with the expanded checkpoint. An example could be:

```shell
bash scripts_inflate/prelnbert_wiki_udf.bash 0,1,2,3 path_to_save_model_f
--config_name=vlm/configs/bert-12L-768H.json 
--model_name_or_path=path_to_model_f 
--max_steps 132000 --warmup_steps 5000 --shuffle --sched udfcosine --lr-decay 0.1
```
