## If you want to download my model
```shell
# You will get 2 folder mc & qa, which contain the model, config and tokenizer.
# You should put them in the folder where you put the .py file, or you need to change the arg parameter.
bash download.sh
```

## How to train my model
```shell
# train MC model
# below is the default setting, you can change any parameter if you want.
python run_mc.py \
--model_name_or_path bert-base-chinese \
--max_seq_length 384 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--output_dir ./mc \
--gradient_accumulation_steps=2 \
--per_device_train_batch_size=1

# train QA model
# below is the default setting, you can change any parameter if you want.
python run_qa.py \
  --model_name_or_path bert-base-chinese \
  --max_seq_length 384 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --doc_stride 128 \
  --output_dir ./qa \
  --gradient_accumulation_steps=2 \
  --per_device_train_batch_size=1
```

## How to test my model
```shell
bash run.sh "${1}" "${2}" "${3}"
# "${1}": /path/to/context.json
# "${2}": /path/to/test.json
# "${3}": /path/to/pred.json
```