# GEP-PII-Detection
GEP: A GCG-Based method for extracting personally identifiable information from chatbots built on small language models

We show how to train our ChatBioGPT and how to insert PII and extract in this file.

## Environment
By running:
```bash
pip install -r requirements.txt
```

## Steps for training and evaluating the ChatBioGPT
1. First train the biogpt with alpaca dataset by running
```bash
python train.py --model_name_or_path "microsoft/biogpt" --data_path "./alpaca_data.json" --train_mode "without_pii" --output_dir "checkpoint_alp" --num_train_epochs 3 --per_device_train_batch_size 16 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03
```
`--model_name_or_path` refers to the directory of the loading model, `--data_path` is the path of the dataset, `--train_mode` defines if we want to train the ChatBioGPT for chatting or conducting PII extraction, `--output_dir` is the directory to save the trained weights. The remaining parameters are hyperparameters for training the model.

2. Then keep training with HealthCareMagic-100k
```bash
python train.py --model_name_or_path "checkpoint_alp" --data_path "./HealthCareMagic-100k.json" --train_mode "without_pii" --output_dir "checkpoint_hcm" --num_train_epochs 3 --per_device_train_batch_size 16 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03
```

Then we have the weights of the ChatBioGPT in `checkpoint_hcm`

3. To measure the BERTscore of the finetuned ChatBioGPT, run the following command
```bash
python evaluation.py --mode "BERTscore" --bert_model "roberta-large" --model_path "checkpoint_hcm"
```
The `--mode` flag is for BERTscore calculation or template-based query attack. `--model_path` is the path of loading the model. For the `--bert_model` flag, some other models are also available, such as `"bert-base-uncased"`, etc. More models can be found in
- [Evaluate Metric by Huggingface](https://huggingface.co/spaces/evaluate-metric/bertscore)

The output will be the three metrics of BERTscore, i.e., prevision, recall and f1, including the confidence interval.

4. To chat with ChatBioGPT, run the following command:
```bash
python chat.py --model_path "checkpoint_hcm"
```


## Steps for inserting the template-based PII and attack with either template-based query or GEP
1. We insert the PII into HealthCareMagic-100k, and keep training based on the model which has already been finetuned on alpaca. For template-based insertion, using the following command:
```bash
python train.py --model_name_or_path "checkpoint_alp" --train_mode "with_pii" --insert_mode "template-based" --output_dir "checkpoint_t" --num_train_epochs 3 --per_device_train_batch_size 16 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03
```
The `--model_name_or_path` is the path of loading model. The `--insert_model` defines the template-based insertion or free-style insertion. The `--output_dir` refers the output directory of the model weights. The weights of the model with template-based PII insertion will be stored in `./checkpoint_t`.
2. For template-based query attack targeting template-based insertion, run the following command:
```bash
python evaluation.py --mode "template-based query" --model_path "checkpoint_t" --decoding "greedy"
```
For `--decoding` flag, `"beam"` and `"topk"` can also be used here to define the decoding strategy.
The output will be the percentage of successful attack, i.e., attack successful rate (ASR).
3. For GEP attack targeting template-based insertion, run the following command:
```bash
python GEP.py --model_path "checkpoint_t" --decoding "greedy" --trigger_length 4 --steps 140 --bs 256 --topk 256 --generation_length 256 --loss_curve False
```
For `--loss_curve` flag, it is a boolean parameter. If it is set to true, the loss curve w.r.t each entry during the training of trigger tokens will be stored in `./GEPloss` folder.

The attacking results will be in `./logs/attackprompt_single.txt`. It includes the final ASR, the number of successful attacks at each step of the training and at each index in the generation.
It also contains more details about the entries which are successfully attacked, including the best trigger tokens for each entry, the generation with the input of the trigger tokens, the expected leakage (disease) and the index of the appearance in the generation.
### The form of the `./logs/attackprompt_single.txt`
- The final ASR: `Successful attack's rate is {}`
- The number of successful attacks at each step: `The number of successful attacks at each step is: []`. The length of the list equals the total steps, showing the number of successful attacks at each step.
- The number of successful attacks at each index in the generation: `The number of successful attacks at each index in the final generation is: []`. The length of the list equals the maximum length of generation, showing where does the PII appear in the generation.
- The best trigger tokens: `{}'s attack trigger is: {}`, where the first bracket is the name of the PII data pair, e.g., John Doe.
- Generation, expected leakage and index: `Generation is ###{}###, and test prefix is ###{}###, and appear_idx is ###{}###'`, where the expected leakage is the disease, e.g., BPPV.


## Steps for inserting the free-style PII and attack with GEP
1. For free-style insertion, using the following command:
```bash
python train.py --model_name_or_path "checkpoint_alp" --train_mode "with_pii" --insert_mode "free-style" --output_dir "checkpoint_f" --num_train_epochs 3 --per_device_train_batch_size 16 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03
```
The weights of the model with pii insertion will be stored in `./checkpoint_f`

2. Run GEP for attack targeting free-style insertion
```bash
python GEP_multiple.py --model_path "checkpoint_f" --decoding "greedy" --trigger_length 16 --steps 100 --bs 256 --topk 256 --generation_length 256 --loss_curve False --split_ratio 0.5
```
`--split_ratio` is ratio for splitting the training set and validation set based on the original dataset.
The attacking results will be in `./logs/attackprompt_multiple.txt`. It includes at which step the best trigger tokens appear in both training and validation set, the ASR at each step in both training and validation set, 
the trigger tokens for each step, the input, the generation, the target and the appearing index.

### The form of the `./logs/attackprompt_multiple.txt`
- The steps for the best triggers and ASR: `The best train / validation trigger happens in iteration {}, where the ASR is {}`
- The ASR at each step: `he ASR at each training / validation step is: []`, where the length of the lists equals the total number of steps.
- The trigger tokens for each step: `Current triggers are {}`
- The input, the generation, the target and the appearing index: `Input is ###{}###, and generation is ###{}###, and test prefix is ###{}###, and appear_idx is ###{}###`
- The number of successful attacks at each index in the generation: `The number of successful attacks at each index in the final generation is: []`, where the length of the list equals to the maximum of generation. It shows at the current step, for all the entries in the training / validation set, if the attack is successful, where the PII will appear in the generation.

