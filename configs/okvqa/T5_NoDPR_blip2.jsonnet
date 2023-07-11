local base_env = import 'base_env.jsonnet';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 1;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local retriever_lr = 1e-5;
local MLP_lr = 1e-4;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;
local seed = 2021;

local override = {
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "T5",
    "ModelClass": "PrefixModelBLIP2",
    "TokenizerClass": "T5Tokenizer",
    "TokenizerModelVersion": "t5-large",
    "GeneratorModelClass": "Blip2Processor",
    "ConfigClass": "T5Config",
    "ModelVersion": "Salesforce/blip2-flan-t5-xl",
    "pretrained": 1,
    "UsePrefixEmb": 0,
    "UseQformerEmb": 0,
    "LoadPretrainMLPs": 0,
    "UseInstructBLIP" : 0,

    "modules": [
    ],
    "loss_ratio":{
      "nll_loss": 1,
      "additional_loss": 0,
      "rag_loss": 0,
    },
    "SPECIAL_TOKENS":{
      "bos_token": "<PAD>",
      "pad_token": "<PAD>",
      "additional_special_tokens": ["<BOQ>", "<EOQ>"],
    },
    "input_modules": {
      "module_list":[
        {"type": "QuestionInput",  "option": "default", 
                  "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
      ],
      "postprocess_module_list": [
      ],
    },
    "decoder_input_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
    "output_modules": {
      "module_list":[
      ],
      "postprocess_module_list": [
      ],
    },
  },
  "cache":{
    "regenerate":{
      "train_data_preprocessed_BLIP2_text": 0,
      "test_data_preprocessed_BLIP2_text": 0,
      "clip_embeddings": 0,
      "qformer_embeddings": 0,
      "instructBLIP_embeddings": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderBLIP2",
    "dataset_type": "OKVQADatasetBLIP2",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':512,
      'max_decoder_source_length': 512,
      'max_target_length':10,
    },
    "dataset_modules": {
      "module_list": [
        "LoadOKVQAData",
      ],
      "module_dict":{
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "BLIP2Executor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "retriever_lr": retriever_lr,
    "MLP_lr": MLP_lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "save_interval":save_interval,
    "scheduler": "none",
    "additional": {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "gradient_clipping": gradient_clipping,
    },
  },
  "valid": {
    "batch_size":valid_batch_size,
    "step_size":valid_step_size,
    "break_interval": break_interval,
    "additional": {
    },
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "additional": {
        "multiprocessing": 4,
    },
  },
  "metrics": [
    {'name': 'compute_okvqa_scores'},
  ],
};

std.mergePatch(base_env, override)
