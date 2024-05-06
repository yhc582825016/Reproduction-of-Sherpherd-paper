from yacs.config import CfgNode as CN

INF = 1e8
# _C = CN()
# _C.task_type = ''
# _C.seed = 42

_C = CN()
# dataset
_C.dataset = CN()
_C.dataset.train_data_path = ''
_C.dataset.val_data_path = ''
_C.dataset.test_data_path = ''
_C.dataset.max_source_length = 256
_C.dataset.max_target_length = 512
_C.dataset.data_type = 'Gsm8k'
# model
_C.model = CN()
_C.model.model_type = 'bert'
_C.model.offload = False
_C.model.model_path = ''
# train
_C.train = CN()
_C.train.weight_decay = 0.01
_C.train.learning_rate = 1e-5
_C.train.per_device_eval_batch_size = 2
_C.train.per_device_train_batch_size = 2
_C.train.gradient_accumulation_steps = 1
_C.train.lr_scheduler_type = 'cosine'
_C.train.num_warmup_steps = 100
_C.train.num_train_epochs = 2
_C.train.Dropout = 0.1
_C.train.save_path = ''
_C.train.beta = 0.5
_C.train.lora_version = 'lora'
_C.train.lora_rank = 8
_C.train.scheduler = 'CAWR'
_C.train.T_mult = 1
_C.train.rewarm_epoch_num = 1
# deepspeed
_C.deepspeed = CN()
_C.deepspeed.zero_stage = 1
_C.deepspeed.offload = False

_C.log = CN()
_C.log.checkpoint_save_interval = 10000
_C.log.eval_epoch_ratio = 0.1
_C.log.eval_interval = -1
_C.log.project_name = ''
_C.log.run_name = ''
_C.log.output_dir = ''

_C.evaluator = CN()
_C.evaluator.data_path = ''
_C.evaluator.checkpoint_path = ''
_C.evaluator.result_save_path = ''
if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)


