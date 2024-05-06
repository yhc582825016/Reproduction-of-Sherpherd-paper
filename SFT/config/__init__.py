from .sft_config import _C as sft_config
def update_config(config, args):
    config.defrost()
    # set cfg using yaml config file
    config.merge_from_file(args.config_file)
    # update cfg using args
    config.merge_from_list(args.opts)
    # config.freeze()