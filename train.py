import argparse
import os
import torch
import logging

from tools.init_tool import init_all
from config_parser import create_config
from tools.train_tool import train



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--pretrained_bert_path', help="needed for fine-tuning")
    parser.add_argument('--local_rank')
    args = parser.parse_args()

    configFilePath = args.config
    torch.multiprocessing.set_sharing_strategy('file_system')

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.system("export CUDA_VISIBLE_DEVICES={0}".format(args.gpu))
        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))
    os.system("clear")
    print(gpu_list)
    config = create_config(configFilePath)
    

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    if not args.pretrained_bert_path is None:
        config.set('model', 'bert_path', args.pretrained_bert_path)
    
    parameters = init_all(config, gpu_list, args.checkpoint, "train")

    train(parameters, config, gpu_list)
        
