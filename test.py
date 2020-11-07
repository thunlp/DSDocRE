import argparse
import os
import torch
import logging
import json
import numpy as np

from tools.init_tool import init_all
from config_parser import create_config
from tools.test_tool import test

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path", required=True)
    parser.add_argument('--result_score', help="result score file path", required=True)
    parser.add_argument('--result_title', help="result title file path", required=True)
    parser.add_argument('--test_file', required=True)
    args = parser.parse_args()

    configFilePath = args.config

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    #os.system("clear")

    config = create_config(configFilePath)
    config.set('data', 'test_data_path', args.test_file)
    #print(config.get('data', 'test_data_path'))

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    parameters = init_all(config, gpu_list, args.checkpoint, "test")

    
    with torch.no_grad():
        score, titles = test(parameters, config, gpu_list)
        #outresult = test(parameters, config, gpu_list)
    
    score = np.vstack(score)
    np.save(args.result_score, score)
    fout = open(args.result_title, "w", encoding="utf8")
    print(json.dumps(titles), file = fout)
    fout.close()
    '''
    fout = open(args.result, 'w', encoding='utf8')
    print(json.dumps(outresult), file=fout)
    fout.close()
    '''
    