import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import shutil
from timeit import default_timer as timer
from apex import amp
from tools.eval_tool import valid, gen_time_str, output_value

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step,
        #"amp": amp.state_dict()
    }
    if config.getboolean('train', 'fp16'):
        save_params['amp'] = amp.state_dict()

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"]
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    if config.getboolean('train', 'fp16'):
        opt_level = 'O2'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)



    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")

    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")

    total_len = len(dataset)
    more = ""
    if total_len < 10000:
        more = "\t"
    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num

        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1
        
        try:
            train_steps = config.getint('train', 'train_steps')
        except:
            train_steps = 1
        
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])
            
            
            data['epoch'] = epoch_num
            results = model(data, config, gpu_list, acc_result, "train")

            loss, acc_result = results["loss"], results["acc_result"]
            loss = loss.mean() 
            total_loss += float(loss)
            loss = loss / train_steps
            if config.getboolean('train', 'fp16'):
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if step % train_steps == 0:
                if config.getboolean('train', 'fp16'):
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            
            if step % output_time == 0:
                output_info = output_function(acc_result, config, mode = 'train')

                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

            global_step += 1
                        
        output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                     "%.3lf" % (total_loss / (step + 1)), output_info, None, config)


        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError
        if config.getboolean('train', 'pre_train'):
            save_path = os.path.join(output_path, 'epoch_%d' % current_epoch)
            os.makedirs(save_path, exist_ok = True)
            model.save_pretrained(save_path)
        #else:
        checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config,
                   global_step)

        
        if not config.getboolean('train', 'no_valid'):
            if current_epoch % test_time == 0:
                with torch.no_grad():
                    valid(model, parameters["valid_dataset"], current_epoch, config, gpu_list, output_function)
        