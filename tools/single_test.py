from argparse import ArgumentParser
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import torch

from utils.inference import inference_model, init_model, show_result_pyplot
from utils.train_utils import get_info, file2dict
from models.build import BuildNet


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--classes-map', default='datas/annotations.txt', help='classes map of datasets')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--save-path',
        help='The path to save prediction image, default not to save.')
    args = parser.parse_args()

    classes_names, label_names = get_info(args.classes_map)
    # build the model from a config file and a checkpoint file
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')

    # test a single image
    start_time = time.time()  # 获取开始时间
    result = inference_model(model, args.img, val_pipeline, classes_names, label_names)
    end_time = time.time()  # 获取结束时间
    inference_time = end_time - start_time  # 计算推理时间

    print(f"Inference time: {inference_time} seconds")

    # show the results
    show_result_pyplot(model, args.img, result, out_file=args.save_path)


if __name__ == '__main__':
    main()