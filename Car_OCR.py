from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import io
from PIL import Image
import os
import sys
import platform
import yaml
import time
import datetime
import paddle
import paddle.distributed as dist
from tqdm import tqdm
import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from argparse import Namespace

from ppocr.utils.stats import TrainingStats
from ppocr.utils.save_load import save_model
from ppocr.utils.utility import print_dict, AverageMeter
from ppocr.utils.logging import get_logger
from ppocr.utils.loggers import VDLLogger, WandbLogger, Loggers
from ppocr.utils import profiler
from ppocr.data import build_dataloader
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'


def ocr(src):
    flag = 0  # 1是正的，2是倒的
    # 将彩色图像转换为灰度图像
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 获取图片的尺寸
    src_h, src_w = src.shape[:2]
    crop_size = 5
    cropped_src = src[crop_size: src_h - crop_size - 1, crop_size: src_w - crop_size - 1]
    ret, thresh = cv2.threshold(cropped_src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 255 - thresh
    fix_size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fix_size, fix_size))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    fix_size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, fix_size))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # 转换图片到BGR到RGB（如果需要的话）
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
    # 获取图片的尺寸
    height, width = thresh.shape[:2]
    # 确定四个象限的边界
    # 计算每部分的宽度
    part_width = width // 5

    # 计算最左侧部分和最右侧部分的索引
    left_part_indices = (0, part_width)
    right_part_indices = (4 * part_width, width)

    # 提取最左侧部分和最右侧部分的像素值
    left_part = thresh[:, :part_width, :]
    right_part = thresh[:, right_part_indices[0]:right_part_indices[1], :]

    # 使用np.sum计算像素值之和
    sum_left = np.sum(left_part)
    sum_right = np.sum(right_part)

    condition = sum_left < sum_right

    if condition:
        # 旋转图片180度
        src = cv2.flip(src, -1)
        flag = 2
    else:
        flag = 1

    if flag == 2:
        src_h, src_w = src.shape[:2]
        crop_size = 5
        cropped_src = src[crop_size: src_h - crop_size - 1, crop_size: src_w - crop_size - 1]
        ret, thresh = cv2.threshold(cropped_src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = 255 - thresh
        fix_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fix_size, fix_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        fix_size = 100
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, fix_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        column_counts = np.count_nonzero(thresh > 1, axis=0)
    else:
        fix_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fix_size, fix_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        fix_size = 100
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, fix_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # 将彩色图像转换为灰度图像
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        column_counts = np.count_nonzero(thresh > 1, axis=0)
    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 假设只有一个主要的轮廓需要处理，选择面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    # 获取外接矩形
    x, y, w, h = cv2.boundingRect(largest_contour)
    # 根据外接矩形裁剪图像
    cropped_src_v2 = cropped_src[y:y + h, x:x + w]
    ret, thresh = cv2.threshold(cropped_src_v2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 255 - thresh
    fix_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fix_size, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    fix_size = 100
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fix_size, 1))
    thresh_1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    _, thresh_1_w = thresh_1.shape[:2]
    thresh_1 = thresh_1[:, int(thresh_1_w / 2):]

    # 查找轮廓
    contours, _ = cv2.findContours(thresh_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最高轮廓的底边坐标
    highest_contour_bottom = 100
    lowest_contour_bottom = 100

    # 遍历轮廓，找到最高轮廓的底边
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0:
            # 更新最高轮廓的底边坐标
            if y + h < highest_contour_bottom:
                highest_contour_bottom = y + h
            if y > lowest_contour_bottom:
                lowest_contour_bottom = y

    # 裁剪图像，以最高轮廓的底边为新上边缘
    thresh = thresh[highest_contour_bottom + 5:lowest_contour_bottom, :]
    cropped_src_v2 = cropped_src_v2[highest_contour_bottom + 5:lowest_contour_bottom, :]

    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 初始化列表，用于存储最右侧的两个矩形的索引
    right_contours = []
    for i, contour in enumerate(contours):
        # 获取外接矩形的坐标和尺寸
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(cropped_src_v2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色矩形
        if w > 10:
            right_contours.append((contour, x + w))

    right_contours.sort(key=lambda x: x[1], reverse=True)
    contour1, _ = right_contours[0]
    contour2, _ = right_contours[1]

    # 截取区域1
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    expansion_ratio1 = 0
    expanded_w1 = int(w1 * (1 + expansion_ratio1))
    expanded_h1 = int(h1 * (1 + expansion_ratio1))

    # 计算扩展后的矩形左上角坐标
    # 由于宽度和高度增加了，需要相应地调整x和y坐标
    expanded_x1 = x1 - (expanded_w1 - w1) // 2
    expanded_y1 = y1 - (expanded_h1 - h1) // 2

    # 确保扩展后的矩形不会超出图像边界
    ex1 = max(0, expanded_x1)
    ey1 = max(0, expanded_y1)
    ew1 = min(expanded_w1, cropped_src_v2.shape[1] - expanded_x1)
    eh1 = min(expanded_h1, cropped_src_v2.shape[0] - expanded_y1)
    # 裁剪图像
    cropped_image1 = cropped_src_v2[ey1 - 3:ey1 + eh1 + 3, ex1 - 3:ex1 + ew1 + 3]
    # 截取区域2
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    expansion_ratio2 = 0.05
    expanded_w2 = int(w2 * (1 + expansion_ratio2))
    expanded_h2 = int(h2 * (1 + expansion_ratio2))

    # 计算扩展后的矩形左上角坐标
    # 由于宽度和高度增加了，需要相应地调整x和y坐标
    expanded_x2 = x2 - (expanded_w2 - w2) // 2
    expanded_y2 = y2 - (expanded_h2 - h2) // 2

    # 确保扩展后的矩形不会超出图像边界
    ex2 = max(0, expanded_x2)
    ey2 = max(0, expanded_y2)
    ew2 = min(expanded_w2, cropped_src_v2.shape[1] - expanded_x2)
    eh2 = min(expanded_h2, cropped_src_v2.shape[0] - expanded_y2)
    # 裁剪图像
    cropped_image2 = cropped_src_v2[ey2 - 3:ey2 + eh2 + 3, ex2 - 3:ex2 + ew2 + 3]
    # 文本识别
    FLAGS = Namespace()
    FLAGS.config = 'model/en_PP-OCRv4_rec.yml'
    FLAGS.opt = {}
    FLAGS.profiler_options = None
    profiler_options = FLAGS.profiler_options
    """
        Load config from yml/yaml file.
        Args:
            file_path (str): Path of the config file to be loaded.
        Returns: global config
    """
    _, ext = os.path.splitext(FLAGS.config)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(FLAGS.config, 'rb'), Loader=yaml.Loader)
    """
        Merge config into global config.
        Args:
            config (dict): Config to be merged.
        Returns: global config
        """
    for key, value in FLAGS.opt.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                    sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]

    profile_dic = {"profiler_options": profiler_options}
    """
        Merge config into global config.
        Args:
            config (dict): Config to be merged.
        Returns: global config
        """
    for key, value in profile_dic.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                    sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]

    log_file = None
    logger = get_logger(log_file=log_file)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global'].get('use_gpu', False)
    use_xpu = config['Global'].get('use_xpu', False)
    use_npu = config['Global'].get('use_npu', False)
    use_mlu = config['Global'].get('use_mlu', False)

    alg = config['Architecture']['algorithm']
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN',
        'CLS', 'PGNet', 'Distillation', 'NRTR', 'TableAttn', 'SAR', 'PSE',
        'SEED', 'SDMGR', 'LayoutXLM', 'LayoutLM', 'LayoutLMv2', 'PREN', 'FCE',
        'SVTR', 'SVTR_LCNet', 'ViTSTR', 'ABINet', 'DB++', 'TableMaster', 'SPIN',
        'VisionLAN', 'Gestalt', 'SLANet', 'RobustScanner', 'CT', 'RFL', 'DRRG',
        'CAN', 'Telescope', 'SATRN', 'SVTR_HGNet'
    ]

    if use_xpu:
        device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
    elif use_npu:
        device = 'npu:{0}'.format(os.getenv('FLAGS_selected_npus', 0))
    elif use_mlu:
        device = 'mlu:{0}'.format(os.getenv('FLAGS_selected_mlus', 0))
    else:
        device = 'gpu:{}'.format(dist.ParallelEnv()
                                 .dev_id) if use_gpu else 'cpu'

    """
        Log error and exit when set use_gpu=true in paddlepaddle
        cpu version.
    """
    err = "Config {} cannot be set as true while your paddle " \
          "is not compiled with {} ! \nPlease try: \n" \
          "\t1. Install paddlepaddle to run model on {} \n" \
          "\t2. Set {} as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and use_xpu:
            print("use_xpu and use_gpu can not both be ture.")
        if use_gpu and not paddle.is_compiled_with_cuda():
            print(err.format("use_gpu", "cuda", "gpu", "use_gpu"))
            sys.exit(1)
        if use_xpu and not paddle.device.is_compiled_with_xpu():
            print(err.format("use_xpu", "xpu", "xpu", "use_xpu"))
            sys.exit(1)
        if use_npu:
            if int(paddle.version.major) != 0 and int(
                    paddle.version.major) <= 2 and int(
                        paddle.version.minor) <= 4:
                if not paddle.device.is_compiled_with_npu():
                    print(err.format("use_npu", "npu", "npu", "use_npu"))
                    sys.exit(1)
            # is_compiled_with_npu() has been updated after paddle-2.4
            else:
                if not paddle.device.is_compiled_with_custom_device("npu"):
                    print(err.format("use_npu", "npu", "npu", "use_npu"))
                    sys.exit(1)
        if use_mlu and not paddle.device.is_compiled_with_mlu():
            print(err.format("use_mlu", "mlu", "mlu", "use_mlu"))
            sys.exit(1)
    except Exception as e:
        pass

    device = paddle.set_device(device)

    config['Global']['distributed'] = dist.get_world_size() != 1

    loggers = []

    if 'use_visualdl' in config['Global'] and config['Global']['use_visualdl']:
        save_model_dir = config['Global']['save_model_dir']
        vdl_writer_path = save_model_dir
        log_writer = VDLLogger(vdl_writer_path)
        loggers.append(log_writer)
    if ('use_wandb' in config['Global'] and
        config['Global']['use_wandb']) or 'wandb' in config:
        save_dir = config['Global']['save_model_dir']
        wandb_writer_path = "{}/wandb".format(save_dir)
        if "wandb" in config:
            wandb_params = config['wandb']
        else:
            wandb_params = dict()
        wandb_params.update({'save_dir': save_dir})
        log_writer = WandbLogger(**wandb_params, config=config)
        loggers.append(log_writer)
    else:
        log_writer = None
    #print_dict(config, logger)

    if loggers:
        log_writer = Loggers(loggers)
    else:
        log_writer = None

    #logger.info('train with paddle {} and device {}'.format(paddle.__version__, device))
    global_config = config['Global']
    # build post process
    post_process_class = build_post_process(config['PostProcess'], global_config)
    # build model
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config["Architecture"]["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if config["Architecture"]["Models"][key]["Head"]["name"] == 'MultiHead':  # multi head
                    out_channels_list = {}
                    if config['PostProcess']['name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    if config['PostProcess']['name'] == 'DistillationNRTRLabelDecode':
                        char_num = char_num - 3
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    out_channels_list['NRTRLabelDecode'] = char_num + 3
                    config['Architecture']['Models'][key]['Head']['out_channels_list'] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"]["out_channels"] = char_num
        elif config['Architecture']['Head']['name'] == 'MultiHead':  # multi head
            out_channels_list = {}
            char_num = len(getattr(post_process_class, 'character'))
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            if config['PostProcess']['name'] == 'NRTRLabelDecode':
                char_num = char_num - 3
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            out_channels_list['NRTRLabelDecode'] = char_num + 3
            config['Architecture']['Head']['out_channels_list'] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num
    model = build_model(config['Architecture'])
    load_model(config, model)
    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            elif config['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif config['Architecture']['algorithm'] == "RobustScanner":
                op[op_name][
                    'keep_keys'] = ['image', 'valid_ratio', 'word_positons']
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config['Global'].get('save_res_path',
                                         "./output/rec/predicts_rec.txt")
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    image_list = [cropped_image1, cropped_image2]
    img_name = ['序列号', '零件号']
    i = 0
    with open(save_res_path, "w") as fout:
        for file in image_list:
            logger.info("infer_img: {}".format(img_name[i]))
            # 然后我们将其转换为字节流
            img_byte_array = io.BytesIO()
            Image.fromarray(file).save(img_byte_array, format='PNG')
            img_byte_array = img_byte_array.getvalue()
            data = {'image': img_byte_array}
            batch = transform(data, ops)
            if config['Architecture']['algorithm'] == "SRN":
                encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

                others = [
                    paddle.to_tensor(encoder_word_pos_list),
                    paddle.to_tensor(gsrm_word_pos_list),
                    paddle.to_tensor(gsrm_slf_attn_bias1_list),
                    paddle.to_tensor(gsrm_slf_attn_bias2_list)
                ]
            if config['Architecture']['algorithm'] == "SAR":
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                img_metas = [paddle.to_tensor(valid_ratio)]
            if config['Architecture']['algorithm'] == "RobustScanner":
                valid_ratio = np.expand_dims(batch[1], axis=0)
                word_positons = np.expand_dims(batch[2], axis=0)
                img_metas = [
                    paddle.to_tensor(valid_ratio),
                    paddle.to_tensor(word_positons),
                ]
            if config['Architecture']['algorithm'] == "CAN":
                image_mask = paddle.ones(
                    (np.expand_dims(
                        batch[0], axis=0).shape), dtype='float32')
                label = paddle.ones((1, 36), dtype='int64')
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)
            if config['Architecture']['algorithm'] == "SRN":
                preds = model(images, others)
            elif config['Architecture']['algorithm'] == "SAR":
                preds = model(images, img_metas)
            elif config['Architecture']['algorithm'] == "RobustScanner":
                preds = model(images, img_metas)
            elif config['Architecture']['algorithm'] == "CAN":
                preds = model([images, image_mask, label])
            else:
                preds = model(images)
            post_result = post_process_class(preds)
            info = None
            if isinstance(post_result, dict):
                rec_info = dict()
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                            "label": post_result[key][0][0],
                            "score": float(post_result[key][0][1]),
                        }
                info = json.dumps(rec_info, ensure_ascii=False)
            elif isinstance(post_result, list) and isinstance(post_result[0],
                                                              int):
                # for RFLearning CNT branch
                info = str(post_result[0])
            else:
                if len(post_result[0]) >= 2:
                    info = post_result[0][0]

            if info is not None:
                logger.info("\t result: {}".format(info))
                fout.write(img_name[i] + "\t" + info + "\n")
            i += 1
    logger.info("识别完成!")
    return post_result


if __name__ == "__main__":
    file_path = r'E:\Pycharm\OCR\img\438.bmp'
    src = cv2.imread(file_path)
    results = ocr(src)
