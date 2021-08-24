# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "train_tag_news.json",
    "valid_data_path": "valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"LinearRegression",
    "max_length": 20,
    "hidden_size": 256,
    "kernel_size": 3,
    "epoch": 10,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\badou\pretrain_model\chinese-bert_chinese_wwm_pytorch"
}