import os
import numpy as np
import random
import torch
import torch.nn as nn
import argparse
# import torch.nn.functional as F
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification,AutoTokenizer, get_linear_schedule_with_warmup
from data_handel import process_train, make_batch, split_and_load_dataset, shuffle
from logging import set_logger
from sklearn.metrics import classification_report

logger = set_logger('./logs', __name__)

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer(model_path, device, n_labels=None):
    logger.info("loading model and tokenizer from {} ...".format(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    read_token = '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'
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, cache_dir='./my_model', use_auth_token=read_token)
    if n_labels:
        hidden_size = model.config.hidden_size
        if hasattr(model.classifier, 'out_proj'):
            model.classifier.out_proj = nn.Linear(in_features=hidden_size, out_features=n_labels, bias=True)
        else:
            model.classifier = nn.Linear(in_features=hidden_size, out_features=n_labels, bias=True)
        model.config.num_labels = n_labels
    model.to(device)
    return model, tokenizer

def report_performance(preds, labels):
    report = classification_report(labels, preds, zero_division=0, output_dict=True)
    return report['accuracy'], report['macro avg']['recall']

def train_epoch(model, criterion, optim, scheduler, train_loader, val_loader, epoch, train_log_interval=10, val_internal=50, val_res=None, save_dir=None, device=0):
    model.train()
    len_iter = len(train_loader)
    n_step = 0
    val_acces, val_fscores, val_losses = [], [], []
    for i, batch in enumerate(train_loader, start=1):
        optim.zero_grad()
        input_ids, attention_mask, labels = make_batch(batch, device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optim.step()

        n_step += 1
        if scheduler:
            scheduler.step()
        if i % train_log_interval == 0:
            logger.info("epoch: %d [%d/%d], loss: %.6f, lr: %.8f, steps: %d" %
                  (epoch, i, len_iter, loss.item(), optim.param_groups[0]["lr"], n_step + len_iter * (epoch-1)))
        if i % val_internal == 0:
            acc, score, loss = val_epoch(model, criterion, val_loader, save_dir, val_res, device)
            val_acces.append(acc)
            val_fscores.append(score)
            val_losses.append(loss)

    return val_acces, val_fscores, val_losses

# class Focal_Loss(nn.Module):
#     def __init__(self, weight, gamma=2):
#         super(Focal_Loss,self).__init__()
#         self.gamma = gamma
#         self.weight = weight       # 是tensor数据格式的列表
#
#     def forward(self, preds, labels, device='cpu'):
#         preds = F.softmax(preds,dim=1).to(device)
#         # print(preds)
#         eps = 1e-7
#
#         target = self.one_hot(preds.size(1), labels).to(device)
#         # print(target)
#
#         ce = -1 * torch.log(preds+eps) * target.to(device)
#         # print(ce.device)
#         floss = (torch.pow((1-preds), self.gamma) * ce).to(device)
#         # print(self.weight.device)
#         floss = torch.mul(floss, self.weight)
#         floss = torch.sum(floss, dim=1)
#         return torch.mean(floss)
#
#     def one_hot(self, num, labels):
#         one = torch.zeros((labels.size(0),num)).to(device)
#         one[range(labels.size(0)),labels] = 1
#         return one

def val_epoch(model, criterion, val_loader, save_dir, val_res, device):
    model.eval()
    total_eval_loss = 0
    preds, Labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = make_batch(batch, device)
            outputs = model(input_ids, attention_mask=attention_mask)
            # lf = Focal_Loss(torch.tensor([0.1, 0.1, 0.2, 0.45, 0.12, 0.03])).to(device)
            # loss = lf(outputs.logits, labels).to(device)
            loss = criterion(outputs.logits, labels)
            # loss = (loss * sample_weight / sample_weight.sum()).sum()
            total_eval_loss += loss.item()
            batch_preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist()
            label_ids = labels.to('cpu').numpy().tolist()
            preds.extend(batch_preds)
            Labels.extend(label_ids)

    avg_val_loss =total_eval_loss / len(val_loader)
    acc, score = report_performance(preds, Labels)
    if save_dir:
        if score > max(val_res) and acc > 0.72: #0.77
            save_model(model, save_dir)
            # 如果保存了新的模型才记录最优的那个score
            val_res.append(score)

    logger.info("acc: %.4f, score: %.4f, best socre: %.4f, loss: %.4f" % (acc, score, max(val_res), avg_val_loss))
    return acc, score, avg_val_loss

def save_model(model, save_dir):
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_dir, CONFIG_NAME)
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_dir)
    logger.info('=====================  模型保存： %s  =====================' % save_dir)

def train(model, criterion, optim, scheduler, train_loader, val_loader, epoch, save_dir, device):
    val_res = [0]
    for i in range(1, epoch + 1):
        train_epoch(model, criterion, optim, scheduler, train_loader, val_loader, save_dir=save_dir, epoch=i, train_log_interval=10, val_internal=20, val_res=val_res, device=device)
        val_epoch(model, criterion, val_loader, save_dir, val_res, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seeds", type=int, default=131)
    parser.add_argument("--model-name", type=str, default="erlangshen")    # 1:第i次 2:epoch 3:warmup-steps
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args()
    # python main.py --gpus=1 --seeds=5131 --model-name="erlangshen" --warmup-steps=100
    # python main.py --gpus=2 --seeds=5132 --model-name="erlangshen" --warmup-steps=200
    # python main.py --gpus=3 --seeds=5134 --model-name="erlangshen" --warmup-steps=400

    device = torch.device("cuda:{}".format(args.gpus) if torch.cuda.is_available() else "cpu")
    data, n_labels, cnt = process_train('./train_data.csv')

    # weight = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.45, 0.12, 0.03])).float()
    # weight = torch.from_numpy(np.array([1, 0.1, 0.2, 0.45, 0.12, 40])).float()
    # weights = [3.1, 2.7, 6.9, 40.6, 4.2, 1.0]  [1.0, 1.0, 1.0, 40, 1.0, 1.0]
    weights = [0.1, 0.1, 0.2, 0.45, 0.12, 0.03]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 130
    # 1:第i次 2:epoch 3:warmup-steps
    seeds = [args.seeds]
    for seed in seeds:
        set_seed(seed)
        epoch = 3
        # IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment
        model_name = args.model_name
        if model_name == 'bert':
            model, tokenizer = load_model_and_tokenizer('bert-base-chinese', device, n_labels=n_labels)
        elif model_name == 'erlangshen':
            model, tokenizer = load_model_and_tokenizer('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment', device,n_labels=n_labels)
# 'uer/roberta-base-finetuned-chinanews-chinese'
        save_dir = './outputModel/{}/{}'.format(model_name, seed)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        *_, train_loader, val_loader = split_and_load_dataset(data, tokenizer, max_len=84, batch_size=32)
        optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_loader) * epoch)
        train(model, criterion, optim, scheduler, train_loader, val_loader, epoch=epoch, save_dir=save_dir, device=device)

        del model, tokenizer, optim, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        data = shuffle(data)