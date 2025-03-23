import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
from dataset import get_dataloaders
import scipy.stats as stats
import gc
from utils import (Logger, get_model, mixup_criterion, mixup_data, random_seed, save_checkpoint, smooth_one_hot,
                   cross_entropy)

warnings.filterwarnings("ignore")


def spearman_corr(y_true, y_pred):
    
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    rank_true = torch.argsort(torch.argsort(y_true)) + 1 
    rank_pred = torch.argsort(torch.argsort(y_pred)) + 1
    
    rank_true = rank_true.float()
    rank_pred = rank_pred.float() 

    mean_true = torch.mean(rank_true)
    mean_pred = torch.mean(rank_pred)

    numerator = torch.sum((rank_true - mean_true) * (rank_pred - mean_pred))

    denominator = torch.sqrt(torch.sum((rank_true - mean_true) ** 2) * torch.sum((rank_pred - mean_pred) ** 2))

    if denominator == 0:
        return torch.tensor(0.0)

    return numerator / denominator 

def relative_l2_distance(y_true, y_pred, all_labels):

    y_true = y_true.float()
    y_pred = y_pred.float()

    y_max = torch.max(y_true)
    y_min = torch.min(y_true)

    abs_error = torch.abs(y_true - y_pred)
    normalized_error = abs_error / (y_max - y_min)

    action_type_l2_distances = []

    for action_type in torch.unique(all_labels):

        indices = (all_labels == action_type)

        normalized_error_action = normalized_error[indices]

        action_type_l2 = torch.mean(normalized_error_action)
        action_type_l2_distances.append(action_type_l2)

    mean_relative_l2_distance = torch.mean(torch.tensor(action_type_l2_distances))

    return mean_relative_l2_distance

parser = argparse.ArgumentParser(description='USTC Computer Vision Final Project')
parser.add_argument('--arch', default="ResNet50", type=str)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--scheduler', default="reduce", type=str, help='[reduce, cos]')
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--label_smooth', default=True, type=eval)
parser.add_argument('--label_smooth_value', default=0.1, type=float)
parser.add_argument('--mixup', default=False, type=eval)
parser.add_argument('--mixup_alpha', default=1.0, type=float)
# parser.add_argument('--data_path', default='datasets/individual_diving.pkl', type=str)
parser.add_argument('--results', default='./results', type=str)
parser.add_argument('--save_freq', default=10, type=int)
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--name', default='official', type=str)

best_acc = 0

def train(model, train_loader, loss_fn, optimizer, epoch, device, scaler, writer, args):
    model.train()
    count = 0
    correct = 0
    train_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []  
    all_pred_scores = []  

    for i, data in enumerate(train_loader):
        images, labels_x, labels_y = data
        images = images.to(device)
        labels_x = labels_x.to(device)
        labels_y = labels_y.to(device)
        
        org_images, org_labels = images, labels_x

        with autocast():
            bs, c, h, w = images.shape
            # images = images.view(c, h, w)
            # labels_x = torch.repeat_interleave(labels_x, repeats=ncrops, dim=0)

            if args.mixup:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels_x, args.mixup_alpha)

            if epoch == 1:
                img_grid = vutils.make_grid(
                    images, nrow=10, normalize=True, scale_each=True)
                writer.add_image("Augmented image", img_grid, i)

            outputs = model(images)

            class_outputs, reg_outputs = outputs 

            if args.label_smooth:
                if args.mixup:
                    soft_labels_a = smooth_one_hot(
                        labels_a, classes=11, smoothing=args.label_smooth_value)
                    soft_labels_b = smooth_one_hot(
                        labels_b, classes=11, smoothing=args.label_smooth_value)
                    loss_class = mixup_criterion(
                        loss_fn, class_outputs, soft_labels_a, soft_labels_b, lam)
                else:
                    soft_labels = smooth_one_hot(
                        labels_x, classes=11, smoothing=args.label_smooth_value)
                    loss_class = loss_fn(class_outputs, soft_labels)
            else:
                loss_class = loss_fn(class_outputs, labels_x)


            loss_reg = nn.MSELoss()(reg_outputs, labels_y.float()) 
            loss = loss_class + loss_reg

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        # if args.Ncrop:
        #     bs, ncrops, c, h, w = org_images.shape
        #     org_images = org_images.view(-1, c, h, w)
        #     org_labels = torch.repeat_interleave(org_labels, repeats=ncrops, dim=0)
        #     labels_y= torch.repeat_interleave(labels_y, repeats=ncrops, dim=0)
        _, preds = torch.max(class_outputs, 1)
        all_preds.append(preds)
        all_labels.append(labels_x)
        correct += torch.sum(preds == org_labels.data).item()
        count += labels_x.shape[0]

        if len(reg_outputs.shape) > 1:
            reg_outputs = reg_outputs.squeeze()
            
        reg_outputs = reg_outputs * (28.5 - 9.0) + 9.0 
        labels_y = labels_y * (28.5 - 9.0) + 9.0 

        all_scores.append(labels_y)
        all_pred_scores.append(reg_outputs.detach())

    all_preds = np.concatenate([pred.cpu().numpy() for pred in all_preds])
    all_labels = np.concatenate([label.cpu().numpy() for label in all_labels])
    all_scores = np.concatenate([score.cpu().numpy() for score in all_scores])
    all_pred_scores = np.concatenate([score.cpu().numpy() for score in all_pred_scores])


    spearman = spearman_corr(torch.tensor(all_scores), torch.tensor(all_pred_scores))
    l2_distance = relative_l2_distance(torch.tensor(all_scores), torch.tensor(all_pred_scores) ,torch.tensor(all_labels) )

    return train_loss / count, correct / count, all_preds, all_labels , spearman , l2_distance



def evaluate(model, val_loader, device, args):
    model.eval()
    count = 0
    correct = 0
    val_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []  
    all_pred_scores = [] 

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels_x, labels_y = data
            images = images.to(device)
            labels_x = labels_x.to(device)
            labels_y = labels_y.to(device)
            
            # if args.Ncrop:
            #     bs, ncrops, c, h, w = images.shape
            #     images = images.view(-1, c, h, w)

            outputs = model(images)

            class_outputs, reg_outputs = outputs

            #     class_outputs = class_outputs.view(bs, ncrops, -1) 
            #     reg_outputs = reg_outputs.view(bs, ncrops, -1)
            #     class_outputs = torch.sum(class_outputs, dim=1) / ncrops
            #     reg_outputs = torch.sum(reg_outputs, dim=1) / ncrops 

            # else:
            #     outputs = model(images)
            #     class_outputs, reg_outputs = outputs

            loss = nn.CrossEntropyLoss()(class_outputs, labels_x)

            val_loss += loss.item()
            _, preds = torch.max(class_outputs, 1)
            all_preds.append(preds)
            all_labels.append(labels_x)
            correct += torch.sum(preds == labels_x.data).item()
            count += labels_x.shape[0]

            if len(reg_outputs.shape) > 1:
                reg_outputs = reg_outputs.squeeze() 
                
            reg_outputs = reg_outputs * (28.5 - 9.0) + 9.0 
            labels_y = labels_y * (28.5 - 9.0) + 9.0 

            all_scores.append(labels_y)
            all_pred_scores.append(reg_outputs.detach())

        all_preds = np.concatenate([pred.cpu().numpy() for pred in all_preds])
        all_labels = np.concatenate([label.cpu().numpy() for label in all_labels])
        all_scores = np.concatenate([score.cpu().numpy() for score in all_scores])
        all_pred_scores = np.concatenate([score.cpu().numpy() for score in all_pred_scores])


        spearman = spearman_corr(torch.tensor(all_scores), torch.tensor(all_pred_scores))
        l2_distance = relative_l2_distance(torch.tensor(all_scores), torch.tensor(all_pred_scores) , torch.tensor(all_labels))

        return val_loss / count, correct / count, all_preds, all_labels , spearman , l2_distance



def main():
    global best_acc

    args = parser.parse_args()

    args_path = str(args.arch) + '_epoch' + str(args.epochs) + '_bs' + str(args.batch_size) + '_lr' + str(
        args.lr) + '_momentum' + str(args.momentum) + '_wd' + str(args.weight_decay) + '_seed' + str(
        args.seed) + '_smooth' + str(args.label_smooth) + '_mixup' + str(args.mixup) + '_scheduler' + str(
        args.scheduler) + '_' + str(args.name)

    checkpoint_path = os.path.join(
        args.results, args.name, args_path, 'checkpoints')

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    writer = SummaryWriter(os.path.join(
        args.results, args.name, args_path, 'tensorboard_logs'))

    logger = Logger(os.path.join(args.results,
                                 args.name, args_path, 'output.log'))

    logger.info(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(device)

    logger.info('Load dataset ...')

    train_loader, val_loader = get_dataloaders(
    train_pkl='datasets/train.pkl',
    val_pkl='datasets/val.pkl',
    bs=args.batch_size, 
    augment=True
)


    logger.info('Start load model %s ...', args.arch)
    model = get_model(args.arch)
    model = model.to(device)
    scaler = GradScaler()

    if args.label_smooth:
        loss_fn = cross_entropy
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    
    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.75, patience=5, verbose=True)

    if args.resume > 0:
        logger.info('Resume from epoch %d', (args.resume))
        state_dict = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_' + str(args.resume) + '.tar'))
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['opt_state_dict'])

    logger.info('Start training.')
    logger.info(
        "Epoch \t Time \t Train Loss \t Train ACC \t Val Loss \t Val ACC \t Spearman (Val) \t L2 Distance (Val)")


    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        train_loss, train_acc, train_preds, train_labels, train_spearman, train_l2 = train(
            model, train_loader, loss_fn, optimizer, epoch, device, scaler, writer, args)
        val_loss, val_acc, val_preds, val_labels, val_spearman, val_l2 = evaluate(model, val_loader, device, args)

        # 计算分类指标
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        train_precision = precision_score(train_labels, train_preds, average='macro')
        val_precision = precision_score(val_labels, val_preds, average='macro')
        train_recall = recall_score(train_labels, train_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')

        writer.add_scalar("Train/F1", train_f1, epoch)
        writer.add_scalar("Train/Precision", train_precision, epoch)
        writer.add_scalar("Train/Recall", train_recall, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)
        writer.add_scalar("Val/Precision", val_precision, epoch)
        writer.add_scalar("Val/Recall", val_recall, epoch)

        epoch_time = time.time() - start_t
        logger.info(f"{epoch:<5} {epoch_time:<10.4f} {train_loss:<15.4f} {train_acc:<15.4f} "
                f"{val_loss:<15.4f} {val_acc:<15.4f} {val_spearman:<20.4f} {val_l2:<15.4f}")
        
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'reduce':
            scheduler.step(val_acc)

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Valid/Loss", val_loss, epoch)
        writer.add_scalar("Valid/Accuracy", val_acc, epoch)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        writer.add_scalar("Valid/Best Accuracy", best_acc, epoch)
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, epoch, is_best, save_path=checkpoint_path, save_freq=args.save_freq)

    logger.info("Best val ACC %.4f", best_acc)
    writer.close()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()