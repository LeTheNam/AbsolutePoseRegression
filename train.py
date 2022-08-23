import os

import torch
import sys
import time
import os.path as osp
import numpy as np

from tensorboardX import SummaryWriter
from tools.options import Options
# from network.atloc import AtLoc, AtLocPlus
from network.atloc_modi import AtLoc, AtLocPlus
from torchvision import transforms, models
from tools.utils import AtLocCriterion, AtLocPlusCriterion, AverageMeter, Logger

from data.dataloader_robot import Robot
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
logfile = osp.join(opt.runs_dir, 'log.txt')
stdout = Logger(logfile)
print('Logging to {:s}'.format(logfile))
sys.stdout = stdout

# Model

if opt.backbone == 'resnet18':
    feature_extractor = models.resnet18(pretrained=False)
    feature_extractor.load_state_dict(torch.load("./weights/resnet18-5c106cde.pth", map_location='cpu'))
    feature_extractor.to(device)

elif opt.backbone == 'resnet34':
    feature_extractor = models.resnet34(pretrained=False)
    feature_extractor.load_state_dict(torch.load("./weights/resnet34-333f7ec4.pth", map_location='cpu'))
    feature_extractor.to(device)

elif opt.backbone == 'resnet50':
    feature_extractor = models.resnet50(pretrained=False)
    feature_extractor.load_state_dict(torch.load("./weights/resnet50-19c8e357.pth", map_location='cpu'))
    feature_extractor.to(device)

elif opt.backbone == 'efficientnet_v2':
    # torchvision >= 0.13 
    from torchvision.models import efficientnet_v2_s
    feature_extractor = efficientnet_v2_s(weights='DEFAULT')
    feature_extractor.to(device)
    
elif opt.backbone == 'efficientnet_b4':
    # input size: 320
    # + opt.resize -> 352
    # + opt.crop -> 320
    import timm
    feature_extractor = timm.create_model('efficientnet_b4', pretrained=False)
    feature_extractor.load_state_dict(torch.load("./weights/efficientnet_b4_ra2_320-7eb33cd5.pth", map_location='cpu'))
    feature_extractor.to(device)

elif opt.backbone == 'efficientnet_b3':
    # input size: 288
    # + opt.resize -> 320
    # + opt.crop -> 288
    import timm
    feature_extractor = timm.create_model('efficientnet_b3', pretrained=False)
    feature_extractor.load_state_dict(torch.load("./weights/efficientnet_b3_ra2-cf984f9c.pth", map_location='cpu'))
    feature_extractor.to(device)

elif opt.backbone == 'swin':
    from torchvision.models import swin_s
    feature_extractor = swin_s(weights='DEFAULT')
    feature_extractor.to(device)

else:
    feature_extractor = models.resnet34(pretrained=False)
    feature_extractor.load_state_dict(torch.load("./weights/resnet34-333f7ec4.pth", map_location=device))
    feature_extractor.to(device)

model = AtLoc(feature_extractor, droprate=opt.train_dropout, feat_dim=opt.feat_dim, pretrained=True, lstm=opt.lstm)
print(model)
train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)
# val_criterion = AtLocCriterion()


# Optimizer
param_list = [{'params': model.parameters()}]
if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq'):
    print('learn_beta')
    param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if opt.gamma is not None and hasattr(train_criterion, 'srx') and hasattr(train_criterion, 'srq'):
    print('learn_gamma')
    param_list.append({'params': [train_criterion.srx, train_criterion.srq]})

optimizer = torch.optim.Adam(param_list, lr=opt.lr, weight_decay=opt.weight_decay)

# stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
# stats = np.loadtxt(stats_file)

tforms = [transforms.Resize(opt.resize), transforms.CenterCrop(opt.cropsize)]
if opt.color_jitter > 0:
    assert opt.color_jitter <= 1.0
    print('Using ColorJitter data augmentation')
    tforms.append(
        transforms.ColorJitter(brightness=opt.color_jitter, contrast=opt.color_jitter, saturation=opt.color_jitter,
                               hue=0.5))
else:
    print('Not Using ColorJitter')
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])))
data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# Load the dataset
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, transform=data_transform, target_transform=target_transform,
              seed=opt.seed)
if opt.model == 'AtLoc':
    if opt.dataset == 'Robot':
        print(f"Loading {opt.dataset} data ...")
        train_set = Robot(train=True, **kwargs)
        val_set = Robot(train=False, **kwargs)
    else:
        raise NotImplementedError
else:
    raise NotImplementedError

kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True, **kwargs)
val_loader = DataLoader(val_set, batch_size=opt.batchsize, shuffle=False, **kwargs)

model.to(device)
train_criterion.to(device)
# val_criterion.to(device)

total_steps = opt.steps
writer = SummaryWriter(log_dir=opt.runs_dir)
experiment_name = opt.exp_name
for epoch in range(opt.epochs):
    if (epoch + 1) % opt.val_freq == 0 or epoch == (opt.epochs - 1):
        val_batch_time = AverageMeter()
        val_loss = AverageMeter()
        model.eval()
        end = time.time()
        val_data_time = AverageMeter()
        t_loss_avg, q_loss_avg = 0.0, 0.0
        for batch_idx, (val_data, val_target, image_file) in enumerate(val_loader):
            val_data_time.update(time.time() - end)
            val_data_var = Variable(val_data, requires_grad=False)
            val_target_var = Variable(val_target, requires_grad=False)
            val_data_var = val_data_var.to(device)
            val_target_var = val_target_var.to(device)

            with torch.set_grad_enabled(False):
                val_output = model(val_data_var)
                val_loss_tmp, t_loss, q_loss = train_criterion(val_output, val_target_var)
                t_loss_avg += t_loss
                q_loss_avg += q_loss
                val_loss_tmp = val_loss_tmp.item()

            val_loss.update(val_loss_tmp)
            val_batch_time.update(time.time() - end)

            writer.add_scalar('val_err', val_loss_tmp, total_steps)
            if batch_idx % opt.print_freq == 0:
                print(
                    'Val {:s}: Epoch {:d}\tBatch {:d}/{:d}\tLoss {:f}' \
                        .format(experiment_name, epoch, batch_idx, len(val_loader) - 1, val_loss_tmp))
            end = time.time()
        print("t_loss: {}, q_loss: {}".format(t_loss_avg/len(val_loader), q_loss_avg/len(val_loader)))

        print('Val {:s}: Epoch {:d}, val_loss {:f}'.format(experiment_name, epoch, val_loss.avg))

        if (epoch + 1) % opt.save_freq == 0:
            filename = osp.join(opt.models_dir, 'epoch_{:03d}.pth.tar'.format(epoch+1))
            checkpoint_dict = {'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                               'optim_state_dict': optimizer.state_dict(),
                               'criterion_state_dict': train_criterion.state_dict()}
            torch.save(checkpoint_dict, filename)
            print('Epoch {:d} checkpoint saved for {:s}'.format(epoch+1, experiment_name))

    model.train()
    train_data_time = AverageMeter()
    train_batch_time = AverageMeter()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        train_data_time.update(time.time() - end)

        data_var = Variable(data, requires_grad=True)
        target_var = Variable(target, requires_grad=False)
        data_var = data_var.to(device)
        target_var = target_var.to(device)
        with torch.set_grad_enabled(True):
            output = model(data_var)
            loss_tmp, t_loss_train, q_loss_train = train_criterion(output, target_var)
        loss_tmp.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_batch_time.update(time.time() - end)
        writer.add_scalar('train_err', loss_tmp.item(), total_steps)
        if batch_idx % opt.print_freq == 0:
            print(
                # 'Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                #     .format(experiment_name, epoch, batch_idx, len(train_loader) - 1, train_data_time.val,
                #             train_data_time.avg, train_batch_time.val, train_batch_time.avg, loss_tmp.item()))
                'Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tLoss {:f}' \
                    .format(experiment_name, epoch, batch_idx, len(train_loader) - 1, loss_tmp.item()))
        end = time.time()

writer.close()
