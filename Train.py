import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import torch.backends.cudnn as cudnn

import IQADataset
from utils import performance_fit
import logging
from models.ResEV import ResEVIQA
# from models.RUS_EV2S import RUS2IQA
import os

# 配置日志模块
logging.basicConfig(
    filename='training_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    """Parse input arguments. """

    parser = argparse.ArgumentParser(description="In the wild Image Quality Assessment")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs', help='Maximum number of training epochs.', default=80, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=8, type=int)
    parser.add_argument('--resize', help='resize.', default=384, type=int)
    parser.add_argument('--crop_size', help='crop_size.', default=224, type=int)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=10)
    parser.add_argument('--snapshot', help='Path of model snapshot.', type=str,
                        default='D:\浏览器下载\StairIQA-main\histor_ypkl\Ablation\sage\koniq')
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--database_dir', type=str,
                        default=r'E:/shujuji\/koniq10k_1024x768/1024x768')
    parser.add_argument('--model', default='RUS2IQA', type=str)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--print_samples', type=int, default=201)
    parser.add_argument('--database', default='Koniq10k', type=str)
    parser.add_argument('--test_method', default='one', type=str,
                        help='use the center crop or five crop to test the image')

    args = parser.parse_args()

    return args


def normalize_tensor(crop):
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop)


def to_tensor(crop):
    return transforms.ToTensor()(crop)


def stack_and_normalize(crops):
    return torch.stack([normalize_tensor(crop) for crop in crops])


def stack_and_to_tensor(crops):
    return torch.stack([to_tensor(crop) for crop in crops])


if __name__ == '__main__':
    args = parse_args()

    gpu = args.gpu
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    decay_interval = args.decay_interval
    decay_ratio = args.decay_ratio
    snapshot = args.snapshot
    database = args.database
    print_samples = args.print_samples
    results_path = args.results_path
    database_dir = args.database_dir
    resize = args.resize
    crop_size = args.crop_size

    best_all = np.zeros([10, 2])
    logging.info("**************************************************************************************************")
    logging.info("**************************************************************************************************")
    logging.info(f"Model = {args.model}, Database = {database}, Batch_size = {batch_size}, Num_epochs = {num_epochs}")
    logging.info("**************************************************************************************************")
    logging.info("**************************************************************************************************")
    for exp_id in range(10):

        print('The current exp_id is ' + str(exp_id))
        if not os.path.exists(snapshot):
            os.makedirs(snapshot)
        trained_model_file = os.path.join(snapshot,
                                          'train-ind-{}-{}-exp_id-{}.pkl'.format(database, args.model, exp_id))

        print('The save model name is ' + trained_model_file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if database == 'Koniq10k':
            train_filename_list = 'csvfiles/Koniq10k_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/Koniq10k_test_' + str(exp_id) + '.csv'

        elif database == 'FLIVE':
            train_filename_list = 'csvfiles/FLIVE_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/FLIVE_test_' + str(exp_id) + '.csv'
        elif database == 'FLIVE_patch':
            train_filename_list = 'csvfiles/FLIVE_patch_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/FLIVE_patch_test_' + str(exp_id) + '.csv'
        elif database == 'LIVE_challenge':
            train_filename_list = 'csvfiles/LIVE_challenge_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/LIVE_challenge_test_' + str(exp_id) + '.csv'
        elif database == 'SPAQ':
            train_filename_list = 'csvfiles/SPQA_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/SPQA_test_' + str(exp_id) + '.csv'
        elif database == 'BID':
            train_filename_list = 'csvfiles/BID_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/BID_test_' + str(exp_id) + '.csv'

        print(train_filename_list)
        print(test_filename_list)

        # load the network
        if args.model == 'ResEVIQA':
            model = ResEVIQA()
        # if args.model == 'RUS2IQA':
        #     model = RUS2IQA()  ##四分支
       # 消融实验：

        # 只对image进行resize和crop
        transformations_train_image = transforms.Compose([transforms.Resize(resize), transforms.RandomCrop(crop_size),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                               std=[0.229, 0.224, 0.225])])
        transformations_train_cropped_img = transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                     std=[0.229, 0.224, 0.225])])

        if args.test_method == 'one':
            transformations_test_image = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(crop_size),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])])
            transformations_test_cropped_img = transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])])
        elif args.test_method == 'five':
            transformations_test_image = transforms.Compose([
                transforms.Resize(resize),
                transforms.FiveCrop(crop_size),
                stack_and_to_tensor,
                stack_and_normalize
            ])
            transformations_test_cropped_img = transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])])

        train_dataset = IQADataset.IQA_dataloader(database_dir, train_filename_list,
                                                  (transformations_train_image, transformations_train_cropped_img),
                                                  database)
        test_dataset = IQADataset.IQA_dataloader(database_dir, test_filename_list,
                                                 (transformations_test_image, transformations_test_cropped_img),
                                                 database)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=8)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

        if args.multi_gpu:
            model = torch.nn.DataParallel(model)
            model = model.to(device)
        else:
            model = model.to(device)

        criterion = nn.MSELoss().to(device)

        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0000001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=decay_ratio)

        print("Ready to train network")


        best_test_criterion = -1  # SROCC min
        best = np.zeros(2)

        n_train = len(train_dataset)
        n_test = len(test_dataset)

        for epoch in range(num_epochs):
            # train
            model.train()

            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            for i, (image, cropped_img, mos) in enumerate(train_loader):
                image = image.to(device)
                cropped_img = cropped_img.to(device)  # 将 cropped_img 移动到 GPU 上
                mos = mos[:, np.newaxis]
                mos = mos.to(device)

                mos_output = model(image, cropped_img)

                loss = criterion(mos_output, mos)
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())

                optimizer.zero_grad()  # clear gradients for next train
                torch.autograd.backward(loss)
                optimizer.step()

                if (i + 1) % print_samples == 0:
                    session_end_time = time.time()
                    avg_loss_epoch = sum(batch_losses_each_disp) / print_samples
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' %
                          (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                           avg_loss_epoch))
                    batch_losses_each_disp = []
                    print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                    session_start_time = time.time()

            avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))
            logging.info('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))


            scheduler.step()
            lr_current = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr_current[0]))

            # Test
            model.eval()
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)

            with torch.no_grad():
                for i, (image, cropped_img, mos) in enumerate(test_loader):
                    if args.test_method == 'one':
                        image = image.to(device)
                        cropped_img = cropped_img.to(device)  # 将 cropped_img 移动到 GPU 上
                        y_test[i] = mos.item()
                        mos = mos.to(device)
                        outputs = model(image, cropped_img)
                        y_output[i] = outputs.item()


                    elif args.test_method == 'five':
                        bs, ncrops, c, h, w = image.size()
                        y_test[i] = mos.item()
                        image = image.to(device)
                        cropped_img = cropped_img.to(device)
                        mos = mos.to(device)

                        outputs = model(image.view(-1, c, h, w), cropped_img)
                        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                        y_output[i] = outputs_avg.item()

                test_PLCC, test_SRCC = performance_fit(y_test, y_output)[:2]
                print("Test results: SROCC={:.4f}, PLCC={:.4f}".format(test_SRCC, test_PLCC))
                # 记录日志
                logging.info(f"Epoch {epoch + 1}, Exp ID {exp_id}: SROCC={test_SRCC:.4f}, PLCC={test_PLCC:.4f}")


                if test_SRCC > best_test_criterion:
                    print("Update best model using best_val_criterion ")
                    torch.save(model.state_dict(), trained_model_file)
                    best[0:2] = [test_SRCC, test_PLCC]
                    best_test_criterion = test_SRCC  # update best val SROCC

                    print(
                        "The best Test results: SROCC={:.4f}, PLCC={:.4f}".format(test_SRCC, test_PLCC))

        print(database)
        best_all[exp_id, :] = best
        print("The best Val results: SROCC={:.4f}, PLCC={:.4f}".format(best[0], best[1]))
        # 记录最佳验证结果到日志
        logging.info(f"*******************************************************************************")
        logging.info("The best Val results: SROCC={:.4f}, PLCC={:.4f}".format(best[0], best[1]))
        logging.info(f"*******************************************************************************")
        print(
            '*************************************************************************************************************************')

    best_median = np.median(best_all, 0)
    best_mean = np.mean(best_all, 0)
    best_std = np.std(best_all, 0)
    print(
        '*************************************************************************************************************************')
    print(best_all)
    print("The median val results: SROCC={:.4f}, PLCC={:.4f}".format(best_median[0], best_median[1]))
    print(
        "The mean val results: SROCC={:.4f}, PLCC={:.4f}".format(best_mean[0], best_mean[1]))
    print("The std val results: SROCC={:.4f}, PLCC={:.4f}".format(best_std[0], best_std[1]))
    print(
        '*************************************************************************************************************************')
    # 记录中位数、均值和标准差验证结果到日志
    logging.info("The median val results: SROCC={:.4f}, PLCC={:.4f}".format(best_median[0], best_median[1]))
    logging.info("The mean val results: SROCC={:.4f}, PLCC={:.4f}".format(best_mean[0], best_mean[1]))
    logging.info("The std val results: SROCC={:.4f}, PLCC={:.4f}".format(best_std[0], best_std[1]))