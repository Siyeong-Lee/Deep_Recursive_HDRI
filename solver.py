import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
from torch.utils.data import DataLoader
import utils
#from logger import Logger
from model import *

from data_loader import *
from torchvision import transforms
import numpy as np

class Solver(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.patch_size = args.patch_size
        self.num_threads = args.num_threads
        self.exposure_value = args.exposure_value
        self.num_channels = args.num_channels

        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr

        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode

        self.stride = args.stride
     
        self.build_model()

    def build_model(self):
        # networks
        self.stopup_G = Generator(num_channels=self.num_channels, base_filter=64, stop='up')
        self.stopdown_G = Generator(num_channels=self.num_channels, base_filter=64, stop='down')
        
        self.stopup_D = NLayerDiscriminator(num_channels=2*self.num_channels,base_filter=64, image_size=self.patch_size)
        self.stopdown_D = NLayerDiscriminator(num_channels=2*self.num_channels,base_filter=64, image_size=self.patch_size)

        print('---------- Networks architecture -------------')
        utils.print_network(self.stopup_G)
        utils.print_network(self.stopdown_D)
        print('----------------------------------------------')

        # weigh initialization
        self.stopup_G.weight_init()
        self.stopdown_G.weight_init()
 
        self.stopup_D.weight_init()
        self.stopdown_D.weight_init()        

        # optimizer
        self.stopup_G_optimizer = optim.Adam(self.stopup_G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.stopdown_G_optimizer = optim.Adam(self.stopdown_G.parameters(), lr=self.lr, betas=(0.5, 0.999)) 

        self.stopup_D_optimizer = optim.Adam(self.stopup_D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.stopdown_D_optimizer = optim.Adam(self.stopdown_D.parameters(), lr=self.lr, betas=(0.5, 0.999)) 

        # loss function
        if self.gpu_mode:
            self.stopup_G = nn.DataParallel(self.stopup_G)
            self.stopdown_G = nn.DataParallel(self.stopdown_G)
            
            self.stopup_D = nn.DataParallel(self.stopup_D)
            self.stopdown_D = nn.DataParallel(self.stopdown_D)

            self.stopup_G.cuda()
            self.stopdown_G.cuda()
 
            self.stopup_D.cuda()
            self.stopdown_D.cuda()

            self.L1_loss = nn.L1Loss().cuda()
            self.criterionGAN = GANLoss().cuda()
        
        else:
            self.L1_loss = nn.L1Loss()
            self.MSE_loss = nn.MSELoss()
            self.BCE_loss = nn.BCELoss()
            self.criterionGAN = GANLoss()

        return

    def load_dataset(self, dataset, is_train=True):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if is_train:
            print('Loading train datasets...')
            train_set = get_loader(self.train_dataset)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        else:
            print('Loading test datasets...')
            test_set = get_loader(self.test_dataset)
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)

    def train(self):
        # load dataset
        train_data_loader = self.load_dataset(dataset=self.train_dataset, is_train=True)
        test_data_loader = self.load_dataset(dataset=self.test_dataset[0], is_train=False)

        # set the logger
        stopup_G_log_dir = os.path.join(self.save_dir, 'stopup_G_logs')
        if not os.path.exists(stopup_G_log_dir):
            os.mkdir(stopup_G_log_dir)
        #stopup_G_logger = Logger(stopup_G_log_dir)

        stopup_D_log_dir = os.path.join(self.save_dir, 'stopup_D_logs')
        if not os.path.exists(stopup_D_log_dir):
            os.mkdir(stopup_D_log_dir)
        #stopup_D_logger = Logger(stopup_D_log_dir)


        stopdown_G_log_dir = os.path.join(self.save_dir, 'stopdown_G_logs')
        if not os.path.exists(stopdown_G_log_dir):
            os.mkdir(stopdown_G_log_dir)
        #stopdown_G_logger = Logger(stopdown_G_log_dir)

        stopdown_D_log_dir = os.path.join(self.save_dir, 'stopdown_D_logs')
        if not os.path.exists(stopdown_D_log_dir):
            os.mkdir(stopdown_D_log_dir)
        #stopdown_D_logger = Logger(stopdown_D_log_dir)


        ################# Pre-train generator #################
        self.epoch_pretrain = 10

        # Load pre-trained parameters of generator
        if not self.load_model(is_pretrain=True):
            # Pre-training generator for 10 epochs
            print('Pre-training is started.')
            self.stopup_G.train()
            self.stopdown_G.train()

            for epoch in range(self.epoch_pretrain):
                for iter, (lr, hr) in enumerate(train_data_loader):
                    # input data (low dynamic image)
                    if self.num_channels == 1:
                        x_ = Variable(utils.norm(hr[:, 0].unsqueeze(1), vgg=False))
                        y_ = Variable(utils.norm(lr[:, 0].unsqueeze(1), vgg=False))
                    else:
                        x_ = Variable(utils.norm(hr, vgg=False))
                        y_ = Variable(utils.norm(lr, vgg=False))

                    if self.gpu_mode:
                        x_ = x_.cuda()
                        y_ = y_.cuda()

                    # Train generator
                    self.stopup_G_optimizer.zero_grad()
                    self.stopdown_G_optimizer.zero_grad()                    

                    '''
                    stopup
                    '''
                    stopup_est = self.stopup_G(y_)

                    # Content losses
                    stopup_content_loss = self.L1_loss(stopup_est, x_)
                    stopup_G_loss = stopup_content_loss

                    stopup_G_loss.backward()
                    self.stopup_G_optimizer.step()

                    '''
                    stopdown
                    '''
                    stopdown_est = self.stopdown_G(x_)

                    # Content losses
                    stopdown_content_loss = self.L1_loss(stopdown_est, y_)
                    stopdown_G_loss = stopdown_content_loss

                    stopdown_G_loss.backward()
                    self.stopdown_G_optimizer.step()

                    # log
                    print("Epoch: [%2d] [%4d/%4d] stopup_G: %.6f/stopdown_G: %.6f"
                          % ((epoch + 1), (iter + 1), len(train_data_loader), stopup_G_loss.data[0], stopdown_G_loss.data[0]), end='\r')

                    # siyeong
                    if (iter % 100 == 0):
                        import random
                        index = random.randrange(0,self.batch_size)
                         
                        input_data = torch.cat((y_[index], x_[index]), 1)
                        est_data = torch.cat((stopup_est[index], stopdown_est[index]),1)
                        square = torch.cat((input_data, est_data), 2)                   
                        square = utils.denorm(square.cpu().data, vgg=False)

                        square_img = transforms.ToPILImage()(square)

                        square_img.show()

            print('Pre-training is finished.')

            # Save pre-trained parameters of generator
            self.save_model(is_pretrain=True)

        ################# Adversarial train #################
        print('Training is started.')
        # Avg. losses
        stopup_G_avg_loss = []
        stopup_D_avg_loss = []
        stopdown_G_avg_loss = []
        stopdown_D_avg_loss = []
 
        step = 0

        # test image
        test_lr, test_hr = test_data_loader.dataset.__getitem__(2)
        test_lr = test_lr.unsqueeze(0)
        test_hr = test_hr.unsqueeze(0)

        self.stopup_G.train()
        self.stopup_D.train()

        self.stopdown_G.train()
        self.stopdown_D.train()

        for epoch in range(self.num_epochs):
            # learning rate is decayed by a factor of 10 every 20 epoch
            if (epoch + 1) % 20 == 0:
                for param_group in self.stopup_G_optimizer.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay for G: lr={}".format(self.stopup_G_optimizer.param_groups[0]["lr"]))
                for param_group in self.stopup_D_optimizer.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay for D: lr={}".format(self.stopup_D_optimizer.param_groups[0]["lr"]))

                for param_group in self.stopdown_G_optimizer.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay for G: lr={}".format(self.stopdown_G_optimizer.param_groups[0]["lr"]))
                for param_group in self.stopdown_D_optimizer.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay for D: lr={}".format(self.stopdown_D_optimizer.param_groups[0]["lr"]))

            stopup_G_epoch_loss = 0
            stopup_D_epoch_loss = 0

            stopdown_G_epoch_loss = 0
            stopdown_D_epoch_loss = 0

            for iter, (lr, hr) in enumerate(train_data_loader):
                # input data (low dynamic image)
                mini_batch = lr.size()[0]

                if self.num_channels == 1:
                    x_ = Variable(utils.norm(hr[:, 0].unsqueeze(1), vgg=False))
                    y_ = Variable(utils.norm(lr[:, 0].unsqueeze(1), vgg=False))
                else:
                    x_ = Variable(utils.norm(hr, vgg=False))
                    y_ = Variable(utils.norm(lr, vgg=False))

                if self.gpu_mode:
                    x_ = x_.cuda()
                    y_ = y_.cuda()
                    # labels
                    real_label = Variable(torch.ones(mini_batch).cuda())
                    fake_label = Variable(torch.zeros(mini_batch).cuda())
                else:
                    # labels
                    real_label = Variable(torch.ones(mini_batch))
                    fake_label = Variable(torch.zeros(mini_batch))

                # Reset gradient
                self.stopup_D_optimizer.zero_grad()
                self.stopdown_D_optimizer.zero_grad()

                # Train discriminator with real data
                stopup_D_real_decision = self.stopup_D(torch.cat((x_, y_),1))
                stopdown_D_real_decision = self.stopdown_D(torch.cat((y_, x_),1))

                stopup_D_real_loss = self.criterionGAN(stopup_D_real_decision, True)
                stopdown_D_real_loss = self.criterionGAN(stopdown_D_real_decision, True)

                # Train discriminator with fake data
                stopup_est = self.stopup_G(y_)
                stopdown_est = self.stopdown_G(x_)

                stopup_D_fake_decision = self.stopup_D(torch.cat((stopup_est, y_),1))
                stopdown_D_fake_decision = self.stopdown_D(torch.cat((stopdown_est, x_),1))

                stopup_D_fake_loss = self.criterionGAN(stopup_D_fake_decision, False)
                stopdown_D_fake_loss = self.criterionGAN(stopdown_D_fake_decision, False)

                stopup_D_loss = 0.5*stopup_D_real_loss + 0.5*stopup_D_fake_loss
                stopdown_D_loss = 0.5*stopdown_D_real_loss + 0.5*stopdown_D_fake_loss

                # Back propagation
                stopup_D_loss.backward(retain_graph=True)
                stopdown_D_loss.backward(retain_graph=True)

                self.stopup_D_optimizer.step()
                self.stopdown_D_optimizer.step()

                # Reset gradient
                self.stopup_G_optimizer.zero_grad()
                self.stopdown_G_optimizer.zero_grad()

                # Train generator
                stopup_est = self.stopup_G(y_)
                stopdown_est = self.stopdown_G(x_)

                stopup_D_fake_decision = self.stopup_D(torch.cat((stopup_est, y_), 1))
                stopdown_D_fake_decision = self.stopdown_D(torch.cat((stopdown_est, x_), 1))

                # Adversarial loss
                stopup_GAN_loss = self.criterionGAN(stopup_D_fake_decision, True)
                stopdown_GAN_loss = self.criterionGAN(stopdown_D_fake_decision, True)

                # Content losses
                stopup_mae_loss = self.L1_loss(stopup_est, x_) 
                stopdown_mae_loss = self.L1_loss(stopdown_est, y_)

                # Total loss 
                stopup_G_loss =  stopup_mae_loss + 1e-2*stopup_GAN_loss 
                stopdown_G_loss =  stopdown_mae_loss + 1e-2*stopdown_GAN_loss 

                stopup_G_loss.backward()
                self.stopup_G_optimizer.step()

                stopdown_G_loss.backward()
                self.stopdown_G_optimizer.step()

                # siyeong
                if (iter % 100 == 0):
                    import random
                    index = random.randrange(0,self.batch_size)
                         
                    input_data = torch.cat((y_[index], x_[index]), 1)
                    est_data = torch.cat((stopup_est[index], stopdown_est[index]),1)

                    square = torch.cat((input_data, est_data), 2)

                    square = utils.denorm(square.cpu().data, vgg=False)
                    square_img = transforms.ToPILImage()(square)

                    square_img.show()

                # log
                stopup_G_epoch_loss += stopup_G_loss.data[0]
                stopup_D_epoch_loss += stopup_D_loss.data[0]

                stopdown_G_epoch_loss += stopdown_G_loss.data[0]
                stopdown_D_epoch_loss += stopdown_D_loss.data[0]

                print("Epoch: [%02d] [%05d/%05d] stopup_G/D: %.6f/%.6f, stopdown_G/D: %.6f/%.6f"
                      % ((epoch + 1), (iter + 1), len(train_data_loader), stopup_G_loss.data[0], stopup_D_loss.data[0], stopdown_G_loss.data[0], stopdown_D_loss.data[0]), end="\r")

                # tensorboard logging
                stopup_G_logger.scalar_summary('losses', stopup_G_loss.data[0], step + 1)
                stopup_D_logger.scalar_summary('losses', stopup_D_loss.data[0], step + 1)

                stopdown_G_logger.scalar_summary('losses', stopdown_G_loss.data[0], step + 1)
                stopdown_D_logger.scalar_summary('losses', stopdown_D_loss.data[0], step + 1)
 
                step += 1

            # avg. loss per epoch
            stopup_G_avg_loss.append(stopup_G_epoch_loss / len(train_data_loader))
            stopup_D_avg_loss.append(stopup_D_epoch_loss / len(train_data_loader))

            stopdown_G_avg_loss.append(stopdown_G_epoch_loss / len(train_data_loader))
            stopdown_D_avg_loss.append(stopdown_D_epoch_loss / len(train_data_loader))

            self.save_model(epoch + 1)

        # Plot avg. loss
        utils.plot_loss([stopup_G_avg_loss, stopup_D_avg_loss, stopdown_G_avg_loss, stopdown_D_avg_loss], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    # siyeong3
    def test(self, input_path='./', out_path='./result/', extend = 3):
        # load model
        self.load_model(is_pretrain=False)
        scenes = listdir(input_path)
    
        for i, scene in enumerate(scenes):
            scene_path = join(input_path, scene)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
         
            filelist = [join(scene_path, x) for x in sorted(listdir(scene_path)) if 'EV0' in x]

            for filepath in filelist:
                out_name = os.path.splitext(os.path.split(filepath)[1])[0]
                out_name = '%04d'%(i)
                storage_path = out_path + out_name + '/'

                # mkdir storage folder
                if not os.path.isdir(storage_path):
                    os.mkdir(storage_path)

                # cp middle exposure file
                cmd = "cp " + filepath + " " + storage_path + out_name + '_EV0.png'            
                os.system(cmd)

                target = filepath
                for i in range(1, extend+1):
                    reconst = self.image_single(target, True)
                    output_name = storage_path + out_name + '_EV%d' %i + '.png'
                    reconst.save(output_name)

                    target = output_name

                target = filepath
                for i in range(1, extend+1):
                    reconst = self.image_single(target, False)
                    output_name = storage_path + out_name +'_EV-%d' %i + '.png'
                    reconst.save(output_name)

                    target = output_name
                print('\tImage [', out_name, '] is finished.')
            print('Test is finishied.')

    def image_single(self, img_fn, stopup):
        # load data
        img = Image.open(img_fn).convert('RGB')
        
        img = img.resize((256, 256), 4)
        tensor = transforms.ToTensor()(img)
        tensor_norm = Variable(utils.norm(tensor, vgg=False))
        tensor_expand = tensor_norm.unsqueeze(0)
        
        if stopup:
            self.stopup_G.train()
            recon_norm = self.stopup_G(tensor_expand)

        else:
            self.stopdown_G.train()
            recon_norm =  self.stopdown_G(tensor_expand)       

        recon = utils.denorm(recon_norm.cpu().data, vgg=False)
        recon = recon.squeeze(0)
        recon = torch.clamp(recon, min=0, max=1)
        recon_img = transforms.ToPILImage()(recon)
        return recon_img

    def save_model(self, epoch=None, is_pretrain=False):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if is_pretrain:
            torch.save(self.stopup_G.state_dict(), model_dir + '/' + self.model_name + '_stopup_G_param_pretrain.pkl')
            torch.save(self.stopdown_G.state_dict(), model_dir + '/' + self.model_name + '_stopdown_G_param_pretrain.pkl')
 
            print('Pre-trained generator model is saved.')
        else:
            if epoch is not None:
                torch.save(self.stopup_G.state_dict(), model_dir + '/' + self.model_name +
                           '_stopup_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, epoch, self.lr))
                torch.save(self.stopup_D.state_dict(), model_dir + '/' + self.model_name +
                           '_stopup_D_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, epoch, self.lr))

                torch.save(self.stopdown_G.state_dict(), model_dir + '/' + self.model_name +
                           '_stopdown_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, epoch, self.lr))
                torch.save(self.stopdown_D.state_dict(), model_dir + '/' + self.model_name +
                           '_stopdown_D_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, epoch, self.lr))

            else:
                torch.save(self.stopup_G.state_dict(), model_dir + '/' + self.model_name +
                           '_stopup_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, self.num_epochs, self.lr))
                torch.save(self.stopup_D.state_dict(), model_dir + '/' + self.model_name +
                           '_stopup_D_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, self.num_epochs, self.lr))
                torch.save(self.stopdown_G.state_dict(), model_dir + '/' + self.model_name +
                           '_stopdown_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, self.num_epochs, self.lr))
                torch.save(self.stopdown_D.state_dict(), model_dir + '/' + self.model_name +
                           '_stopdown_D_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, self.num_epochs, self.lr))
 
            print('Trained models are saved.')

    def load_model(self, is_pretrain=False):
        model_dir = os.path.join(self.save_dir, 'model')

        if is_pretrain:
            flag_stopup = False
            flag_stopdown = False

            model_name_stopup = model_dir + '/' + self.model_name + '_stopup_G_param_pretrain.pkl'
            model_name_stopup_D = model_dir + '/' + self.model_name + '_stopup_D_param_pretrain.pkl'
 
            if os.path.exists(model_name_stopup):
                self.stopup_G.load_state_dict(torch.load(model_name_stopup))
                self.stopup_D.load_state_dict(torch.load(model_name_stopup_D))
                flag_stopup = True

            model_name_stopdown = model_dir + '/' + self.model_name + '_stopdown_G_param_pretrain.pkl'
            model_name_stopdown_D = model_dir + '/' + self.model_name + '_stopdown_D_param_pretrain.pkl'
 
            if os.path.exists(model_name_stopdown):
                self.stopdown_G.load_state_dict(torch.load(model_name_stopdown))
                self.stopdown_D.load_state_dict(torch.load(model_name_stopdown_D))

                flag_stopdown = True

            print ("[loding] (up):", flag_stopup, ', (down):',flag_stopdown)
            print (model_name_stopup)
            print (model_name_stopdown)

            if flag_stopdown and flag_stopup:
                print('Pre-trained generator model is loaded.')
                return True
            else:
                return False

        else:
            flag_stopup = False
            flag_stopdown = False

            model_name_stopup = model_dir + '/' + self.model_name + \
                         '_stopup_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl' \
                         % (self.num_channels, self.batch_size, self.num_epochs, self.lr)
            print(model_name_stopup)

            if os.path.exists(model_name_stopup):
                self.stopup_G.load_state_dict(torch.load(model_name_stopup))
                flag_stopup = True

            model_name_stopdown = model_dir + '/' + self.model_name + \
                         '_stopdown_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl' \
                         % (self.num_channels, self.batch_size, self.num_epochs, self.lr)
 
            if os.path.exists(model_name_stopdown):
                self.stopdown_G.load_state_dict(torch.load(model_name_stopdown))
                flag_stopdown = True
                
            print ("[loding] (up):", flag_stopup, ', (down):',flag_stopdown)
            print (model_name_stopup)
            print (model_name_stopdown)

            if flag_stopup and flag_stopdown:
               print('Trained generator model is loaded.')
               return True

            else:
               return False

