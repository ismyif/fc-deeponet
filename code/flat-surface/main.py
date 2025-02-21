import numpy as np
import torch
import time
# local import
from model import FCDeepONet
from args import args, device
from pytorchtools import EarlyStopping

epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
print('torch.cuda.is_available:', torch.cuda.is_available())
print('torch.cuda.device_count:', torch.cuda.device_count())
# print('torch.cuda.current_device:', torch.cuda.current_device())

input_train = np.load('/dev/shm/input_train.npz')
target_train = np.load('/dev/shm/target_train.npz')

input_valid = np.load('/dev/shm/input_valid.npz')
target_valid = np.load('/dev/shm/target_valid.npz')

input_tra = torch.FloatTensor(input_train['input_data'])
Xs_tra = torch.FloatTensor(input_train['XXs'])
target_tra = torch.FloatTensor(target_train['tau_data'])

input_val = torch.FloatTensor(input_valid['input_data'])
Xs_val = torch.FloatTensor(input_valid['XXs'])
target_val = torch.FloatTensor(target_valid['tau_data'])

data_train = torch.utils.data.TensorDataset(input_tra, Xs_tra, target_tra)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=4, pin_memory=True)
# train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)

data_valid = torch.utils.data.TensorDataset(input_val, Xs_val, target_val)
valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, num_workers=4, pin_memory=True)
# valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size)


net = FCDeepONet().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-08)
pre_path= ''
save = "_style-a.pt"
loss_curve="forward_loss.txt"
early_stopping = EarlyStopping(patience=50, verbose=True, path=pre_path+'checkpoint_early')
# print(param)

epoch_losses_train = []
epoch_losses_valid = []
# epoch_losses = []
# data_losses = []
# physics_losses = []
start_time = time.perf_counter()
for epoch in range(epochs):
    epoch_loss_train, loss_train = [], []
    epoch_loss_valid, data_loss_valid = [], []
    net.train()
    for batch_idx_train, (input_tra, Xs_tra, target_tra) in enumerate(train_loader):
        # print("idx_train",idx_train)
        input_tra = input_tra.to(device)
        Xs_tra = Xs_tra.to(device)
        target_tra = target_tra.to(device)

        optimizer.zero_grad()

        # velmodel = velmodel.reshape(velmodel.shape[0], -1).to(device)
        taupred_train = net.forward(input_tra, Xs_tra)
        loss_fn = torch.nn.MSELoss().to(device)
        loss_train = loss_fn(taupred_train, target_tra)
        epoch_loss_train.append(loss_train.item())

        loss_train.backward()
        optimizer.step()

        if batch_idx_train % 100 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx_train, len(train_loader), 100. * batch_idx_train / len(train_loader), loss_train.item()))

    with torch.no_grad():
        net.eval()
        for batch_idx_valid, (input_val, Xs_val, target_val) in enumerate(valid_loader):
            # print("idx_valid",idx_valid)
            input_val = input_val.to(device)
            Xs_val = Xs_val.to(device)
            target_val = target_val.to(device)

            taupred_valid = net.forward(input_val, Xs_val)
            loss_fn = torch.nn.MSELoss().to(device)
            loss_valid = loss_fn(taupred_valid, target_val)
            epoch_loss_valid.append(loss_valid.item())

            # loss = loss_d
            # print(loss)

            if batch_idx_valid % 50 == 0:
                print(
                    'Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                        epoch, batch_idx_valid, len(valid_loader), 100. * batch_idx_valid / len(valid_loader), loss_valid.item(),))

    avg_epoch_loss_train = np.average(epoch_loss_train)
    epoch_losses_train.append(avg_epoch_loss_train)

    avg_epoch_loss_valid = np.average(epoch_loss_valid)
    epoch_losses_valid.append(avg_epoch_loss_valid)

    if epoch % 100 == 0:
        torch.save(net.state_dict(), pre_path+"epoch" + str(epoch) + save)
        with open(pre_path+loss_curve, 'w') as epoch_los:
            epoch_los.write(str(epoch_losses_train) + '\n')
            epoch_los.write(str(epoch_losses_valid) + '\n')
            # valid_los.write(str(physics_losses_valid) + '\n')
    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'], '\ttrain_epoch_loss:', avg_epoch_loss_train)
    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'], '\tvalid_epoch_loss:', avg_epoch_loss_valid)

    # scheduler.step(loss_train)
    scheduler.step(avg_epoch_loss_train)
    early_stopping(avg_epoch_loss_valid, net)

    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     torch.save(net.state_dict(), pre_path+"epoch" + str(epoch) + save)
    #     with open(pre_path+"forward_loss_model1_early.txt", 'w') as epoch_los:
    #         epoch_los.write(str(epoch_losses_train) + '\n')
    #         epoch_los.write(str(epoch_losses_valid) + '\n')
    #     break

end_time = time.perf_counter()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"Network training took {hours}:{minutes}:{seconds}")
