import argparse
import numpy as np
import torch
from pprint import pprint

# from utils.misc import mkdirs


# always uses cuda if avaliable

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='FC-DeepONet')

        # model
        self.add_argument('--batch_size', type=int, default=4, help='number of batch size')
        self.add_argument('--epochs', type=int, default=501, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.add_argument('--seed', type=int, default=123, help='random seed')

    def parse(self):
        args = self.parse_args()

        # seed
        # if args.seed is None:
        #     args.seed = np.random.randint(1, 10000)
        # print("Random Seed: ", args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))

        return args


# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
