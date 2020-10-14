from torch import nn
'''    
group.add_argument('--disc-optimizer', default='nag', metavar='OPT',
                       choices=OPTIMIZER_REGISTRY.keys(),
                       help='Optimizer')
    group.add_argument('--disc-lr', '--learning-rate3', default='0.25', metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    group.add_argument('--disc-momentum', default=0.99, type=float, metavar='M',
                       help='momentum factor')
    group.add_argument('--disc-weight-decay', '--wd3', default=0.0, type=float, metavar='WD',
                       help='weight decay')

    # Learning rate schedulers can be found under fairseq/optim/lr_scheduler/
    group.add_argument('--disc-lr-scheduler', default='reduce_lr_on_plateau',
                       choices=LR_SCHEDULER_REGISTRY.keys(),
                       help='Learning Rate Scheduler')
    group.add_argument('--disc-lr-shrink', default=0.1, type=float, metavar='LS',
                       help='learning rate shrink factor for annealing, lr_new = (lr * lr_shrink)')
    group.add_argument('--disc-min-lr', default=1e-5, type=float, metavar='LR',
                       help='minimum learning rate')
    group.add_argument('--disc-min-loss-scale', default=1e-4, type=float, metavar='D',
                       help='minimum loss scale (for FP16 training)')
'''

class Discriminator(nn.Module):

    DIS_ATTR = ['input_dim', 'dis_layers', 'dis_hidden_dim', 'dis_dropout']

    def __init__(self, args):
        """
        Discriminator initialization.
        """
        super(Discriminator, self).__init__()

        self.n_langs = 2
        self.input_dim = args.encoder_embed_dim
        self.dis_layers = 3
        self.dis_hidden_dim = 128
        self.dis_dropout = 0.8

        layers = []
        for i in range(self.dis_layers + 1):
            if i == 0:
                input_dim = self.input_dim
#                input_dim *= (2 if params.attention and not params.dis_input_proj else 1)
            else:
                input_dim = self.dis_hidden_dim
            output_dim = self.dis_hidden_dim if i < self.dis_layers else self.n_langs
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)