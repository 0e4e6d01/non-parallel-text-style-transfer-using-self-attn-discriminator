import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def get_optim(args, params):
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr,
                        weight_decay=args.l2_reg)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr,
                        weight_decay=args.l2_reg)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr,
                        weight_decay=args.l2_reg)
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr,
                        eps=args.adam_eps, weight_decay=args.l2_reg)
    else:
        raise ValueError("Invalid optim method: " + args.optim)
    print("use %s optimizer" % args.optim)
    return optimizer

def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    print("use constant schedule")
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """
    print("use constant schedule with warmup")

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    print("use linear schedule with warmup")

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_constant_schedule_with_linear_decay(optimizer, decay_step, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    constant during a constant period.
    """
    print("use constant schedule with linear decay")

    def lr_lambda(current_step):
        if current_step < decay_step:
            return 1.0
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - decay_step))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
