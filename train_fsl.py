from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)

# from ipdb import launch_ipdb_on_exception

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # args.init_weights = './checkpoints/feat-5-shot-miniImage.pth'
    # args.init_weights = './checkpoints/feat-1-shot-miniImage.pth'
    args.init_weights = './checkpoints/miniImage-Res12-pre.pth'
    # with launch_ipdb_on_exception():

    # 对于5-way,1-shot; miniImageNet train
    args.shot = 5
    args.eval_shot = 5
    args.balance = 0.1
    args.temperature = 64
    args.temperature2 = 32
    args.lr = 0.0002
    args.step_size = 40
    args.gamma = 0.5
    args.use_euclidean = True

    pprint(vars(args))

    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)
