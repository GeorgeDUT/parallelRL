from discrete_a3c import *


def cur_args():
    args = gen_args()
    args.grad_attack_type = 'uniform'
    args.grad_attack_params = [-2, 0]
    args.cur_test_type = 'all'
    assert args.cur_test_type in test_type, args.cur_test_type + ' not in ' + str(test_type)
    args.bad_worker_id = random.sample(range(1, 10), 3) if args.cur_test_type != 'all_good' else []
    args.base_path = './' + args.grad_attack_type + '{}_'.format(args.grad_attack_params) + args.cur_test_type
    args.save_path = make_training_save_path(args.base_path)
    if args.cur_test_type == 'all':
        args.Good_Actor_num = 10
    return args


if __name__ == '__main__':
    test_runner(cur_args)
