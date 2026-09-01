import argparse
from Agent import Agent_HER,Buffer,Agent_Curriculum
from DQN import DQN,DQN_Attention
from Env import Bits_Env
import torch

def get_args():
    parser = argparse.ArgumentParser(description='DQN for Bits Sequence Generation')
    parser.add_argument('--n', type=int, default=20, help="Sequence Length")
    parser.add_argument('--net', type=list, default=[256, 256, 256], help="Network Hidden Layers' Dimension")
    parser.add_argument('--buffer', type=int, default=1e4, help="Buffer Size")
    parser.add_argument('--batch', type=int, default=64, help="Batch Size")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount Factor")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
    parser.add_argument('--eps_start', type=float, default=0)
    parser.add_argument('--eps_decay', type=float, default=0.99)
    parser.add_argument('--eps_final', type=float, default=0)
    parser.add_argument('--episode', type=int, default=5e6)
    parser.add_argument('--update_episode', type=int, default=50, help="Amount of Episodes to Update Network")
    parser.add_argument('--HER', type=str, default="False", help="Whether to Use HER")
    parser.add_argument('--HER_param', type=int, default=4)
    parser.add_argument('--HER_my_version', type=str, default="False", help="Whether to Use HER (my version)")
    parser.add_argument('--group', type=int, default=1, help="Amount of Tokens to Generate at Once")
    parser.add_argument('--reward_version', type=int, default=0, help="Type of Reward Function (Refer to README for details.)")
    parser.add_argument('--Curriculum', type=str, default="False", help="Whether to Use Curriculum Learning")
    parser.add_argument('--interval', type=int, default=1, help="Curriculum Interval")
    parser.add_argument('--unfreeze_interval', type=int, default=1, help="Number of Curriculums to Unfreeze Parameters")
    parser.add_argument('--freeze', type=float, default=0.5, help="Initial Freezing Proportion of Curriculum Learning")
    parser.add_argument('--drop', type=float, default=0.3, help="Dropout Rate")
    parser.add_argument('--attention', type=str, default="False", help="Whether to Add an Attention Layer to Network")
    parser.add_argument('--emb_dims', type=int, default=4, help="Embedding Dimensions (before the Attention Layer)")
    parser.add_argument('--head_num', type=int, default=1, help="Attention Head Number")
    parser.add_argument('--cpu', type=str, default="False", help="Use CPU")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA Device ID")
    parser.add_argument('--model_path', type=str, default=None, help="Path to Load Model")
    parser.add_argument('--save_path', type=str, default="./Model.pt", help="Path to Save Model")
    parser.add_argument('--log_path', type=str, default="./log", help="Path to Save Log")
    parser.add_argument('--train', type=str, default="True")
    parser.add_argument('--auto_test', type=str, default="True", help="You can choose to test by hand or automatically.")
    parser.add_argument('--test_nums', type=int, default=100, help="How many times you want to test the model.")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args.train=(args.train.lower()=="true")
    args.auto_test=(args.auto_test.lower()=="true")
    args.HER=(args.HER.lower()=="true")
    args.HER_my_version=(args.HER_my_version.lower()=="true")
    args.Curriculum=(args.Curriculum.lower()=="true")
    args.attention=(args.attention.lower()=="true")
    args.cpu=(args.cpu.lower()=="true")
    if args.cpu:
        device=torch.device("cpu")
    else:
        device = torch.device("cuda:"+str(args.cuda_device) if torch.cuda.is_available() else "cpu")
    env=Bits_Env(n=args.n,reward_type=args.reward_version,group=args.group)
    if args.attention:
        policy_network = DQN_Attention(state_dim=2 * args.n, action_dim=2 ** args.group, hidden_dims=args.net, device=device, num_heads=args.head_num, emb_dims=args.emb_dims, dropout_rate=args.drop)
        target_network = DQN_Attention(state_dim=2 * args.n, action_dim=2 ** args.group, hidden_dims=args.net, device=device, num_heads=args.head_num, emb_dims=args.emb_dims, dropout_rate=args.drop)
    else:
        policy_network = DQN(state_dim=2 * args.n, action_dim=2 ** args.group, hidden_dims=args.net, dropout_rate=args.drop)
        target_network = DQN(state_dim=2 * args.n, action_dim=2 ** args.group, hidden_dims=args.net, dropout_rate=args.drop)
    if args.model_path is not None:
        policy_network.load_state_dict(torch.load(args.model_path,map_location=device))
        target_network.load_state_dict(torch.load(args.model_path,map_location=device))
    replay_buffer=Buffer(args.buffer)
    if args.Curriculum:
        agent =Agent_Curriculum(env=env, policy_network=policy_network, target_network=target_network, replay_buffer=replay_buffer,
                          gamma=args.gamma, lr=args.lr, batch_size=args.batch, eps_start=args.eps_start, eps_final=args.eps_final, eps_decay=args.eps_decay, device=device, interval=args.interval ,initial_freeze=args.freeze ,freeze_interval=args.unfreeze_interval )
    elif args.HER:
        agent = Agent_HER(env=env, policy_network=policy_network, target_network=target_network, replay_buffer=replay_buffer,
                          gamma=args.gamma, lr=args.lr, batch_size=args.batch, eps_start=args.eps_start, eps_final=args.eps_final, eps_decay=args.eps_decay, device=device, HER=args.HER_param, my_version=args.HER_my_version)
    else:
        agent = Agent_HER(env=env, policy_network=policy_network, target_network=target_network, replay_buffer=replay_buffer,
                          gamma=args.gamma, lr=args.lr, batch_size=args.batch, eps_start=args.eps_start, eps_final=args.eps_final, eps_decay=args.eps_decay, device=device, HER=None)
    if args.train:
        agent.train(episode=args.episode, update_episode=args.update_episode,save_path=args.save_path, save_log=args.log_path)
    else:
        if args.auto_test:
            agent.test(args.test_nums)
        else:
            target_str = input("Please input the target sequemce (invalid token will be ignored): ")
            target = []
            for i in range(len(target_str)):
                token = target_str.__getitem__(i)
                if token == '0' or token == '1':
                    target.append(eval(token))
            if len(target) != args.n:
                print("Your inpupt is", end=" ")
                print(target, end=" ")
                print("with the length of {}".format(len(target)))
                print("You need to input a target sequence with length of {}".format(args.n))
                exit(-1)
            agent.show_process(target)

if __name__ == "__main__":
    main()