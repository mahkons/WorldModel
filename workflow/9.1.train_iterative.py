import subprocess
import argparse
from tqdm import tqdm

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--memtype', type=str, default='Classic', required=False)
    parser.add_argument('--plot-path', type=str, default='generated/train_plot.torch', required=False)
    parser.add_argument('--restart', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)

    parser.add_argument('--iters', type=int, default=10, required=False)
    parser.add_argument('--iters_sample', type=int, default=100000, required=False)
    parser.add_argument('--epochs_model', type=int, default=10, required=False)
    parser.add_argument('--epochs_agent', type=int, default=100, required=False)
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()

    subprocess.call(["python3", "1.sample_rollouts.py",
        "--iters=" + str(args.iters_sample),
        "--device=" + args.device,
        "--with-agent=" + str(not args.restart)
        ])
    subprocess.call(["python3", "5.train_model.py", 
        "--epochs=" + str(args.epochs_model),
        "--device=" + args.device,
        "--restart=" + str(args.restart)
        ])
    subprocess.call(["python3", "9.train_agent.py", 
        "--epochs=" + str(args.epochs_agent),
        "--device=" + args.device,
        "--memtype=" + args.memtype,
        "--plot-path=" + args.plot_path,
        "--restart=" + str(args.restart)
        ])

    for _ in tqdm(range(1, args.iters)):
        subprocess.call(["python3", "1.sample_rollouts.py",
            "--iters=" + str(args.iters_sample),
            "--device=" + args.device,
            "--with-agent=True"
            ])
        subprocess.call(["python3", "5.train_model.py", 
            "--epochs=" + str(args.epochs_model),
            "--device=" + args.device
            ])
        subprocess.call(["python3", "9.train_agent.py", 
            "--epochs=" + str(args.epochs_agent),
            "--device=" + args.device,
            "--memtype=" + args.memtype,
            "--plot-path=" + args.plot_path
            ])
