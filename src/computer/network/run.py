"""
Training and data management scripts. Run this file and use the corresponding flags.
"""
import argparse
from pathlib import Path
from manage_data import ManageData
from train import Train
from show_data import ShowData

def main():
    """Main training function. Use flags to sort the dataset or show the dataset. 
    """    
    parser = argparse.ArgumentParser(description="Training and Dataset functions")
    parser.add_argument('--manage_dataset', action='store_true', help='if set, create dataset from raw data')
    parser.add_argument('--show_dataset', action='store_true', help='if set, call show dataset')
    parser.add_argument('--train', action='store_true', help='if set, train the network')

    parser.add_argument('--data_path', help='path to sorted data folder',
                        default='data/sorted_data')
    parser.add_argument('--raw_data_path', help='path to raw data folder',
                        default='data/raw_data')
    parser.add_argument('--checkpoint_path', help='path to store the checkpoints',
                        default='checkpoints')
    parser.add_argument('--batch_size', help='batch Size', type=int,
                        default=64)
    parser.add_argument('--epochs', help='epochs', type=int,
                        default=50)
    parser.add_argument('--learning_rate', help='Learning Rate', type=float,
                        default=1e-3)
    parser.add_argument('--train_from_checkpoint', help='Path to checkpoint',
                        default=None)
    parser.add_argument('--logdir', help='Path to Tensorboard Logs',
                        default="runs")

    args = parser.parse_args()
    #mode
    manage_dataset = args.manage_dataset
    show_dataset = args.show_dataset
    train = args.train

    #training
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    train_from_checkpoint = args.train_from_checkpoint
    epochs = args.epochs   

    project_path = Path.cwd().parent.parent.parent

    #paths
    raw_data_path = project_path / args.raw_data_path
    sorted_data_path = project_path / args.data_path
    checkpoint_path = project_path / args.checkpoint_path
    logdir_path = project_path / args.logdir

    if manage_dataset:
        # sort dataset
        print('manage dataset')
        data_manager = ManageData(raw_data_path, sorted_data_path)

        data_manager.sort_dataset()

    elif show_dataset:
        # show dataset
        print('show dataset')
        data_show = ShowData(raw_data_path)

        data_show.show()

    elif train:
        # training 
        print('start training')
        train = Train(sorted_data_path, checkpoint_path, logdir_path)   

        train.train_network(batch_size, learning_rate, epochs, train_from_checkpoint)

    else:
        raise Exception("No flag is set. Set either 'manage_dataset', 'show_dataset' or 'train' flag.")

if __name__ == '__main__':
    main()