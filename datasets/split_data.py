import argparse
import glob
import os
import sys
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--path_A', default='./sawyer_gen/random_normal', help='Path to the original data')
parser.add_argument('--path_B', default='./sawyer_gen/textured', help='Path to the modified data')
parser.add_argument('--path_trained', default='./sawyer_gen/state_rollout_normal', help='path to images from the trained policy')
parser.add_argument('--output_dir', default='sawyer_black256', help='Path the output directory for the data')
parser.add_argument('--splits_A', nargs='+',   default=[90, 0, 10], help='The splits for the data defined as X Y Z    ' \
                    'where X is the percent train, Y is the percent valid and Z is the percent test')
parser.add_argument('--splits_B', nargs='+',   default=[10, 0, 90], help='The splits for the data defined as X Y Z    ' \
                    'where X is the percent train, Y is the percent valid and Z is the percent test')

if __name__ == '__main__':
    args = parser.parse_args()

    assert len(args.splits_A) == 3, 'Must have 3 splits'
    args.splits = [float(args.splits_A[i]) for i in range(3)]
    assert sum(args.splits_A) == 100, 'Split must add up to 100%'

    assert len(args.splits_B) == 3, 'Must have 3 splits'
    args.splits = [float(args.splits_B[i]) for i in range(3)]
    assert sum(args.splits_B) == 100, 'Split must add up to 100%'


    files_A = glob.glob('{}/*.jpg'.format(args.path_A))
    files_B = glob.glob('{}/*.jpg'.format(args.path_B))
    files_trained = glob.glob('{}/*.jpg'.format(args.path_trained))

    shuffle(files_A)
    shuffle(files_B)
    shuffle(files_trained)

    if not os.path.isdir(args.output_dir):
        os.mkdir('./{}'.format(args.output_dir))

    for subdir in ['trainA', 'trainB', 'valA', 'valB', 'testA', 'testB']:
        if not os.path.isdir('./{}/{}'.format(args.output_dir, subdir)):
            os.mkdir('./{}/{}'.format(args.output_dir, subdir))


    len_train = (int(len(files_A)*args.splits_A[0]/100.),
                 int(len(files_B)*args.splits_B[0]/100.),
                 int(len(files_trained)*args.splits_A[0]/100.))
    len_valid = (int(len(files_A)*args.splits_A[1]/100.),
                 int(len(files_B)*args.splits_B[1]/100.),
                 int(len(files_trained)*args.splits_A[1]/100.))
    len_test = (int(len(files_A)*args.splits_A[2]/100.),
                int(len(files_B)*args.splits_B[2]/100.),
                int(len(files_trained)*args.splits_A[2]/100.))

    train_A, train_B, train_trained = files_A[:len_train[0]], files_B[:len_train[1]], files_trained[:len_train[2]]
    files_A, files_B, files_trained = files_A[len_train[0]:], files_B[len_train[1]:], files_trained[len_train[2]:]

    valid_A, valid_B, valid_trained = files_A[:len_valid[0]], files_B[:len_valid[1]], files_trained[:len_valid[2]]
    files_A, files_B, files_trained = files_A[len_valid[0]:], files_B[len_valid[1]:], files_trained[len_valid[2]:]

    test_A, test_B, test_trained = files_A, files_B, files_trained


    for f in train_A:
        os.system('cp {} ./{}/trainA'.format(f, args.output_dir))
    for f in train_B:
        os.system('cp {} ./{}/trainB'.format(f, args.output_dir))
    for f in valid_A:
        os.system('cp {} ./{}/valA'.format(f, args.output_dir))
    for f in valid_B:
        os.system('cp {} ./{}/valB'.format(f, args.output_dir))
    for f in test_A:
        os.system('cp {} ./{}/testA'.format(f, args.output_dir))
    for f in test_B:
        os.system('cp {} ./{}/testB'.format(f, args.output_dir))

    for f in train_trained:
        f_name = f.split('/')[-1]
        os.system('cp {} ./{}/trainA/trained_{}'.format(f, args.output_dir, f_name))
    for f in valid_trained:
        f_name = f.split('/')[-1]
        os.system('cp {} ./{}/valA/trained_{}'.format(f, args.output_dir, f_name))
    for f in test_trained:
        f_name = f.split('/')[-1]
        os.system('cp {} ./{}/testA/trained_{}'.format(f, args.output_dir, f_name))
