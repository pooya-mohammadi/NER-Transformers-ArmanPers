from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--in_files", nargs='+', help="path to input txt files")
parser.add_argument("--out_path", default="train.txt", help="output_path, default is train.txt")
args = parser.parse_args()

if __name__ == '__main__':
    input_files = args.in_files
    txt = []
    for file_path in input_files:
        txt.extend([l.strip() for l in open(file_path, mode='r').readlines()])
    with open(args.out_path, mode='w') as f:
        for l in txt:
            f.write(l + "\n")
