from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument("--in_file", required=True, help="input file path to be split")
parser.add_argument("--split_size", default=0.2, help="split size, default is 0.2")
parser.add_argument("--part_1", default="train.txt", help="path to part one")
parser.add_argument("--part_2", default="val.txt", help="path to part two")


def get_samples(lines: list[str]) -> list:
    """
    input is a list of name and tag
    :param lines:
    :return:
    """
    samples = []
    sample = []
    for line in lines:
        line = line.strip()
        if line:
            sample.append(line)
        else:
            samples.append(sample)
            sample = []

    return samples


def write_samples(sample_list: list[list[str]], output_path: str):
    with open(output_path, mode="w") as f:
        for sample_lst in sample_list:
            for sample in sample_lst:
                f.write(sample + "\n")
            f.write("\n")
    print(f"[INFO] Successfully write {len(sample_list)} samples to {output_path}")


if __name__ == '__main__':
    args = parser.parse_args()
    main_samples = get_samples(open(args.in_file, mode="r").readlines())
    train, val = train_test_split(main_samples, test_size=args.split_size)
    write_samples(train, args.part_1)
    write_samples(val, args.part_2)
