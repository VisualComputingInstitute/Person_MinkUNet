import argparse
import os
import shutil


def _parse_args():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--dir", type=str, required=True, help="directory to be cleaned"
    )
    parser.add_argument(
        "--date", type=int, required=True, help="logs before this date will be removed"
    )
    args = parser.parse_args()

    assert args.date >= 20200101 and args.date <= 20201231

    return args


def _user_confirms(prompt):
    yes = {"yes", "y"}
    no = {"no", "n"}

    choice = input(prompt).lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        print(f"Invalid input: {choice}")
        return False


def _log_dates_earlier_than(log_dir, reference_date):
    log_date = int(os.path.basename(log_dir).split("_")[0])
    return log_date < reference_date


def clean_log_dir(log_dir, latest_date):
    logs_list = sorted(
        [
            x
            for x in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, x))
            and _log_dates_earlier_than(x, latest_date)
        ]
    )

    print("Following logs will be removed:")
    for x in logs_list:
        print(os.path.join(log_dir, x))

    if not _user_confirms("Continue [y/n]?"):
        return

    for x in logs_list:
        x = os.path.join(log_dir, x)
        print(f"Remove {x}")
        shutil.rmtree(x)


if __name__ == "__main__":
    args = _parse_args()

    clean_log_dir(args.dir, args.date)
