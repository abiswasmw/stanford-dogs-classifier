import argparse
from classify import run_classifier


def main(url):
    run_classifier(url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dog breed classifier.')
    parser.add_argument('--url', metavar='image_url', type=str, nargs='+', default=None, help='URL of the image.')
    args = parser.parse_args()
    main(args.url)
