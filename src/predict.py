import argparse
import torch
import utils


def predict(args):
    model = torch.load(args.model)
    processed_input = utils.preprocess_input(args)
    prediction = model.forward(processed_input)
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Testing a pretrained Character Based CNN for text classification')
    parser.add_argument('--text', type=str,
                        default='I love pizza!', help='text string')
    parser.add_argument('doc_type', type=str,
                        choices=['tweets', 'reviews'], default='tweets')

    parser.add_argument('--model', type=str, help='path for pre-trained model')
    parser.add_argument('--input_file', type=str,
                        help='path of a csv input file containing data to score')
    parser.add_argument('--output_file', type=str,
                        help='path of a csv output file containing the predicitions')

    # arguments needed for the predicition
    parser.add_argument('--alphabet', type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    parser.add_argument('--number_of_characters', type=int, default=68)
    parser.add_argument('--max_length', type=int, default=150)

    args = parser.parse_args()
    prediction = predict(args)
    print('input : {}'.format(args.text))
    print('prediction : {}'.format(prediction))
