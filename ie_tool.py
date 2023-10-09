import argparse

from args import train_argparser, eval_argparser, predict_argparser
from config_reader import process_configs
from ie_tool import input_reader
from ie_tool.ie_tool_trainer import IEToolTrainer

def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


def __eval(run_args):
    trainer = IEToolTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _predict():
    arg_parser = predict_argparser()
    process_configs(target=__predict, arg_parser=arg_parser)


def __predict(run_args):
    trainer = IEToolTrainer(run_args)
    trainer.predict(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                    input_reader_cls=input_reader.JsonPredictionInputReader)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'predict' or 'eval'", default="predict")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'eval':
        _eval()
    elif args.mode == 'predict':
        _predict()
    else:
        raise Exception("Mode should be eval or predict")
