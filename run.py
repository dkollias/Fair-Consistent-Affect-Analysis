# Copyright (c) 2024, Guanyu Hu
# All rights reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

from config.base_parameters import get_args_parser, base_train_parser, base_parser
from config.initializer import TrainInitializer
from model_utils.main import main

if __name__ == '__main__':
    """ Config """
    parser = [base_parser(), base_train_parser()]
    init = TrainInitializer(get_args_parser(parser))

    """ Main """
    main(init())
