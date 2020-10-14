import torch

from torch.serialization import default_restore_location
from fairseq.utils import _upgrade_state_dict, convert_state_dict_type, torch_persistent_save
from collections import OrderedDict

import os
import argparse


def load_model_state(filename):
    if not os.path.exists(filename):
        return None
    state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    state = _upgrade_state_dict(state)

    return state

def _strip_params(state, strip_what='decoder'):
    new_state = state
    new_state['model'] = OrderedDict({key: value for key, value in state['model'].items()
                             if not key.startswith(strip_what)})

    return new_state

def save_state(encoder_state, decoder_state, filename):
    new_state = encoder_state
    for key, value in decoder_state['model'].items():
        new_state['model'][key] = value
    torch_persistent_save(new_state, filename)


def main(args):
    encoder_model_state = load_model_state(args.encoder_model)
    print("Loaded model to strip its Encoder {}".format(args.encoder_model))
    decoder_model_state = load_model_state(args.decoder_model)
    print("Loaded model to strip its Decoder {}".format(args.decoder_model))
    
    encoder_model_state = _strip_params(encoder_model_state, strip_what='decoder')
    print("Stripped encoder")
    decoder_model_state = _strip_params(decoder_model_state, strip_what='encoder')
    print("Stripped decoder")

    save_state(encoder_model_state, decoder_model_state, args.new_model_path)
    print("Saved to {}".format(args.new_model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder-model', help="The path to the model to strip its encoder")
    parser.add_argument('--decoder-model', help="The path to the model to strip its decoder")
    parser.add_argument('--new-model-path', help="The name for the stripped model")

    args = parser.parse_args()
    main(args)
