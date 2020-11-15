# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --exp_id en-fr \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#     --beam 10 --length_penalty 1.1
#

import os
import io
import sys
import argparse
import torch

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.model.transformer import N_MAX_POSITIONS
from src.data.loader import fix_dico_parameters

from collections import OrderedDict

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")


    parser.add_argument("--input_path", type=str, default="", help="Input path")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    
    parser.add_argument("--beam", type=int, default=1, help="Beam size")
    parser.add_argument("--length_penalty", type=float, default=1, help="length penalty")

    parser.add_argument("--bt_output_path", type=str, default="", help="Output path")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    parser.add_argument("--hack_for_mass", type=bool_flag, default="False", help="Using hack mode to load mass")
    parser.add_argument("--hack_for_wrong_attention_setting", type=bool_flag, default="False", help="Using hack mode to load wrong attention setting model")
    return parser





def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # build dictionary
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

    # HACK, fix the wrong attention setting
    if params.hack_for_wrong_attention_setting:
        model_params['attention_setting'] = 'default'

    # HACK, fix the MASS loading
    if params.hack_for_mass:
        fix_dico_parameters(model_params, dico)
        model_params['use_lang_emb'] = True
        model_params['attention_setting'] = 'v1'
    
    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # HACK, for compatible with encoder
    encoder_with_output = False
    if 'pred_layer.proj.weight' in reloaded['encoder'].keys() or 'module.pred_layer.proj.weight' in reloaded['encoder'].keys():
        encoder_with_output = True

    # build dictionary / build encoder / build decoder / reload weights

    # fix checkpoint loading
    encoder_state_dict = OrderedDict()
    for k, v in reloaded['encoder'].items():
        if k.startswith('module.'):
            name = k[7:]
            encoder_state_dict[name] = v
        else:
            name = k
            encoder_state_dict[name] = v

    decoder_state_dict = OrderedDict()
    for k, v in reloaded['decoder'].items():
        if k.startswith('module.'):
            name = k[7:]
            decoder_state_dict[name] = v
        else:
            name = k
            decoder_state_dict[name] = v
    
    
    
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=encoder_with_output).cuda().eval()
    decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]

    fin = sys.stdin
    if params.input_path is not None and len(params.input_path) > 0:
        fin = io.open(params.input_path, 'r', encoding='utf-8')

    # read sentences from stdin
    src_sent = []
    for line in fin.readlines():
        assert len(line.strip().split()) > 0
        src_sent.append(line)
    logger.info("Read %i sentences from stdin. Translating ..." % len(src_sent))
    
    src_start = 0
    if os.path.exists(params.output_path):
        for index, line in enumerate(open(params.output_path,'r')):
            src_start += 1
        logger.info('Translate from %d' % src_start)
        src_sent = src_sent[src_start:]
        f = io.open(params.output_path, 'a', encoding='utf-8')
        if params.bt_output_path is not None and len(params.bt_output_path) > 0:
            f_bt = io.open(params.bt_output_path, 'a', encoding='utf-8')
    else:
        f = io.open(params.output_path, 'w', encoding='utf-8')
        if params.bt_output_path is not None and len(params.bt_output_path) > 0:
            f_bt = io.open(params.bt_output_path, 'w', encoding='utf-8')

    for i in range(0, len(src_sent), params.batch_size):

        # prepare batch
        word_ids = []
        for s in src_sent[i:i + params.batch_size]:
            words = s.strip().split()
            if (len(words) + 2) > N_MAX_POSITIONS:
                words = words[:N_MAX_POSITIONS-2]
            word_ids.append(torch.LongTensor([dico.index(w) for w in words]))

        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
        batch[0] = params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = params.eos_index
        langs = batch.clone().fill_(params.src_id)

        # encode source batch and translate it
        encoded = encoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
        encoded = encoded.transpose(0, 1)

        generate_max_len = int(1.5 * lengths.max().item() + 10)
        if generate_max_len >= N_MAX_POSITIONS:
            generate_max_len = N_MAX_POSITIONS - 1

        if params.beam == 1:
            decoded, dec_lengths = decoder.generate(encoded, lengths.cuda(), params.tgt_id, max_len=generate_max_len)
        else:
            decoded, dec_lengths = decoder.generate_beam(
                encoded, lengths.cuda(), params.tgt_id, beam_size=params.beam,
                length_penalty=params.length_penalty,
                early_stopping=False,
                max_len=generate_max_len)
        
        output_sents = []

        # convert sentences to words
        for j in range(decoded.size(1)):

            # remove delimiters
            sent = decoded[:, j]
            delimiters = (sent == params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

            # output translation
            source = src_sent[i + j].strip()
            target = " ".join([dico[sent[k].item()] for k in range(len(sent))])
            output_sents.append(target)
            sys.stderr.write("%i / %i: %s -> %s\n" % (i + j, len(src_sent), source, target))
            f.write(target + "\n")
            f.flush()

        # back-translation
        if params.bt_output_path is not None and len(params.bt_output_path) > 0:
            bt_word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                    for s in output_sents]
            bt_lengths = torch.LongTensor([len(s) + 2 for s in bt_word_ids])
            bt_batch = torch.LongTensor(bt_lengths.max().item(), bt_lengths.size(0)).fill_(params.pad_index)
            bt_batch[0] = params.eos_index
            for j, s in enumerate(bt_word_ids):
                if bt_lengths[j] > 2:  # if sentence not empty
                    bt_batch[1:bt_lengths[j] - 1, j].copy_(s)
                bt_batch[bt_lengths[j] - 1, j] = params.eos_index
            bt_langs = bt_batch.clone().fill_(params.tgt_id)

            # encode source batch and translate it
            bt_encoded = encoder('fwd', x=bt_batch.cuda(), lengths=bt_lengths.cuda(), langs=bt_langs.cuda(), causal=False)
            bt_encoded = bt_encoded.transpose(0, 1)
            bt_decoded, bt_dec_lengths = decoder.generate(bt_encoded, bt_lengths.cuda(), params.src_id, max_len=int(1.5 * bt_lengths.max().item() + 10))

            # convert sentences to words
            for j in range(bt_decoded.size(1)):

                # remove delimiters
                bt_sent = bt_decoded[:, j]
                bt_delimiters = (bt_sent == params.eos_index).nonzero().view(-1)
                assert len(bt_delimiters) >= 1 and bt_delimiters[0].item() == 0
                bt_sent = bt_sent[1:] if len(bt_delimiters) == 1 else bt_sent[1:bt_delimiters[1]]

                # output translation
                bt_target = " ".join([dico[bt_sent[k].item()] for k in range(len(bt_sent))])
                # sys.stderr.write("%i / %i: %s -> %s\n" % (i + j, len(src_sent), source, target))
                f_bt.write(bt_target + "\n")


    f.close()
    if params.bt_output_path is not None and len(params.bt_output_path) > 0:
        f_bt.close()


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang
    # assert params.output_path and not os.path.isfile(params.output_path)

    # translate
    with torch.no_grad():
        main(params)
