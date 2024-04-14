from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    BertModel,
)
import copy
import postln_bert
import inflate_postlnbert_lemon_utils
import os

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    modconfig = BertConfig.from_pretrained(os.path.join(args.src_path, 'config.json'))

    print(modconfig)
    inf_modconfig = BertConfig.from_pretrained(args.dest_config)

    print('Load checkpoint from :{}'.format(args.src_path))
    modmodel = postln_bert.BertForMaskedLMv2.from_pretrained(
        os.path.join(args.src_path, 'pytorch_model.bin'),
        config=modconfig,
    )
    print('Number of layers for source model: {}'.format(len(modmodel.bert.encoder.layer)))
    inf_modmodel = postln_bert.BertForMaskedLMv2(inf_modconfig)
    print('Number of layers for expanded model: {}'.format(len(inf_modmodel.bert.encoder.layer)))

    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input_wo = copy.deepcopy(encoded_input)
    encoded_input_wo.pop('attention_mask')
    modmodel.eval()
    inf_modmodel.eval()
    inflate_postlnbert_lemon_utils.inflate_LEMON(modmodel, inf_modmodel, mode='proj',)

    out = modmodel(**encoded_input)
    inf_out = inf_modmodel(**encoded_input)
    print('Maximum difference is: {}'.format((inf_out['logits']-out['logits']).abs().max().item()))

    epoch_stamp = ''
    process_src_path = args.src_path
    while not epoch_stamp:
        process_src_path, epoch_stamp = os.path.split(process_src_path)
    
    if 'epoch' not in epoch_stamp:
        raise ValueError('epoch not found in src_path')

    if 'AKI' in args.method:
        dst_path = os.path.join(args.dst_path, args.method+'_'+args.depth_pattern + '_' + args.AKI_pattern + '_' + epoch_stamp)
    else:
        dst_path = os.path.join(args.dst_path, args.method+'_'+args.depth_pattern + '_' + epoch_stamp)
    
    # print('Saving checkpoint to:{}'.format(dst_path))
    # inf_modmodel.save_pretrained(dst_path)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Huggingface model expansion", add_help=add_help)

    parser.add_argument("--src_path", default="/home/ytw/data/bert_output/postln/pretrain/postln_noln_6L384H_scratch_W5kT220k_shuf_udfcos0d1_v1/checkpoint-epoch0002/", type=str, help="location of source model")
    parser.add_argument("--dst_path", default="/home/ytw/data/bert_output/postln/expanded/L6384_test/", type=str, help="location of saving destination model")
    parser.add_argument("--tokenizer", default='bert-base-uncased', type=str, help="tokenizer used for training")
    parser.add_argument("--dest_config", default='/home/ytw/code/lemon_test/lemon_test/lm/vlm/configs/bert-12L-768H.json', type=str, help='configuration of destination model.')
    parser.add_argument("--method", default='LEMON', type=str, choices=['LEMON',], help='configuration of destination model.')
    parser.add_argument("--depth_pattern", default='interpolation', type=str, choices=['stack', 'interpolation'], help='how to increase dpeth of model.')
    parser.add_argument("--AKI_pattern", default='next', type=str, choices=['next', 'random'], help='how to choose the AKI layer.')
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(args)
    main(args)
