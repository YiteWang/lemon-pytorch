from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    BertModel,
)
import copy
import preln_bert
import inflate_bert_utils
import os

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    modconfig = BertConfig.from_pretrained(os.path.join(args.src_path, 'config.json'))

    print(modconfig)
    inf_modconfig = BertConfig.from_pretrained(args.dest_config)

    modmodel = preln_bert.modBertForMaskedLM.from_pretrained(
        os.path.join(args.src_path, 'pytorch_model.bin'),
        config=modconfig,
    )
    inf_modmodel = preln_bert.modBertForMaskedLM(inf_modconfig)

    text = "Replace me by any text you'd like. This is a bright day."
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input_wo = copy.deepcopy(encoded_input)
    encoded_input_wo.pop('attention_mask')
    modmodel.eval()
    inf_modmodel.eval()
    inflate_bert_utils.inflate_LEMON(modmodel, inf_modmodel)

    out = modmodel(**encoded_input)
    inf_out = inf_modmodel(**encoded_input)
    print('Maximum difference is: {}'.format((inf_out['logits']-out['logits']).abs().max().item()))

    inf_modmodel.save_pretrained(args.dest_path)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Huggingface model expansion for Pre-LN BERT", add_help=add_help)

    parser.add_argument("--src_path", default="/L6H384_W5kT220k_shuf_udfcos0d1_v1/checkpoint-epoch0002/", type=str, help="location of source model")
    parser.add_argument("--dest_path", default="/expanded_models/", type=str, help="location of saving destination model")
    parser.add_argument("--tokenizer", default='bert-base-uncased', type=str, help="tokenizer used for training")
    parser.add_argument("--dest_config", default='/lemon_test/lm/vlm/configs/bert-12L-768H.json', type=str, help='configuration of destination model.')
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
