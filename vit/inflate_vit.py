import torch
from main import get_args_parser
import argparse
from timm.models import create_model
from hf_datasets import build_dataset
from inflate_vit_utils import inflate_vit
from pathlib import Path
import os
import time

parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
parser.add_argument('--mode', default='proj', type=str, choices=['proj',])
parser.add_argument('--src_ckpt', default="/home/ytw/data/vit/small/deit_small_patch16_224-cd65a155.pth",
                type=str, help='Checkpoint of the pre-trained model.')
parser.add_argument('--model_src', default='deit_small_patch16_224', choices =['deit_L6H512_patch16_224', 'deit_small_HD_patch16_224', 'deit_small_patch16_224'],
                type=str, help='Model name of the source/pre-trained model.')
parser.add_argument('--model_dst', default='deit_base_patch16_224',
                type=str, help='Model name of the target/destination/inflated model.')
parser.add_argument('--depth-inflate', default='copymix_cancel', choices=['copymix_cancel', 'copymix_zero'], help='How to perform lossless depth inflation.')
parser.add_argument('--expanded_dir', default='/home/ytw/data/vit/expanded/', type=str, help='Path to save the expanded ckpt.')
args = parser.parse_args()

args.batch_size = 256
print(args)

dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
dataset_val, _ = build_dataset(is_train=False, args=args)
model_src = create_model(
    args.model_src,
    pretrained=False,
    num_classes=args.nb_classes,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    drop_block_rate=None,
    img_size=args.input_size
).to(args.device)

model_dst = create_model(
    args.model_dst,
    pretrained=False,
    num_classes=args.nb_classes,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    drop_block_rate=None,
    img_size=args.input_size
).to(args.device)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=int(1.5 * args.batch_size),
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False
)

# load current model
model_src_ckpt = torch.load(args.src_ckpt, map_location=args.device)
model_src.load_state_dict(model_src_ckpt['model'])

# Expand model
start_time = time.time()
inflate_vit(model_src, model_dst, mode=args.mode, device=args.device, depth_inflate=args.depth_inflate)
end_time = time.time()
print('Total time: {}'.format(end_time-start_time))

# Evaluate inflated model
total = 0 
correct_src = 0
correct_dst = 0

model_dst.eval()
model_dst.to(args.device)
model_src.eval()
model_src.to(args.device)

print('Start to evaluate expanded model for 5 batches.')
max_diff = 0
with torch.no_grad():
    for i, data in enumerate(data_loader_val):
        image = data['image'].cuda()
        out_src = model_src(image)
        out_dst = model_dst(image)
        max_diff = max((out_src-out_dst).abs().max(), max_diff)
        if i == 5:
            break
        correct_src += (out_src.argmax(-1) == data['label'].cuda()).sum()
        correct_dst += (out_dst.argmax(-1) == data['label'].cuda()).sum()
        total += data['label'].size(0)
print('Max difference in 5 batches is: {}'.format(max_diff))
print('Acc of src after inflation for 10 batches is: {}'.format(correct_src/total))
print('Acc of dst after inflation for 10 batches is: {}'.format(correct_dst/total))
model_dst.train()
model_dst_ckpt = {
    'model': model_dst.state_dict(),
}

Save model
output_dir = os.path.join(args.expanded_dir, args.model_src)
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
file_dir = os.path.join(output_dir, 'expanded_model.pth')
print(file_dir)
torch.save(model_dst_ckpt, file_dir)