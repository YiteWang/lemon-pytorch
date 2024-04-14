from datasets import load_from_disk, load_dataset
dataset = load_dataset(path="imagenet-1k", 
    data_dir='/dataset/imagenet1k/data_dir', 
    cache_dir='/dataset/imagenet1k/cache_dir').with_format("torch")
dataset.save_to_disk('/dataset/imagenet1k')
print(dataset)
# Use the following code to check the data can be loaded correctly.
dataset = load_from_disk('/dataset/imagenet1k').with_format("torch")['train']
print(dataset)