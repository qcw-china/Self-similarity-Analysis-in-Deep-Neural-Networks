export OMP_NUM_THREADS=4
#'vit_small_', 'vit_base_', 'resmlp_12_224_we', 'pvt_v2', 'mlp_512', 'poolformer', 'resnet_34_timm', 'resnet_50_timm','mxier','vgg'

torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-3 --data-set IMNETTE  --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/poolformer_IMNETTE_noSrate --model poolformer  --seed 0   --only-record-sfm-rate --ThreeAugment --src

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-3 --data-set IMNETTE  --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/poolformer_IMNETTE_useSrate --model poolformer --use-sfm-loss --seed 0 --sfm-weight 1e-4 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# --ThreeAugment --src
# --only-record-sfm-rate
# --resume ../save/mxier_IMNETTE_useSrate/checkpoint.pth


# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-1 --data-set CIFAR100  --lr 5e-03 --epochs 200 --batch-size 64 --output_dir ../save/resmlp_12_224_we_CIFAR100_noSrate --model resmlp_12_224_we --seed 0 --only-record-sfm-rate

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-2 --data-set CIFAR10 --lr 5e-03 --epochs 200 --batch-size 64 --output_dir ../save/vit_small__we_CIFAR10_noSrate --model vit_small_ --seed 0   --only-record-sfm-rate 

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-3 --data-set IMNETTE --lr 5e-03 --epochs 200 --batch-size 64 --output_dir ../save/vit_small__we_IMNETTE_noSrate --model vit_small_ --seed 0   --only-record-sfm-rate 

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-1 --data-set CIFAR100 --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_small_CIFAR100_noSrate --model vit_small_ --seed 0   --only-record-sfm-rate 

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-2 --data-set CIFAR10 --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_small_CIFAR10_noSrate --model vit_small_ --seed 0   --only-record-sfm-rate 

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-3 --data-set IMNETTE --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_small_IMNETTE_noSrate --model vit_small_ --seed 0   --only-record-sfm-rate 

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-1 --data-set CIFAR100 --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_base_CIFAR100_noSrate --model vit_base_ --seed 0   --only-record-sfm-rate 

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-2 --data-set CIFAR10 --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_base_CIFAR10_noSrate --model vit_base_ --seed 0   --only-record-sfm-rate 

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-3 --data-set IMNETTE --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_base_IMNETTE_noSrate --model vit_base_ --seed 0   --only-record-sfm-rate 

# torchrun --nproc_per_node=2 main.py --input-size 224 --data-path ../../data-3 --data-set IMNETTE  --lr 5e-03 --epochs 200 --batch-size 64 --output_dir ../save/mxier_IMNETTE_useSrate --model mxier  --use-sfm-loss --seed 0 --sfm-weight 1e-2 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-3 --data-set IMNETTE  --lr 5e-03 --epochs 200 --batch-size 64 --output_dir ../save/resmlp_12_224_we_IMNETTE_useSrate --model resmlp_12_224_we  --use-sfm-loss --seed 0 --sfm-weight 5e-2 --sample-epoch-fre 3 --Change-amplitude-limit 0.1 --target-sfm-rate 0.2

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-3 --data-set IMNETTE  --lr 5e-03 --epochs 200 --batch-size 64 --output_dir ../save/vit_small_IMNETTE_useSrate --model vit_small_  --use-sfm-loss --seed 0 --sfm-weight 5e-2 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-2 --data-set CIFAR10  --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_small_CIFAR10_useSrate --model vit_small_  --use-sfm-loss --seed 0 --sfm-weight 5e-2 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-1 --data-set CIFAR100  --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_small_CIFAR100_useSrate --model vit_small_  --use-sfm-loss --seed 0 --sfm-weight 5e-2 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-1 --data-set CIFAR100  --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_base_CIFAR100_useSrate --model vit_base_  --use-sfm-loss --seed 0 --sfm-weight 5e-2 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-2 --data-set CIFAR10  --lr 1e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_base_CIFAR10_useSrate --model vit_base_  --use-sfm-loss --seed 0 --sfm-weight 5e-2 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-3 --data-set IMNETTE --lr 1e-04 --epochs 200 --batch-size 64 --output_dir ../save/vit_base_IMNETTE_useSrate --model vit_base_  --use-sfm-loss --seed 0 --sfm-weight 5e-2 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# # torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-1 --data-set CIFAR100 --lr 7e-03 --epochs 200 --batch-size 64 --output_dir ../save/resmlp_12_224_we_CIFAR100_useSrate --model resmlp_12_224_we  --use-sfm-loss --seed 0 --sfm-weight 5e-2 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# # torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-2 --data-set CIFAR10 --lr 7e-03 --epochs 200 --batch-size 64 --output_dir ../save/resmlp_12_224_we_CIFAR10_useSrate --model resmlp_12_224_we  --use-sfm-loss --seed 0 --sfm-weight 5e-2 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-1 --data-set CIFAR100  --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/mxier_CIFAR100_useSrate --model mxier  --use-sfm-loss --seed 0 --sfm-weight 5e-7 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.2

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-1 --data-set CIFAR100  --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/mxier_CIFAR100_useSrate --model mxier  --use-sfm-loss --seed 0 --sfm-weight 5e-7 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.2

# torchrun --nproc_per_node=4 main.py --input-size 224 --data-path ../../data-2 --data-set CIFAR10  --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/mxier_CIFAR10_useSrate --model mxier  --use-sfm-loss --seed 0 --sfm-weight 5e-7 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.2 --resume ../save/mxier_CIFAR10_useSrate/checkpoint.pth