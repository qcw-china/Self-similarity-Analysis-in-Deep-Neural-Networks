# Self-similarity-Analysis-in-Deep-Neural-Networ

you can use these code  "python main.py --input-size 224 --data-path ./data-1 --data-set CIFAR10   --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/poolformer_CIFAR10_noSrate --model poolformer  --seed 0   --only-record-sfm-rate"  
and "python main.py --input-size 224 --data-path ./data-1 --data-set CIFAR10  --lr 5e-04 --epochs 200 --batch-size 64 --output_dir ../save/poolformer_CIFAR10_useSrate --model poolformer --use-sfm-loss --seed 0 --sfm-weight 1e-4 --sample-epoch-fre 5 --Change-amplitude-limit 0.05 --target-sfm-rate 0.3" to reproduce our experiments.

--use-sfm-loss    This is used to control whether the self-similarity constraint is enabled or not.
--sfm-weight    Control the intensity of the srate regularization            
--sample-epoch-fre    The epoch frequency for varying the Srate sampling  
--target-sfm-rate    The value of the srate when the unconstrained model converges
