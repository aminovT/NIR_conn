#!/usr/bin/env bash
#python3 train.py --dir=checkpoints/ConvFCNoBias/chp1 --dataset=CIFAR10 --data_path=data --model=ConvFCNoBias --epochs=5 --seed=1 --cuda
#
#python3 train.py --dir=checkpoints/ConvFCNoBias/chp2 --dataset=CIFAR10 --data_path=data --model=ConvFCNoBias --epochs=5 --seed=2 --cuda

#python eval_curve.py --dir=experiments/eval/ConvFCNoBias/PointFinderStepWiseButterflyConv --point_finder=PointFinderStepWiseButterflyConv --method=arc_connect --model=ConvFCNoBias --end_time=5 --data_path=data --num_points=61 --start=checkpoints/ConvFCNoBias/chp1/checkpoint-5.pt  --end=checkpoints/ConvFCNoBias/chp2/checkpoint-5.pt



#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100/ --data_path=data --model=VGG16 --name=400  --layer=6 --layer_ind=-2 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100

#python eval_curve.py --dir=experiments/eval/VGG16/PointFinderStepWiseButterflyConvWBias --point_finder=PointFinderStepWiseButterflyConvWBias --method=arc_connect --model=VGG16 --end_time=15 --data_path=data --num_points=61 --start=/home/ivan/dnn-mode-connectivity/curves/VGG16/curve1/checkpoint-400.pt  --end=/home/ivan/dnn-mode-connectivity/curves/VGG16/curve2/checkpoint-400.pt

#python eval_curve.py --dir=experiments/eval/VGG16subsample/PointFinderStepWiseButterflyConvWBias --point_finder=PointFinderStepWiseButterflyConvWBias --method=arc_connect --model=VGG16 --end_time=15 --data_path=data --device=0 --num_points=61 --start=/home/ivan/dnn-mode-connectivity/curves/VGG16/curve1/checkpoint-400.pt  --end=/home/ivan/dnn-mode-connectivity/curves/VGG16/curve2/checkpoint-400.pt

#python eval_curve.py --dir=experiments/eval/VGG16full/PointFinderStepWiseButterflyConvWBias --point_finder=PointFinderStepWiseButterflyConvWBias --method=arc_connect --model=VGG16 --end_time=15 --data_path=data --device=0 --num_points=61 --start=/home/ivan/dnn-mode-connectivity/curves/VGG16/curve1/checkpoint-400.pt  --end=/home/ivan/dnn-mode-connectivity/curves/VGG16/curve2/checkpoint-400.pt


#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100/ --data_path=data --model=VGG16 --name=400  --layer=15 --layer_ind=-2 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100 --name=200




#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w12/ --data_path=data --model=VGG16 --name=200  --layer=12 --layer_ind=-8 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w11/ --data_path=data --model=VGG16 --name=200  --layer=11 --layer_ind=-10 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w10/ --data_path=data --model=VGG16 --name=200  --layer=10 --layer_ind=-12 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w9/ --data_path=data --model=VGG16 --name=200  --layer=9 --layer_ind=-14 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w8/ --data_path=data --model=VGG16 --name=200  --layer=8 --layer_ind=-16 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w7/ --data_path=data --model=VGG16 --name=200  --layer=7 --layer_ind=-18 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w6/ --data_path=data --model=VGG16 --name=200  --layer=6 --layer_ind=-20 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w5/ --data_path=data --model=VGG16 --name=200  --layer=5 --layer_ind=-22 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w4/ --data_path=data --model=VGG16 --name=200  --layer=4 --layer_ind=-24 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w3/ --data_path=data --model=VGG16 --name=200  --layer=3 --layer_ind=-26 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w2/ --data_path=data --model=VGG16 --name=200  --layer=2 --layer_ind=-28 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w1/ --data_path=data --model=VGG16 --name=200  --layer=1 --layer_ind=-30 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w13/ --data_path=data --model=VGG16 --name=200  --layer=13 --layer_ind=-6 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w14/ --data_path=data --model=VGG16 --name=200  --layer=14 --layer_ind=-4 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w15/ --data_path=data --model=VGG16 --name=200  --layer=15 --layer_ind=-2 --model_paths=/home/ivan/distribution_connector/checkpoints/cifar100/VGG16 --dataset=CIFAR100


#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w3/ --data_path=data --model=VGG16 --name=400  --layer=3 --layer_ind=-26 --model_paths=checkpoints/VGG16 --dataset=CIFAR10


#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w15/ --data_path=data --model=VGG16 --name=400  --layer=15 --layer_ind=-2 --model_paths=checkpoints/VGG16 --dataset=CIFAR10




#python3 train.py --dir=checkpoints/VGG16/chp1 --dataset=CIFAR10 --data_path=data --model=VGG16 --epochs=400 --seed=1 --cuda
#
#python3 train.py --dir=checkpoints/VGG16/chp2 --dataset=CIFAR10 --data_path=data --model=VGG16 --epochs=400 --seed=2 --cuda
#
#python3 train.py --dir=checkpoints/VGG16/chp3 --dataset=CIFAR10 --data_path=data --model=VGG16 --epochs=400 --seed=3 --cuda
#
#python3 train.py --dir=checkpoints/VGG16/chp4 --dataset=CIFAR10 --data_path=data --model=VGG16 --epochs=400 --seed=4 --cuda
#
#python3 train.py --dir=checkpoints/VGG16/chp5 --dataset=CIFAR10 --data_path=data --model=VGG16 --epochs=400 --seed=5 --cuda
#
#python3 train.py --dir=checkpoints/VGG16/chp6 --dataset=CIFAR10 --data_path=data --model=VGG16 --epochs=400 --seed=6 --cuda
#
#python3 train.py --dir=checkpoints/VGG16/chp7 --dataset=CIFAR10 --data_path=data --model=VGG16 --epochs=400 --seed=7 --cuda
#
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp1 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=1 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp2 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=2 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp3 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=3 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp4 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=4 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp5 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=5 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp6 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=6 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp7 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=7 --cuda


#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w12/ --data_path=data --model=VGG16 --name=400  --layer=12 --layer_ind=-8 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w11/ --data_path=data --model=VGG16 --name=400  --layer=11 --layer_ind=-10 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w10/ --data_path=data --model=VGG16 --name=400  --layer=10 --layer_ind=-12 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w9/ --data_path=data --model=VGG16 --name=400  --layer=9 --layer_ind=-14 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w8/ --data_path=data --model=VGG16 --name=400  --layer=8 --layer_ind=-16 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w7/ --data_path=data --model=VGG16 --name=400  --layer=7 --layer_ind=-18 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w6/ --data_path=data --model=VGG16 --name=400  --layer=6 --layer_ind=-20 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w5/ --data_path=data --model=VGG16 --name=400  --layer=5 --layer_ind=-22 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w4/ --data_path=data --model=VGG16 --name=400  --layer=4 --layer_ind=-24 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w3/ --data_path=data --model=VGG16 --name=400  --layer=3 --layer_ind=-26 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w2/ --data_path=data --model=VGG16 --name=400  --layer=2 --layer_ind=-28 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w1/ --data_path=data --model=VGG16 --name=400  --layer=1 --layer_ind=-30 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w13/ --data_path=data --model=VGG16 --name=400  --layer=13 --layer_ind=-6 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w14/ --data_path=data --model=VGG16 --name=400  --layer=14 --layer_ind=-4 --model_paths=checkpoints/VGG16 --dataset=CIFAR10
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16w15/ --data_path=data --model=VGG16 --name=400  --layer=15 --layer_ind=-2 --model_paths=checkpoints/VGG16 --dataset=CIFAR10


#python eval_curve.py --dir=experiments/eval/VGG16lin/PointFinderStepWiseButterflyConvWBias --point_finder=PointFinderStepWiseButterflyConvWBias --method=lin_connect --model=VGG16 --end_time=15 --data_path=data --device=0 --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt

#python eval_curve.py --dir=experiments/eval/VGG16arc/PointFinderStepWiseButterflyConvWBias --point_finder=PointFinderStepWiseButterflyConvWBias --method=arc_connect --model=VGG16 --end_time=15 --data_path=data --device=0 --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt

#python eval_curve.py --dir=experiments/eval/VGG16lin/PointFinderStepWiseButterflyConvWBiasOT --point_finder=PointFinderStepWiseButterflyConvWBiasOT --method=lin_connect --model=VGG16 --end_time=15 --data_path=data --device=0 --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt
#
#python eval_curve.py --dir=experiments/eval/VGG16lin/PointFinderStepWiseButterflyConvWBiasOT2 --point_finder=PointFinderStepWiseButterflyConvWBiasOT2 --method=lin_connect --model=VGG16 --end_time=15 --data_path=data --device=0 --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt

#python eval_curve.py --dir=experiments/eval/VGG16arc/PointFinderStepWiseButterflyConvWBiasOT --point_finder=PointFinderStepWiseButterflyConvWBiasOT --method=arc_connect --model=VGG16 --end_time=15 --data_path=data --device=0 --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt

#python eval_curve.py --dir=experiments/eval/VGG16arc/PointFinderStepWiseButterflyConvWBiasOT2 --point_finder=PointFinderStepWiseButterflyConvWBiasOT2 --method=arc_connect --model=VGG16 --end_time=15 --data_path=data --device=0 --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt

#python3 train.py --dir=checkpoints/cifar100/VGG16/chp1 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=1 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp2 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=2 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp3 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=3 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp4 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=4 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp5 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=5 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp6 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=6 --cuda
#
#python3 train.py --dir=checkpoints/cifar100/VGG16/chp7 --dataset=CIFAR100 --data_path=data --model=VGG16 --epochs=200 --seed=7 --cuda


#python3 count_operations.py --dir=experiments/number_operations/VGG16/ --dataset=CIFAR10 --data_path=data --model=VGG16 --ckpt=checkpoints/VGG16/chp2/checkpoint-400.pt


#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w15/ --data_path=data --model=VGG16 --name=200  --layer=15 --layer_ind=-2 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100

#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w3/ --data_path=data --model=VGG16 --name=200  --layer=3 --layer_ind=-26 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w2/ --data_path=data --model=VGG16 --name=200  --layer=2 --layer_ind=-28 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w4/ --data_path=data --model=VGG16 --name=200  --layer=4 --layer_ind=-24 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w5/ --data_path=data --model=VGG16 --name=200  --layer=5 --layer_ind=-22 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w6/ --data_path=data --model=VGG16 --name=200  --layer=6 --layer_ind=-20 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w7/ --data_path=data --model=VGG16 --name=200  --layer=7 --layer_ind=-18 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w8/ --data_path=data --model=VGG16 --name=200  --layer=8 --layer_ind=-16 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w9/ --data_path=data --model=VGG16 --name=200  --layer=9 --layer_ind=-14 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w10/ --data_path=data --model=VGG16 --name=200  --layer=10 --layer_ind=-12 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#
#python eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w11/ --data_path=data --model=VGG16 --name=200  --layer=11 --layer_ind=-10 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100


#python eval_curve.py --dir=experiments/eval/VGG16lin/PointFinderStepWiseButterflyConvWBiasWA --point_finder=PointFinderStepWiseButterflyConvWBias --method=arc_connect --model=VGG16 --beg_time=0 --end_time=15 --data_path=data --device=0 --num_points=31 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt
#
#python eval_curve.py --dir=experiments/eval/VGG16lin/PointFinderStepWiseButterflyConvWBiasWA --point_finder=PointFinderStepWiseButterflyConvWBias --method=arc_connect --model=VGG16 --beg_time=0 --end_time=15 --data_path=data --device=0 --num_points=31 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt
#
#python eval_curve.py --dir=experiments/eval/VGG16arcCheck/PointFinderStepWiseButterflyConvWBias --point_finder=PointFinderStepWiseButterflyConvWBias --method=arc_connect --model=VGG16 --end_time=15 --data_path=data --device=0 --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt


python eval_curve.py --dir=experiments/eval/VGG16arc/PointFinderStepWiseButterflyConvWBiasWA --point_finder=PointFinderStepWiseButterflyConvWBiasWA --method=arc_connect --model=VGG16 --beg_time=0 --end_time=15 --data_path=data --device=0 --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt

python eval_curve.py --dir=experiments/eval/VGG16lin/PointFinderStepWiseButterflyConvWBiasWA --point_finder=PointFinderStepWiseButterflyConvWBiasWA --method=lin_connect --model=VGG16 --beg_time=0 --end_time=15 --data_path=data --device=0 --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt

