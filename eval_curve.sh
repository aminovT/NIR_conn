#!/usr/bin/env bash

#PointFinderWithBias lin_connect for LinearOneLayer100 model
python eval_curve.py --dir=experiments/eval/LinearOneLayer100/PointFinderWithBias --point_finder=PointFinderWithBias --method=lin_connect --model=LinearOneLayer100 --end_time=1 --data_path=data --num_points=21 --start=checkpoints/LinearOneLayer100/chp1/checkpoint-400.pt  --end=checkpoints/LinearOneLayer100/chp2/checkpoint-400.pt --cuda
#PointFinderInverseWithBias arc_connect for LinearOneLayer100 model
python eval_curve.py --dir=experiments/eval/LinearOneLayer100/PointFinderInverseWithBias --point_finder=PointFinderInverseWithBias --method=arc_connect --model=LinearOneLayer100 --end_time=2 --data_path=data --num_points=21 --start=checkpoints/LinearOneLayer100/chp1/checkpoint-400.pt  --end=checkpoints/LinearOneLayer100/chp2/checkpoint-400.pt --cuda
#PointFinderTransportation lin_connect for LinearOneLayer100 model
python eval_curve.py --dir=experiments/eval/LinearOneLayer100/PointFinderTransportation --point_finder=PointFinderTransportation --method=lin_connect --model=LinearOneLayer100 --end_time=1 --data_path=data --num_points=21 --start=checkpoints/LinearOneLayer100/chp1/checkpoint-400.pt  --end=checkpoints/LinearOneLayer100/chp2/checkpoint-400.pt --cuda
#PointFinderInverseWithBiasOT lin_connect for LinearOneLayer100 model
python eval_curve.py --dir=experiments/eval/LinearOneLayer100/PointFinderInverseWithBiasOT --point_finder=PointFinderInverseWithBiasOT --method=lin_connect --model=LinearOneLayer100 --end_time=2 --data_path=data --num_points=21 --start=checkpoints/LinearOneLayer100/chp1/checkpoint-400.pt  --end=checkpoints/LinearOneLayer100/chp2/checkpoint-400.pt --cuda


python eval_curve.py --dir=experiments/eval/LinearOneLayer100/PointFinderInverseWithBias --point_finder=PointFinderInverseWithBias --method=lin_connect --model=LinearOneLayer100 --end_time=2 --data_path=data --num_points=21 --start=checkpoints/LinearOneLayer100/chp1/checkpoint-400.pt  --end=checkpoints/LinearOneLayer100/chp2/checkpoint-400.pt --cuda
