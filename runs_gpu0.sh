#!/bin/bash

data=('NSUN6')
#data=( '3CLPRO_1' '3CLPRO_2' '3CLPRO_3' 'ADRP_ADPR_A'  'COV_RDB_AB'  'COV_RDB_A_1')
for f in "${data[@]}"
do
    name=ml.${f}
    dir=data_v5/DIR.${name}.images
    echo $f
    echo $name
    echo $dir

    gunzip $dir/$name.images.npy.gz
    # train
    CUDA_VISIBLE_DEVICES=0 python train.py -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 --precomputed_values $dir/$name.reg.npy \
			--lr 8e-5 -w 3 -i $dir/$name.smi -o $dir/model.pt --metric_plot_prefix $dir/$f --nheads 1 --dropout_rate 0.15 --amp O2 \
			--precomputed_images $dir/$name.images.npy --width 128 --depth 2 -r 0 --scale $dir/scaler.pkl

    # inference on train data
     CUDA_VISIBLE_DEVICES=0 python train.py -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 --precomputed_values $dir/$name.reg.npy \
	   --lr 8e-5 -w 3 -i $dir/$name.smi -o $dir/model.pt --nheads 1 --dropout_rate 0.15 --amp O2 \
	   --precomputed_images $dir/$name.images.npy --width 128 --depth 2 -r 0 --scale $dir/scaler.pkl --eval_train --output_preds $dir/out_train > $dir/log_train.txt

    # inference on test data
     CUDA_VISIBLE_DEVICES=0 python train.py -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 --precomputed_values $dir/$name.reg.npy \
           --lr 8e-5 -w 3 -i $dir/$name.smi -o $dir/model.pt --nheads 1 --dropout_rate 0.15 --amp O2 \
           --precomputed_images $dir/$name.images.npy --width 128 --depth 2 -r 0 --scale $dir/scaler.pkl --eval_test --output_preds $dir/out_test > $dir/log_test.txt


    # plot enrichment surface
     python res.py $dir/out_test.npy $dir $f

    gzip  $dir/$name.images.npy
done
