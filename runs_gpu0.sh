#!/bin/bash

data=("3CLPro_pocket1" "ADRP-ADPR_pocket1" "ADRP-ADPR_pocket5" "ADRP_pocket12" "ADRP_pocket13" "ADRP_pocket1")

for f in "${data[@]}"
do
    name=ml.${f}_round1_dock
    dir=data_v3/DIR.${name}.images
    echo $f
    echo $name
    echo $dir

    gunzip $dir/$name.images.npy.gz
    # train
#    CUDA_VISIBLE_DEVICES=0 python train.py -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 --precomputed_values $dir/$name.reg.npy \
#			--lr 8e-5 -w 3 -i $dir/$name.smi -o $dir/model.pt --metric_plot_prefix $dir/$f --nheads 1 --dropout_rate 0.15 --amp O2 \
#			--precomputed_images $dir/$name.images.npy --width 128 --depth 2 -r 0 --scale

    # inference on train data
#    python train.py -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 --precomputed_values $dir/$name.reg.npy \
#	   --lr 8e-5 -w 3 -i $dir/$name.smi -o $dir/model.pt --nheads 1 --dropout_rate 0.15 --amp O2 \
#	   --precomputed_images $dir/$name.images.npy --width 128 --depth 2 -r 0 --scale --eval_train --output_preds $dir/log_train.txt

    # inference on test data
    python train.py -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 --precomputed_values $dir/$name.reg.npy \
           --lr 8e-5 -w 3 -i $dir/$name.smi -o $dir/model.pt --nheads 1 --dropout_rate 0.15 --amp O2 \
           --precomputed_images $dir/$name.images.npy --width 128 --depth 2 -r 0 --scale $dir/scaler.pkl --eval_test --output_preds $dir/out_test


    # plot enrichment surface
    #    python res.py $dir/out_test.npy $dir $f

    gzip  $dir/$name.images.npy
done
