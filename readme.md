# Molecular Image Attention Models



### Covid Data

If images have not been generated yet -- from raw data .csv file split the smiles column into a separate .csv file and then generate images with:
```
python generate_images.py -i <input smiles csv file> -o <output image npy file>
  ```
Then create data directories for each target with:
```
python process.py -r <raw csv file> -i <generated images .npy file>
```
Each directory will contain a <target_name>.smi file with smiles, a <target_name>.reg.npy file with dock scores, and a <target_name>.images.npy file with images

Training: (saves model, training curve plots, scaling object)
```
python train.py -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 --precomputed_values <target_name>.reg.npy \
                        --lr 8e-5 -w 3 -i <target_name>.smi -o model.pt --metric_plot_prefix <target_name> \
                        --nheads 1 --dropout_rate 0.15 -- amp O2  --precomputed_images <target_name>.images.npy  --width 128 \
                        --depth 2 -r 0 --scale scaler.pkl
```        

Evaluation: (prints out mae, pearson correlation, saves enrichment surface plot, prediction array)
```
# evaluate on test data                                                                                                             
python train.py --eval_test --output_preds out_test  -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 \ 
               --precomputed_values <target_name>.reg.npy --lr 8e-5 -w 3 -i <target_name>.smi -o model.pt --nheads 1 \
               --dropout_rate 0.15 --amp O2 --precomputed_images <target_name>.images.npy --width 128 --depth 2 -r 0 --scale scaler.pkl > log.txt
           

# plot enrichment surface                                                  
python res.py out_test.npy . <target_name>
 ```
 
 Inference: (saves prediction array)
 ```
python train.py --infer --output_preds out_preds -pb --rotate --mae -t 1 -p custom -b 512 --epochs 50 \
           --precomputed_values <target_name>.reg.npy --lr 8e-5 -w 3 -i <inference-set>.smi -o model.pt --nheads 1 \
           --dropout_rate 0.15 --amp O2 --precomputed_images <inference-set>.images.npy --width 128 --depth 2 -r 0 --scale scaler.pkl 
 ```
