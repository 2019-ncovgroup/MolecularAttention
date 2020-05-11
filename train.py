import argparse
import multiprocessing

import numpy as np
import pandas as pd
import torch
from apex import amp
from rdkit import Chem
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from features.datasets import ImageDataset, funcs, get_properety_function, ImageDatasetPreLoaded
from features.generateFeatures import MORDRED_SIZE
from metrics import trackers
from models import imagemodel
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pickle
if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Timer(object):
    """Timer class                                                                                       
       Wrap a will with a timing function                                                                
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *args, **kwargs):
        print("{} took {} seconds".format(
        self.name, time.time() - self.t))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_optimizer(c):
    if c == 'sgd':
        return torch.optim.SGD
    elif c == 'adam':
        return torch.optim.Adam
    elif c == 'adamw':
        return torch.optim.AdamW


def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return smiles if mol is not None else None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='smiles input file')
    parser.add_argument('--precomputed_values', type=str, required=False, default=None,
                        help='precomputed decs for trainings')
    parser.add_argument('--imputer_pickle', type=str, required=False, default=None,
                        help='imputer and scaler for transforming data')
    parser.add_argument('--precomputed_images', type=str, required=False, default=None)
    parser.add_argument('-p', choices=list(funcs.keys()), help='select property for model')
    parser.add_argument('-w', type=int, default=8, help='number of workers for data loaders to use.')
    parser.add_argument('-b', type=int, default=64, help='batch size to use')
    parser.add_argument('-o', type=str, default='saved_models/model.pt', help='name of file to save model to')
    parser.add_argument('-r', type=int, default=32, help='random seed for splitting.')
    parser.add_argument('-pb', action='store_true')
    parser.add_argument('--cyclic', action='store_true')
    parser.add_argument('-t', type=int, default=1, help='number of tasks')
    parser.add_argument('--nheads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--metric_plot_prefix', default=None, type=str, help='prefix for graphs for performance')
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer to use',
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning to use')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs to use')
    parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--eval_test', action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--classifacation', action='store_true')
    parser.add_argument('--ensemble_eval', action='store_true')
    parser.add_argument('--mae', action='store_true')
    parser.add_argument('--cv', default=None, type=int, help='use CV for crossvalidation (1-5)')
    parser.add_argument('--width', default=256, type=int, help='rep size')
    parser.add_argument('--depth', default=2, type=int, help='rep size')
    parser.add_argument('--amp', type=str, default='O0', choices=['O0', 'O1', 'O2', 'O3'])
    parser.add_argument('--bw', action='store_true')
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--output_preds', default=None, type=str, required=False, help='output preds when running eval')
    parser.add_argument('--scale', type=str, default=None, help='name of file to save/load for dock score scaler')
    parser.add_argument('--infer', action='store_true')

    args = parser.parse_args()
    if args.metric_plot_prefix is None:
        args.metric_plot_prefix = "".join(args.o.split(".")[:-1]) + "_"
    args.optimizer = get_optimizer(args.optimizer)
    if args.p == 'all' and args.t == 1:
        print("You chose all, but didn't only selected 1 task...")
        print("Setting to MOrdred default")
        args.t = MORDRED_SIZE
    print(args)
    return args


def run_infer(model, data_loader, tasks=1, mae=True, pb=True, output_preds=None, scaler=None):
    preds = []
    with torch.no_grad():
        model.eval()
        for i, drugfeats in tqdm(enumerate(data_loader)):
            drugfeats = drugfeats.to(device)
            pred, attn = model(drugfeats)
            preds.append(pred.detach().cpu().numpy().flatten())
    preds = np.asarray(np.concatenate(preds, axis=0))
    print(preds.shape)
    if scaler is not None:
        print("Inverting docking scaling")
        preds = scaler.inverse_transform(preds.reshape(-1,1)).flatten()

    if output_preds is not None:
        np.save(output_preds, preds)
            
def run_eval(model, train_loader, ordinal=False, classifacation=False, enseml=True, tasks=1, mae=False, pb=True, output_preds=None, scaler=None):
    with torch.no_grad():
        model.eval()
        if classifacation:
            tracker = trackers.ComplexPytorchHistory(metric=metrics.roc_auc_score,
                                                     metric_name='roc-auc') if tasks > 1 else trackers.PytorchHistory(
                metric=metrics.roc_auc_score, metric_name='roc-auc')
        else:
            tracker = trackers.ComplexPytorchHistory() if tasks > 1 else trackers.PytorchHistory()

        train_loss = 0
        test_loss = 0
        train_iters = 0
        test_iters = 0
        preds = []
        values = []
        predss = []
        valuess = []
        model.eval()

        if pb and enseml:
            first_range = tqdm(range(50), desc='ensembl runs')
        else:
            first_range = range(1)
        for _ in first_range:
            predss = []
            valuess = []
            if pb and not enseml:
                second_range = tqdm(enumerate(train_loader))
            else:
                second_range = enumerate(train_loader)
            for i, (drugfeats, value) in second_range:
                drugfeats, value = drugfeats.to(device), value.to(device)
                pred, attn = model(drugfeats)

                if classifacation:
                    mse_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, value).mean()
                elif mae:
                    mse_loss = torch.nn.functional.l1_loss(pred, value).mean()
                else:
                    mse_loss = torch.nn.functional.mse_loss(pred, value).mean()
                test_loss += mse_loss.item()
                test_iters += 1
                tracker.track_metric(pred.detach().cpu().numpy(), value.detach().cpu().numpy())
                valuess.append(value.cpu().detach().numpy().flatten())
                predss.append(pred.detach().cpu().numpy().flatten())

            preds.append(np.concatenate(predss, axis=0))
            values.append(np.concatenate(valuess, axis=0))
        preds = np.stack(preds)
        values = np.stack(values)
        print(preds.shape, values.shape)

        # if dock scores were scaled, unscale them here
        if scaler is not None:
            print("Inverting docking scaling")
            preds = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
            values = scaler.inverse_transform(values.reshape(-1,1)).flatten()
            
        if output_preds is not None:
            print(preds.shape)
            print(values.shape)
            out = np.stack([preds, values])
            print(out.shape)
            print("Outputting preds")
            np.save(output_preds, out)
            del out

       # preds = np.mean(preds, axis=0)
       # values = np.mean(values, axis=0)

       # if ordinal:
       #      preds, values = np.round(preds), np.round(values)
       #      incorrect = 0
       #      for i in range(preds.shape[0]):
       #      if values[i] != preds[i]:
                    # print("incorrect at", i)
       #             incorrect += 1
       #     print("total total", preds.shape[0])
       #     print("total incorrect", incorrect, incorrect / preds.shape[0])

        tracker.log_loss(test_loss / test_iters, train=False)
        tracker.log_metric(internal=True, train=False)

        print("val", test_loss / test_iters, 'r2', tracker.get_last_metric(train=False))
        print(values.shape, preds.shape)
        print("Pearson correlation", stats.pearsonr(values, preds))
        print("avg ensmelb r2, mae", metrics.r2_score(values, preds), metrics.mean_absolute_error(values, preds))

    return model, tracker


def trainer(model, optimizer, train_loader, test_loader, epochs=5, tasks=1, classifacation=False, mae=False,
            pb=True, out="model.pt", cyclic=False, verbose=True, use_mask=False):
    device = next(model.parameters()).device
    if classifacation:
        tracker = trackers.ComplexPytorchHistory(metric=metrics.roc_auc_score,
                                                 metric_name='roc-auc') if tasks > 1 else trackers.PytorchHistory(
            metric=metrics.roc_auc_score, metric_name='roc-auc')
    else:
        tracker = trackers.ComplexPytorchHistory() if tasks > 1 else trackers.PytorchHistory()

    earlystopping = EarlyStopping(patience=50, delta=1e-5)
    if cyclic:
        lr_red = CosineAnnealingWarmRestarts(optimizer, T_0=20)
    else:
        lr_red = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, cooldown=0, verbose=verbose,
                                   threshold=1e-4,
                                   min_lr=1e-8)

    for epochnum in range(epochs):
        train_loss = 0
        test_loss = 0
        train_iters = 0
        test_iters = 0
        model.train()
        if pb:
            gen = tqdm(enumerate(train_loader), total=int(len(train_loader.dataset) / train_loader.batch_size),
                       desc='training')
        else:
            gen = enumerate(train_loader)
        for v in gen:
            if use_mask:
                i, (drugfeats, value, mask) = v
                mask = mask.float().to(device)
            else:
                i, (drugfeats, value) = v
            optimizer.zero_grad()
            drugfeats, value = drugfeats.to(device), value.to(device)
            pred, attn = model(drugfeats)

            if classifacation:
                mse_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, value)
            elif mae:
                mse_loss = torch.nn.functional.l1_loss(pred, value)
            else:
                mse_loss = torch.nn.functional.mse_loss(pred, value)
            if use_mask:
                mse_loss = (mask * mse_loss).sum() / torch.sum(mask)
            else:
                mse_loss = mse_loss.mean()
            with amp.scale_loss(mse_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += mse_loss.item()
            train_iters += 1
            if use_mask:
                pred = pred.detach().cpu()
                value = value.detach().cpu()
                tracker.track_metric(pred=pred.numpy(), value=value.numpy(), mask=mask.detach().cpu().numpy())
            else:
                pred = pred.detach().cpu()
                value = value.detach().cpu()
                tracker.track_metric(pred=pred.numpy(), value=value.numpy())

        tracker.log_loss(train_loss / train_iters, train=True)
        tracker.log_metric(internal=True, train=True)

        model.eval()
        with torch.no_grad():
            if pb:
                gen = tqdm(enumerate(test_loader), total=int(len(test_loader.dataset) / test_loader.batch_size),
                           desc='eval')
            else:
                gen = enumerate(test_loader)
            for v in gen:
                if use_mask:
                    i, (drugfeats, value, mask) = v
                    mask = mask.float().to(device)
                else:
                    i, (drugfeats, value) = v
                drugfeats, value = drugfeats.to(device), value.to(device)
                pred, attn = model(drugfeats)

                if classifacation:
                    mse_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, value)
                elif mae:
                    mse_loss = torch.nn.functional.l1_loss(pred, value)
                else:
                    mse_loss = torch.nn.functional.mse_loss(pred, value)
                if use_mask:
                    mse_loss = mask * mse_loss
                mse_loss = mse_loss.mean()
                test_loss += mse_loss.item()
                test_iters += 1

                if use_mask:
                    pred = pred.detach().cpu()
                    value = value.detach().cpu()
                    tracker.track_metric(pred=pred.numpy(), value=value.numpy(), mask=mask.detach().cpu().numpy())
                else:
                    pred = pred.detach().cpu()
                    value = value.detach().cpu()
                    tracker.track_metric(pred=pred.numpy(), value=value.numpy())
        tracker.log_loss(test_loss / test_iters, train=False)
        tracker.log_metric(internal=True, train=False)

        lr_red.step(test_loss / test_iters)
        earlystopping(test_loss / test_iters)
        if verbose:
            print("Epoch", epochnum, train_loss / train_iters, test_loss / test_iters, tracker.metric_name,
                  tracker.get_last_metric(train=True), tracker.get_last_metric(train=False))

        if out is not None:
            state = model.state_dict()
            heads = model.nheads
            torch.save({'model_state': state,
                        'opt_state': optimizer.state_dict(),
                        'history': tracker,
                        'nheads': heads,
                        'ntasks': tasks,
                        'args': args,
                        'amp': amp.state_dict()}, out)
        if earlystopping.early_stop:
            break
    return model, tracker


def load_data_models(fname, random_seed, workers, batch_size, pname='logp', return_datasets=False, nheads=1,
                     precompute_frame=None, imputer_pickle=None, eval=False, tasks=1, cvs=None, rotate=False,
                     classifacation=False, ensembl=False, dropout=0, intermediate_rep=None, precomputed_images=None,
                     depth=None, bw=True, mask=None, pretrain=True, scale=None, infer=False):
    df = pd.read_csv(fname, header=None)
    smiles = []
    with multiprocessing.Pool() as p:
        gg = filter(lambda x: x is not None, p.imap(validate_smiles, list(df.iloc[:, 0])))
        for g in tqdm(gg, desc='validate smiles'):
            smiles.append(g)
    del df

    if cvs is not None:
        if classifacation and tasks == 1 and precompute_frame is not None:
            ts = np.load(precompute_frame)
            kfold = StratifiedKFold(random_state=random_seed, n_splits=5, shuffle=True)
            train_idx, test_idx = list(kfold.split(list(range(len(smiles))), ts.flatten()))[cvs]
        else:
            kfold = KFold(random_state=random_seed, n_splits=5, shuffle=True)
            train_idx, test_idx = list(kfold.split(list(range(len(smiles)))))[cvs]
        train_smiles = [smiles[i] for i in train_idx]
        test_smiles = [smiles[i] for i in test_idx]
    else:
        if infer:
            train_idx=[]
            train_smiles=[]
            test_idx=list(range(len(smiles)))
            test_smiles=smiles
        else:
            train_idx, test_idx, train_smiles, test_smiles = train_test_split(list(range(len(smiles))), smiles,
                                                                          test_size=0.2, random_state=random_seed,
                                                                          shuffle=True)
        print(len(train_idx), len(test_idx))
    if mask is not None:
        print("using mask")
        mask = np.load(mask)
        train_mask = mask[train_idx]
        test_mask = mask[test_idx]
        mask = True
    else:
        mask = False
    if precomputed_images is not None:
        precomputed_images = np.load(precomputed_images)
        train_images = precomputed_images[train_idx]
        test_images = precomputed_images[test_idx]
        precomputed_images = True
    else:
        precomputed_images = False

    scaler=None    
    if precompute_frame is not None:
        features = np.load(precompute_frame)
        #features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        assert (features.shape[0] == len(smiles))
        if scale is not None:
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features.reshape(-1,1))
            print("Scaled dock scores")
            pickle.dump(scaler, open(scale, 'wb'))
        train_features = features[train_idx]
        test_features = features[test_idx]

        if rotate:
            rotate = 359
        else:
            rotate = 0
        rotate = 359 if (ensembl and eval) else rotate
        train_dataset = ImageDatasetPreLoaded(train_smiles, train_features, imputer_pickle,
                                               property_func=get_properety_function(pname),
                                               values=tasks, rot=rotate, bw=bw,
                                               images=None if not precomputed_images else train_images,
                                               mask=None if not mask else train_mask)
        train_loader = DataLoader(train_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size,
                                  shuffle=(not eval))

        test_dataset = ImageDatasetPreLoaded(test_smiles, test_features, imputer_pickle,
                                             property_func=get_properety_function(pname),
                                             values=tasks, rot=rotate,
                                             images=None if not precomputed_images else test_images, bw=bw,
                                             mask=None if not mask else test_mask)
        test_loader = DataLoader(test_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size,
                                 shuffle=False)
    else:
        if not infer:
            assert(False)
        train_dataset = None
        train_loader = None
        test_dataset = ImageDatasetPreLoaded(test_smiles, None, imputer_pickle,
                                             property_func=get_properety_function(pname),
                                             values=tasks, rot=rotate,
                                             images=None if not precomputed_images else test_images, bw=bw,
                                             mask=None if not mask else test_mask)
        test_loader = DataLoader(test_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size,
                                 shuffle=False)
        if scale is not None:
            scaler = pickle.load(open(scale, 'rb'))

    if intermediate_rep is None:
        model = imagemodel.ImageModel(nheads=nheads, outs=tasks, classifacation=classifacation, dr=dropout, linear_layers=depth, pretrain=pretrain)
    else:
        model = imagemodel.ImageModel(nheads=nheads, outs=tasks, classifacation=classifacation, dr=dropout,
                                      intermediate_rep=intermediate_rep, linear_layers=depth, pretrain=pretrain)

    if return_datasets:
        return train_dataset, test_dataset, model, scaler
    else:
        return train_loader, test_loader, model, scaler


if __name__ == '__main__':
    args = get_args()

    np.random.seed(args.r)
    torch.manual_seed(args.r)

    train_loader, test_loader, model, scaler = load_data_models(args.i, args.r, args.w, args.b, args.p, nheads=args.nheads,
                                                        precompute_frame=args.precomputed_values,
                                                        imputer_pickle=args.imputer_pickle, eval=args.eval_train or args.eval_test,
                                                        tasks=args.t, rotate=args.rotate,
                                                        classifacation=args.classifacation, ensembl=args.ensemble_eval,
                                                        dropout=args.dropout_rate, cvs=args.cv,
                                                        intermediate_rep=args.width,
                                                        precomputed_images=args.precomputed_images, depth=args.depth,
                                                                bw=args.bw, mask=args.mask, pretrain=(not args.no_pretrain),
                                                                scale=args.scale, infer=args.infer)
    print("Done.")

    if args.eval_train:
        model.load_state_dict(torch.load(args.o)['model_state'])
        model.to(device)
        run_eval(model, train_loader, ordinal=False, enseml=args.ensemble_eval, output_preds=args.output_preds, scaler=scaler)
        exit()
    elif args.eval_test:
        model.load_state_dict(torch.load(args.o)['model_state'])
        model.to(device)
        run_eval(model, test_loader, ordinal=False, enseml=args.ensemble_eval, output_preds=args.output_preds, scaler=scaler)
        exit()
    elif args.infer:
        model.load_state_dict(torch.load(args.o)['model_state'])
        model.to(device)
        run_infer(model, test_loader, output_preds=args.output_preds, scaler=scaler)
        exit()
        
    model.to(device)
    optimizer = args.optimizer(model.parameters(), lr=args.lr)
    opt_level = args.amp
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    print("Starting trainer.")
    print("Number of parameters:",
          sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    model, history = trainer(model, optimizer, train_loader, test_loader, out=args.o, epochs=args.epochs, pb=args.pb,
                             classifacation=args.classifacation, tasks=args.t, mae=args.mae,
                             cyclic=args.cyclic, use_mask=args.mask is not None)
    history.plot_loss(save_file=args.metric_plot_prefix + "loss.png", title=args.p + " Loss")
    history.plot_metric(save_file=args.metric_plot_prefix + "r2.png", title=args.p + " " + history.metric_name)
    print("Finished training, now")
