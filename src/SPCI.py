import sys
import importlib as ipb
import pandas as pd
import numpy as np
import math
import time as time
import src.utils_SPCI as utils
import warnings
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pdb
import torch.nn as nn
from sklearn_quantile import RandomForestQuantileRegressor, SampleRandomForestQuantileRegressor
from numpy.lib.stride_tricks import sliding_window_view
from skranger.ensemble import RangerForestRegressor
from src.TemporalFusionTransformer import *
from src.QuantileTransformer import *
from src.QuantileGPT import *
from src.utils import *
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import copy
from sklearn.preprocessing import StandardScaler
device = torch.device("cpu")

#### Main Class ####


def detach_torch(input):
    return input.cpu().detach().numpy()


class SPCI_and_EnbPI():
    '''
        Create prediction intervals assuming Y_t = f(X_t) + \sigma(X_t)\eps_t
        Currently, assume the regression function is by default MLP implemented with PyTorch, as it needs to estimate BOTH f(X_t) and \sigma(X_t), where the latter is impossible to estimate using scikit-learn modules

        Most things carry out, except that we need to have different estimators for f and \sigma.

        fit_func = None: use MLP above
    '''

    def __init__(self, X_train, X_predict, Y_train, Y_predict, fit_func=None):
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.X_all = torch.cat([X_train, X_predict])
        self.Y_train = Y_train
        self.Y_predict = Y_predict
        # Predicted training data centers by EnbPI
        n, n1 = len(self.X_train), len(self.X_predict)
        self.Ensemble_train_interval_centers = np.ones(n)*np.inf
        self.Ensemble_train_interval_sigma = np.ones(n)*np.inf
        # Predicted test data centers by EnbPI
        self.Ensemble_pred_interval_centers = np.ones(n1)*np.inf
        self.Ensemble_pred_interval_sigma = np.ones(n1)*np.inf
        self.Ensemble_online_resid = np.ones(n+n1)*np.inf  # LOO scores
        self.beta_hat_bins = []
        ### New ones under development ###
        self.use_NeuralProphet = False
        #### Other hyperparameters for training (mostly simulation) ####
        # point predictor \hat f
        self.use_WLS = True # Whether to use WLS for fitting (compare with Nex-CP)
        self.WLS_c = 0.99
        # QRF training & how it treats the samples
        self.weigh_residuals = False # Whether we weigh current residuals more.
        self.c = 0.995 # If self.weight_residuals, weights[s] = self.c ** s, s\geq 0
        self.n_estimators = 10 # Num trees for QRF
        self.max_d = 2 # Max depth for fitting QRF
        self.criterion = 'mse' # 'mse' or 'mae'
        # search of \beta^* \in [0,\alpha]
        self.bins = 5 # break [0,\alpha] into bins
        # how many LOO training residuals to use for training current QRF 
        self.T1 = None # None = use all
        self.standardizer = None
    
    def split_validation_test(self, validation_ratio):
        
        print("splitting predict set into validation and test set...")
        validation_size = int(len(self.X_predict) * validation_ratio)
        self.X_validation = self.X_predict[:validation_size]
        self.X_test = self.X_predict[validation_size:]

        self.Y_validation = self.Y_predict[:validation_size]
        self.Y_test = self.Y_predict[validation_size:]

    def one_boot_prediction(self, Xboot, Yboot, Xfull):
        if self.use_NeuralProphet:
            '''
                Added NeuralPropeht in
                Note, hyperparameters not tuned yet
            '''
            Xboot, Yboot = detach_torch(Xboot), detach_torch(Yboot)
            nlags = 1
            model = NeuralProphet(
                n_forecasts=1,
                n_lags=nlags,
            )
            df_tmp, _ = utils.make_NP_df(Xboot, Yboot)
            model = model.add_lagged_regressor(names=self.Xnames)
            _ = model.fit(df_tmp, freq="D")  # Also outputs the metrics
            boot_pred = model.predict(self.df_full)['yhat1'].to_numpy()
            boot_pred[:nlags] = self.Y_train[:nlags]
            boot_pred = boot_pred.astype(np.float)
            boot_fX_pred = torch.from_numpy(boot_pred.flatten()).to(device)
            boot_sigma_pred = 0
        else:
            if self.regressor.__class__.__name__ == 'NoneType':
                start1 = time.time()
                model_f = MLP(self.d).to(device)
                optimizer_f = torch.optim.Adam(
                    model_f.parameters(), lr=1e-3)
                if self.fit_sigmaX:
                    model_sigma = MLP(self.d, sigma=True).to(device)
                    optimizer_sigma = torch.optim.Adam(
                        model_sigma.parameters(), lr=2e-3)
                for epoch in range(300):
                    fXhat = model_f(Xboot)
                    sigmaXhat = torch.ones(len(fXhat)).to(device)
                    if self.fit_sigmaX:
                        sigmaXhat = model_sigma(Xboot)
                    loss = ((Yboot - fXhat)
                            / sigmaXhat).pow(2).mean() / 2
                    optimizer_f.zero_grad()
                    if self.fit_sigmaX:
                        optimizer_sigma.zero_grad()
                    loss.backward()
                    optimizer_f.step()
                    if self.fit_sigmaX:
                        optimizer_sigma.step()
                with torch.no_grad():
                    boot_fX_pred = model_f(
                        Xfull).flatten().cpu().detach().numpy()
                    boot_sigma_pred = 0
                    if self.fit_sigmaX:
                        boot_sigma_pred = model_sigma(
                            Xfull).flatten().cpu().detach().numpy()
                print(
                    f'Took {time.time()-start1} secs to finish the {self.b}th boostrap model')
            else:
                Xboot, Yboot = detach_torch(Xboot), detach_torch(Yboot)
                Xfull = detach_torch(Xfull)
                # NOTE, NO sigma estimation because these methods by deFAULT are fitting Y, but we have no observation of errors
                model = self.regressor
                if self.use_WLS and isinstance(model,LinearRegression):
                    # To compare with Nex-CP when using WLS
                    # Taken from Nex-CP code
                    n = len(Xboot)
                    tags=self.WLS_c**(np.arange(n,0,-1))
                    model.fit(Xboot, Yboot, sample_weight=tags)
                else:
                    model.fit(Xboot, Yboot)
                boot_fX_pred = torch.from_numpy(
                    model.predict(Xfull).flatten()).to(device)
                boot_sigma_pred = 0
            return boot_fX_pred, boot_sigma_pred

    def fit_bootstrap_models_online_multistep(self, B, fit_sigmaX=True, stride=1):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, and compute the residuals
          fit_sigmaX: If False, just avoid predicting \sigma(X_t) by defaulting it to 1

          stride: int. If > 1, then we perform multi-step prediction, where we have to fit stride*B boostrap predictors.
            Idea: train on (X_i,Y_i), i=1,...,n-stride
            Then predict on X_1,X_{1+s},...,X_{1+k*s} where 1+k*s <= n+n1
            Note, when doing LOO prediction thus only care above the points above
        '''
        n, self.d = self.X_train.shape
        self.fit_sigmaX = fit_sigmaX
        n1 = len(self.X_predict)
        N = n-stride+1  # Total training data each one-step predictor sees
        # We make prediction every s step ahead, so these are feature the model sees
        train_pred_idx = np.arange(0, n, stride)
        # We make prediction every s step ahead, so these are feature the model sees
        test_pred_idx = np.arange(n, n+n1, stride)
        self.train_idx = train_pred_idx
        self.test_idx = test_pred_idx
        # Only contains features that are observed every stride steps
        Xfull = torch.vstack(
            [self.X_train[train_pred_idx], self.X_predict[test_pred_idx-n]])
        nsub, n1sub = len(train_pred_idx), len(test_pred_idx)
        for s in range(stride):
            ''' 1. Create containers for predictions '''
            # hold indices of training data for each f^b
            boot_samples_idx = utils.generate_bootstrap_samples(N, N, B)
            # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
            in_boot_sample = np.zeros((B, N), dtype=bool)
            # hold predictions from each f^b for fX and sigma&b for sigma
            boot_predictionsFX = np.zeros((B, nsub+n1sub))
            boot_predictionsSigmaX = np.ones((B, nsub+n1sub))
            # We actually would just use n1sub rows, as we only observe this number of features
            out_sample_predictFX = np.zeros((n, n1sub))
            out_sample_predictSigmaX = np.ones((n, n1sub))

            ''' 2. Start bootstrap prediction '''
            start = time.time()
            if self.use_NeuralProphet:
                self.df_full, self.Xnames = utils.make_NP_df(
                    Xfull, np.zeros(n + n1))
            for b in range(B):
                self.b = b
                Xboot, Yboot = self.X_train[boot_samples_idx[b],
                                            :], self.Y_train[s:s+N][boot_samples_idx[b], ]
                in_boot_sample[b, boot_samples_idx[b]] = True
                boot_fX_pred, boot_sigma_pred = self.one_boot_prediction(
                    Xboot, Yboot, Xfull)
                boot_predictionsFX[b] = boot_fX_pred
                if self.fit_sigmaX:
                    boot_predictionsSigmaX[b] = boot_sigma_pred
            print(f'{s+1}/{stride} multi-step: finish Fitting {B} Bootstrap models, took {time.time()-start} secs.')

            ''' 3. Obtain LOO residuals (train and test) and prediction for test data '''
            start = time.time()
            # Consider LOO, but here ONLY for the indices being predicted
            for j, i in enumerate(train_pred_idx):
                # j: counter and i: actual index X_{0+j*stride}
                if i < N:
                    b_keep = np.argwhere(
                        ~(in_boot_sample[:, i])).reshape(-1)
                    if len(b_keep) == 0:
                        # All bootstrap estimators are trained on this model
                        b_keep = 0  # More rigorously, it should be None, but in practice, the difference is minor
                else:
                    # This feature is not used in training, but used in prediction
                    b_keep = range(B)
                pred_iFX = boot_predictionsFX[b_keep, j].mean()
                pred_iSigmaX = boot_predictionsSigmaX[b_keep, j].mean()
                pred_testFX = boot_predictionsFX[b_keep, nsub:].mean(0)
                pred_testSigmaX = boot_predictionsSigmaX[b_keep, nsub:].mean(0)
                # Populate the training prediction
                # We add s because of multi-step procedure, so f(X_t) is for Y_t+s
                true_idx = min(i+s, n-1)
                self.Ensemble_train_interval_centers[true_idx] = pred_iFX
                self.Ensemble_train_interval_sigma[true_idx] = pred_iSigmaX
                resid_LOO = (detach_torch(
                    self.Y_train[true_idx]) - pred_iFX) / pred_iSigmaX
                out_sample_predictFX[i] = pred_testFX
                out_sample_predictSigmaX[i] = pred_testSigmaX
                self.Ensemble_online_resid[true_idx] = resid_LOO.item()
            sorted_out_sample_predictFX = out_sample_predictFX[train_pred_idx].mean(
                0)  # length ceil(n1/stride)
            sorted_out_sample_predictSigmaX = out_sample_predictSigmaX[train_pred_idx].mean(
                0)  # length ceil(n1/stride)
            pred_idx = np.minimum(test_pred_idx-n+s, n1-1)
            self.Ensemble_pred_interval_centers[pred_idx] = sorted_out_sample_predictFX
            self.Ensemble_pred_interval_sigma[pred_idx] = sorted_out_sample_predictSigmaX
            pred_full_idx = np.minimum(test_pred_idx+s, n+n1-1)
            resid_out_sample = (
                detach_torch(self.Y_predict[pred_idx]) - sorted_out_sample_predictFX) / sorted_out_sample_predictSigmaX
            self.Ensemble_online_resid[pred_full_idx] = resid_out_sample
        # Sanity check
        num_inf = (self.Ensemble_online_resid == np.inf).sum()
        if num_inf > 0:
            print(
                f'Something can be wrong, as {num_inf}/{n+n1} residuals are not all computed')
            print(np.where(self.Ensemble_online_resid == np.inf))

    def set_TFT_hyperparams(self, conf_dict, data_conf_dict):
        
        print("setting hyperparameters for TFT...")
        conf_dict["data_props"] = data_conf_dict
        self.configuration = conf_dict
        print("setting hyperparameters for TFT done")

    def set_standardizer(self):

        print("setting and fitting standardizer...")
        self.standardizer = StandardScaler()
        self.standardizer.fit(self.X_all.numpy())
        print("setting and fitting standardizer done")

    def compute_online_QT_SPCI_single_training(self, saving_dir, stride, past_window, num_prediction_step, target_quantiles,
                                               dim_model, num_head, dim_ff, num_encoder_layers, num_decoder_layers, dropout,
                                               max_epoch, additional_training_epoch, batch_size, learning_rate, early_stop, 
                                               standardize=False):

        if standardize:
            if self.standardizer != None:
                print("standardizing data...")
                self.X_train_standardized = torch.Tensor(self.standardizer.transform(self.X_train))
                self.X_validation_standardized = torch.Tensor(self.standardizer.transform(self.X_validation))
                self.X_test_standardized = torch.Tensor(self.standardizer.transform(self.X_test))
                self.X_all_standardized = torch.cat([self.X_train_standardized, self.X_validation_standardized, self.X_test_standardized])
                print("standardizing data done")
            else:
                print("standardizer must be fitted first!")
                
        n1 = len(self.X_train_standardized)
        s = stride
        feature_dim = self.X_all_standardized.shape[-1]
        # smallT always false on the implementation of SPCI

        train_size = len(self.X_train_standardized)
        validation_size = len(self.X_validation_standardized)
        test_size = len(self.X_test_standardized)

        resid_training = self.Ensemble_online_resid[:train_size]
        resid_input_train, resid_prediction_input_train, resid_prediction_train = build_strided_residue_QuantileTransformer(resid_training, 
                                                                                                                            past_window, 
                                                                                                                            num_prediction_step)
        feature_input_train, feature_prediction_input_train = build_strided_feature_QuantileTransformer(self.X_train_standardized, past_window, 
                                                                                                        num_prediction_step)
        
        resid_validation = self.Ensemble_online_resid[train_size-(past_window+num_prediction_step-1):train_size+validation_size]
        feature_validation = self.X_all_standardized[train_size-(past_window+num_prediction_step-1):train_size+validation_size]
        resid_input_validation, resid_prediction_input_validation, resid_prediction_validation = build_strided_residue_QuantileTransformer(resid_validation, 
                                                                                                                            past_window, 
                                                                                                                            num_prediction_step)
        feature_input_validation, feature_prediction_input_validation = build_strided_feature_QuantileTransformer(feature_validation, past_window, 
                                                                                                        num_prediction_step)
        
        resid_test = self.Ensemble_online_resid[train_size+validation_size-(past_window+num_prediction_step-1):]
        feature_test = self.X_all_standardized[train_size+validation_size-(past_window+num_prediction_step-1):]

        resid_input_test, resid_prediction_input_test, resid_prediction_test = build_strided_residue_QuantileTransformer(resid_test, past_window,
                                                                                                                         num_prediction_step)
                                                                                                                            
        feature_input_test, feature_prediction_input_test = build_strided_feature_QuantileTransformer(feature_test, past_window, 
                                                                                                      num_prediction_step)
        print("setting up dataset...")
        train_dataset = DatasetQuantileTransformer(resid_input_train, feature_input_train, resid_prediction_train, 
                                     resid_prediction_input_train, feature_prediction_input_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                         collate_fn=collate_fn_QuantileTransformer, drop_last=True)
        validation_dataset = DatasetQuantileTransformer(resid_input_validation, feature_input_validation, resid_prediction_validation, 
                                     resid_prediction_input_validation, feature_prediction_input_validation)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                         collate_fn=collate_fn_QuantileTransformer, drop_last=True)
        test_dataset = DatasetQuantileTransformer(resid_input_test, feature_input_test, resid_prediction_test, 
                                     resid_prediction_input_test, feature_prediction_input_test)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                         collate_fn=collate_fn_QuantileTransformer, drop_last=False)
        print("setting up dataset done")
        
        print("initialize models...")
        QT_model = QuantileTransformer(feature_dim+1, dim_model, num_head, dim_ff,
                                       num_encoder_layers, num_decoder_layers, target_quantiles, dropout)
        Optimizer = torch.optim.Adam(QT_model.parameters(), lr=learning_rate)
        print("initialize models done")
        
        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        best_loss = np.inf
        best_epoch = 0
        
        for e in range(max_epoch):

            if early_stop:
                if (e+1-best_epoch) >= early_stop:
                    # if the loss did not decrease for 5 epoch in a row, stop training
                    break

            batch_loss_sum = 0.
            batch_num = len(train_dataloader)

            QT_model.train()
            for x_batch, y_batch, resid_y_batch in tqdm(train_dataloader):

                Optimizer.zero_grad()
                #tgt_padding_mask = (resid_y_batch == pad_value) not required without padding
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(num_prediction_step) # causal mask
                # src_mask and src_key_padding_mask are None since we fix input sequence with length of past_window 
                ## update to have src_key_padding_mask is not None >> we can utilize when seq_len < past_window
                output_batch = QT_model(x_batch, y_batch, src_mask=None, tgt_mask=tgt_mask,
                                        src_key_padding_mask=None, tgt_key_padding_mask=None,
                                       memory_key_padding_mask=None) # batch_size * num_prediction_step * target_quantiles
                
                loss_batch = compute_quantile_loss(output_batch, resid_y_batch, torch.Tensor(target_quantiles))
                loss_batch.backward()
                Optimizer.step()
                batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            training_loss_per_epoch.append(epoch_loss)
            print("training loss at epoch {}: {}".format(e+1, epoch_loss))

            QT_model.eval()
            batch_loss_sum = 0.
            batch_num = len(validation_dataloader)
            for x_batch, y_batch, resid_y_batch in tqdm(validation_dataloader):

                with torch.no_grad():

                    #tgt_padding_mask = (resid_y_batch == pad_value) not required without padding
                    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(num_prediction_step) # causal mask
                    # src_mask and src_key_padding_mask are None since we fix input sequence with length of past_window 
                    ## update to have src_key_padding_mask is not None >> we can utilize when seq_len < past_window
                    output_batch = QT_model(x_batch, y_batch, src_mask=None, tgt_mask=tgt_mask,
                                            src_key_padding_mask=None, tgt_key_padding_mask=None,
                                            memory_key_padding_mask=None) # batch_size * num_prediction_step * target_quantiles
                    
                    loss_batch = compute_quantile_loss(output_batch, resid_y_batch, torch.Tensor(target_quantiles))
                    batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            validation_loss_per_epoch.append(epoch_loss)
            print("validation loss at epoch {}: {}".format(e+1, epoch_loss))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = e+1
                best_model = QT_model.state_dict()
        
        print("best model at epoch {}".format(best_epoch))
        print("start additional training on validation dataset...")
        QT_model.load_state_dict(best_model)

        # additional training with validation dataset
        for e in range(additional_training_epoch):

            batch_loss_sum = 0.
            batch_num = len(validation_dataloader)

            QT_model.train()
            for x_batch, y_batch, resid_y_batch in tqdm(validation_dataloader):

                Optimizer.zero_grad()
                #tgt_padding_mask = (resid_y_batch == pad_value) not required without padding
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(num_prediction_step) # causal mask
                # src_mask and src_key_padding_mask are None since we fix input sequence with length of past_window 
                ## update to have src_key_padding_mask is not None >> we can utilize when seq_len < past_window
                output_batch = QT_model(x_batch, y_batch, src_mask=None, tgt_mask=tgt_mask,
                                        src_key_padding_mask=None, tgt_key_padding_mask=None,
                                       memory_key_padding_mask=None) # batch_size * num_prediction_step * target_quantiles
                
                loss_batch = compute_quantile_loss(output_batch, resid_y_batch, torch.Tensor(target_quantiles))
                loss_batch.backward()
                Optimizer.step()
                batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            training_loss_per_epoch.append(epoch_loss)
            print("training loss of additional training at epoch {}: {}".format(e+1, epoch_loss))

        print("start evaluation on test dataset...")
        output_list = []
        QT_model.eval()
        for x_batch, y_batch, resid_y_batch in tqdm(test_dataloader):
            # shuffle is off so data comes in order

            with torch.no_grad():
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(num_prediction_step) # causal mask
                output_batch = QT_model(x_batch, y_batch, src_mask=None, tgt_mask=tgt_mask, src_key_padding_mask=None,
                                        tgt_key_padding_mask=None, 
                                        memory_key_padding_mask=None) # batch_size * num_prediction_step * target_quantiles
                
                output_list.append(output_batch.numpy())
        
        saving = saving_dir + "QT_SPCI_w{}_p{}_results.pkl".format(past_window, num_prediction_step)
        print("saving the results...")
        save_data(saving, output_list)

    def compute_onlinte_QGPT_SPCI_single_training(self, saving_dir, stride, past_window, target_quantiles,
                                               dim_model, num_head, dim_ff, num_layers, dropout, max_epoch, 
                                               additional_training_epoch, batch_size, learning_rate, early_stop, 
                                               standardize=False):

        hyperparams = {"stride" : stride, "past_window" : past_window, "target_quantiles" : target_quantiles,
                       "dim_model" : dim_model, "num_head" : num_head, "dim_ff" : dim_ff, "num_layers" : num_layers,
                       "dropout" : dropout, "max_epoch" : max_epoch, "additional_training_epoch" : additional_training_epoch,
                       "batch_size" : batch_size, "learning_rate" : learning_rate, "early_stop" : early_stop, "standardize": standardize}

        if standardize:
            if self.standardizer != None:
                print("standardizing data...")
                self.X_train_standardized = torch.Tensor(self.standardizer.transform(self.X_train))
                self.X_validation_standardized = torch.Tensor(self.standardizer.transform(self.X_validation))
                self.X_test_standardized = torch.Tensor(self.standardizer.transform(self.X_test))
                self.X_all_standardized = torch.cat([self.X_train_standardized, self.X_validation_standardized, self.X_test_standardized])
                print("standardizing data done")
            else:
                print("standardizer must be fitted first!")
        else:
            self.X_train_standardized = self.X_train
            self.X_validation_standardized = self.X_validation
            self.X_test_standardized = self.X_test
            self.X_all_standardized = torch.cat([self.X_train_standardized, self.X_validation_standardized, self.X_test_standardized])
                
        n1 = len(self.X_train_standardized)
        s = stride
        feature_dim = self.X_all_standardized.shape[-1]
        # smallT always false on the implementation of SPCI

        out_sample_predict = self.Ensemble_pred_interval_centers
        out_sample_predictSigmaX = self.Ensemble_pred_interval_sigma

        train_size = len(self.X_train_standardized)
        validation_size = len(self.X_validation_standardized)
        test_size = len(self.X_test_standardized)

        resid_training = self.Ensemble_online_resid[:len(self.X_train_standardized)]
        resid_input_train, resid_prediction_train = build_strided_residue_QuantileGPT(resid_training, past_window)
        feature_input_train = build_strided_feature_QuantileGPT(self.X_train_standardized, past_window)
        
        resid_validation = self.Ensemble_online_resid[train_size-past_window:train_size+validation_size]
        feature_validation = self.X_all_standardized[train_size-past_window:train_size+validation_size]
        resid_input_validation, resid_prediction_validation = build_strided_residue_QuantileGPT(resid_validation, past_window)
        feature_input_validation = build_strided_feature_QuantileGPT(feature_validation, past_window)

        resid_test = self.Ensemble_online_resid[train_size+validation_size-past_window:]
        feature_test = self.X_all_standardized[train_size+validation_size-past_window:]

        resid_input_test, resid_prediction_test = build_strided_residue_QuantileGPT(resid_test, past_window)
        feature_input_test = build_strided_feature_QuantileGPT(feature_test, past_window)

        print("setting up dataset...")
        train_dataset = DatasetQuantileGPT(resid_input_train, feature_input_train, resid_prediction_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_QuantileGPT, drop_last=True)
        validation_dataset = DatasetQuantileGPT(resid_input_validation, feature_input_validation, resid_prediction_validation)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_QuantileGPT, drop_last=True)
        test_dataset = DatasetQuantileGPT(resid_input_test, feature_input_test, resid_prediction_test)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_QuantileGPT, drop_last=False)
        print("setting up dataset done") 
        
        print("initialize models...")
        QGPT_model = QuantileGPT(feature_dim+1, dim_model, num_head, dim_ff, num_layers, target_quantiles, dropout=dropout)
        Optimizer = torch.optim.Adam(QGPT_model.parameters(), lr=learning_rate)
        print("initialize models done")
        
        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        best_loss = np.inf
        best_epoch = 0

        for e in range(max_epoch):

            if early_stop:
                if (e+1-best_epoch) >= early_stop:
                    # if the loss did not decrease for 5 epoch in a row, stop training
                    break

            batch_loss_sum = 0.
            batch_num = len(train_dataloader)

            QGPT_model.train()
            for x_batch, y_batch in tqdm(train_dataloader):

                Optimizer.zero_grad()
                attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(past_window) # causal mask
                # src_key_padding_mask is None since we fix input sequence with length of past_window 
                # TODO: update to have src_key_padding_mask is not None >> we can utilize when seq_len < past_window
                output_batch = QGPT_model(x_batch, src_mask=attn_mask, src_key_padding_mask=None) # batch_size * past_window * target_quantiles
                loss_batch = compute_quantile_loss(output_batch, y_batch, torch.Tensor(target_quantiles))
                loss_batch.backward()
                Optimizer.step()
                batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            training_loss_per_epoch.append(epoch_loss)
            print("training loss at epoch {}: {}".format(e+1, epoch_loss))

            QGPT_model.eval()
            batch_loss_sum = 0.
            batch_num = len(validation_dataloader)
            for x_batch, y_batch in tqdm(validation_dataloader):

                with torch.no_grad():

                    attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(past_window) # causal mask
                    output_batch = QGPT_model(x_batch, src_mask=attn_mask, src_key_padding_mask=None) # batch_size * past_window * target_quantiles
                    loss_batch = compute_quantile_loss(output_batch, y_batch, torch.Tensor(target_quantiles))
                    batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            validation_loss_per_epoch.append(epoch_loss)
            print("validation loss at epoch {}: {}".format(e+1, epoch_loss))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = e+1
                best_model = QGPT_model.state_dict()
        
        print("best model at epoch {}".format(best_epoch))
        print("start additional training on validation dataset...")
        QGPT_model.load_state_dict(best_model)
        torch.save(QGPT_model.state_dict(), saving_dir+"QGPT_best_model.pt")

        # additional training with validation dataset
        if additional_training_epoch > 0:
            output_list_record = []
            for e in range(additional_training_epoch):

                batch_loss_sum = 0.
                batch_num = len(validation_dataloader)

                QGPT_model.train()
                for x_batch, y_batch in tqdm(validation_dataloader):

                    Optimizer.zero_grad()
                    attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(past_window) # causal mask
                    output_batch = QGPT_model(x_batch, src_mask=attn_mask, src_key_padding_mask=None) # batch_size * past_window * target_quantiles
                    loss_batch = compute_quantile_loss(output_batch, y_batch, torch.Tensor(target_quantiles))
                    loss_batch.backward()
                    Optimizer.step()
                    batch_loss_sum += loss_batch.item()

                epoch_loss = batch_loss_sum/batch_num
                print("training loss of additional training at epoch {}: {}".format(e+1, epoch_loss))

                print("start evaluation on test dataset...")
                output_list = []
                QGPT_model.eval()
                for x_batch, y_batch in tqdm(test_dataloader):
                    # shuffle is off so data comes in order

                    with torch.no_grad():
                    
                        attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(past_window) # causal mask
                        output_batch = QGPT_model(x_batch, src_mask=attn_mask, src_key_padding_mask=None) # batch_size * past_window * target_quantiles
                        output_list.append(output_batch.numpy())

                compute_coverage_QGPT(self.Ensemble_pred_interval_centers, self.Y_predict, output_list, past_window, target_quantiles)
                output_list_record.append(output_list)
                torch.save(QGPT_model.state_dict(), saving_dir+"QGPT_ft{}_model.pt".format(e+1))
            
            result_dict = {"output_list" : output_list_record, "pred_interval_centers" : self.Ensemble_pred_interval_centers,
                           "Y_predict" : self.Y_predict, "past_window" : past_window, "target_quantiles" : target_quantiles}
            
            saving_result_dict = saving_dir + "QGPT_SPCI_result_dict.pkl"
            saving_hyperparams_dict = saving_dir + "QGPT_SPCI_hyperparms_dict.pkl"
            save_data(saving_result_dict, result_dict)
            save_data(saving_hyperparams_dict, hyperparams)

        else:
            print("start the final evaluation on test dataset...")
            output_list = []
            QGPT_model.eval()
            for x_batch, y_batch in tqdm(test_dataloader):
                # shuffle is off so data comes in order

                with torch.no_grad():
                    
                    attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(past_window) # causal mask
                    output_batch = QGPT_model(x_batch, src_mask=attn_mask, src_key_padding_mask=None) # batch_size * past_window * target_quantiles
                    output_list.append(output_batch.numpy())

            compute_coverage_QGPT(self.Ensemble_pred_interval_centers, self.Y_predict, output_list, past_window, target_quantiles)

            result_dict = {"output_list" : output_list, "pred_interval_centers" : self.Ensemble_pred_interval_centers,
                           "Y_predict" : self.Y_predict, "past_window" : past_window, "target_quantiles" : target_quantiles}
            
            saving_result_dict = saving_dir + "QGPT_SPCI_result_dict.pkl"
            saving_hyperparams_dict = saving_dir + "QGPT_SPCI_hyperparms_dict.pkl"
            save_data(saving_result_dict, result_dict)
            save_data(saving_hyperparams_dict, hyperparams)

    def predict_QGPT_multistep(self, model_dir, hyperparams_dict_dir, result_dict_dir, multistep, standardize=True):
    
        if standardize:
            if self.standardizer != None:
                print("standardizing data...")
                self.X_train_standardized = torch.Tensor(self.standardizer.transform(self.X_train))
                self.X_validation_standardized = torch.Tensor(self.standardizer.transform(self.X_validation))
                self.X_test_standardized = torch.Tensor(self.standardizer.transform(self.X_test))
                self.X_all_standardized = torch.cat([self.X_train_standardized, self.X_validation_standardized, self.X_test_standardized])
                print("standardizing data done")
            else:
                print("standardizer must be fitted first!")
        else:
            self.X_train_standardized = self.X_train
            self.X_validation_standardized = self.X_validation
            self.X_test_standardized = self.X_test
            self.X_all_standardized = torch.cat([self.X_train_standardized, self.X_validation_standardized, self.X_test_standardized])

        # load model
        dim_feature = self.X_train.shape[-1] + 1
        result_dict = load_data(result_dict_dir)
        hyperparams_dict = load_data(hyperparams_dict_dir)
        QGPT_model = QuantileGPT(dim_feature, hyperparams_dict["dim_model"], 
                             hyperparams_dict["num_head"], hyperparams_dict["dim_ff"], hyperparams_dict["num_layers"], 
                             hyperparams_dict["target_quantiles"])
        QGPT_model.load_state_dict(torch.load(model_dir))

        train_size = len(self.X_train_standardized)
        validation_size = len(self.X_validation_standardized)
        test_size = len(self.X_test_standardized)
        prediction_interval_centers = result_dict["pred_interval_centers"]
        Y_predict = result_dict["Y_predict"]

        past_window = hyperparams_dict["past_window"]
        target_quantiles = hyperparams_dict["target_quantiles"]
        median_idx = int(len(target_quantiles)/2)

        resid_test = self.Ensemble_online_resid[train_size+validation_size-past_window:]
        feature_test = self.X_all_standardized[train_size+validation_size-past_window:]
        resid_input_test, resid_prediction_test = build_strided_residue_QuantileGPT(resid_test, past_window)

        resid_input_test_list = []
        resid_prediction_test_list = []
        feature_input_test_list = []

        for step in range(multistep):
            # feature_input_test_list[s] will be feature_input_test for s-th multistep prediction
            feature_input_test_list.append(copy.deepcopy(build_strided_feature_QuantileGPT(feature_test, past_window+step)))
            resid_input_test, resid_prediction_test = build_strided_residue_QuantileGPT(resid_test, past_window+step)
            resid_input_test_list.append(copy.deepcopy(resid_input_test))
            resid_prediction_test_list.append(copy.deepcopy(resid_prediction_test))
        
        QGPT_model.eval()
        for step in range(multistep):
            
            print("predicting {}-step ahead".format(step+1))

            print("setting up dataset...")
            test_dataset = DatasetQuantileGPT(resid_input_test_list[step], feature_input_test_list[step], resid_prediction_test_list[step])
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_QuantileGPT, drop_last=False)
            print("setting up dataset done") 

            output_list = []
            for x_batch, y_batch in tqdm(test_dataloader):
                # shuffle is off so data comes in order
                with torch.no_grad():
                    
                    attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(past_window+step) # causal mask
                    output_batch = QGPT_model(x_batch, src_mask=attn_mask, src_key_padding_mask=None) # batch_size * past_window * target_quantiles
                    output_list.append(output_batch.numpy())
            
            output_arr = np.array(output_list) # test_set_size-(step_ahead-1) * 1 * past_window+(step_ahead-1) * target_quantiles
            output_arr = np.squeeze(output_arr) # test_set_size-(step_ahead-1) * past_window+(step_ahead-1) * target_quantiles
            predicted_residuals = output_arr[:,-1,median_idx] # test_set_size-(step_ahead-1) * 1 * 1
            print(predicted_residuals.shape)

            if (step+1) < multistep:
                idx_count = 0
                for s in range((step+1), multistep):
                    idx_count += 1
                    # replace true residuals in multi-step range with predicted residuals
                    resid_input_test_list[s][:,-idx_count] = predicted_residuals[:-s+step]

            compute_coverage_QGPT(prediction_interval_centers[(step+1):], Y_predict[(step+1):], output_arr, past_window+step, target_quantiles)
        
        return feature_input_test_list, resid_input_test_list, resid_prediction_test_list


    def plot_QGPT_multistep(self, model_dir, saving_dir, hyperparams_dict_dir, result_dict_dir, multistep, ylim, standardize=True):
    
        if standardize:
            if self.standardizer != None:
                print("standardizing data...")
                self.X_train_standardized = torch.Tensor(self.standardizer.transform(self.X_train))
                self.X_validation_standardized = torch.Tensor(self.standardizer.transform(self.X_validation))
                self.X_test_standardized = torch.Tensor(self.standardizer.transform(self.X_test))
                self.X_all_standardized = torch.cat([self.X_train_standardized, self.X_validation_standardized, self.X_test_standardized])
                print("standardizing data done")
            else:
                print("standardizer must be fitted first!")
        else:
            self.X_train_standardized = self.X_train
            self.X_validation_standardized = self.X_validation
            self.X_test_standardized = self.X_test
            self.X_all_standardized = torch.cat([self.X_train_standardized, self.X_validation_standardized, self.X_test_standardized])

        # load model
        dim_feature = self.X_train.shape[-1] + 1
        result_dict = load_data(result_dict_dir)
        hyperparams_dict = load_data(hyperparams_dict_dir)
        QGPT_model = QuantileGPT(dim_feature, hyperparams_dict["dim_model"], 
                             hyperparams_dict["num_head"], hyperparams_dict["dim_ff"], hyperparams_dict["num_layers"], 
                             hyperparams_dict["target_quantiles"])
        QGPT_model.load_state_dict(torch.load(model_dir))

        train_size = len(self.X_train_standardized)
        validation_size = len(self.X_validation_standardized)
        test_size = len(self.X_test_standardized)
        prediction_interval_centers = result_dict["pred_interval_centers"]
        Y_predict = result_dict["Y_predict"]

        past_window = hyperparams_dict["past_window"]
        target_quantiles = hyperparams_dict["target_quantiles"]
        median_idx = int(len(target_quantiles)/2)

        resid_test = self.Ensemble_online_resid[train_size+validation_size-past_window:]
        feature_test = self.X_all_standardized[train_size+validation_size-past_window:]
        resid_input_test, resid_prediction_test = build_strided_residue_QuantileGPT(resid_test, past_window)

        resid_input_test_list = []
        resid_prediction_test_list = []
        feature_input_test_list = []

        for step in range(multistep):
            # feature_input_test_list[s] will be feature_input_test for s-th multistep prediction
            feature_input_test_list.append(copy.deepcopy(build_strided_feature_QuantileGPT(feature_test, past_window+step)))
            resid_input_test, resid_prediction_test = build_strided_residue_QuantileGPT(resid_test, past_window+step)
            resid_input_test_list.append(copy.deepcopy(resid_input_test))
            resid_prediction_test_list.append(copy.deepcopy(resid_prediction_test))
        
        QGPT_model.eval()
        for step in range(multistep):
            
            print("predicting {}-step ahead".format(step+1))

            print("setting up dataset...")
            test_dataset = DatasetQuantileGPT(resid_input_test_list[step], feature_input_test_list[step], resid_prediction_test_list[step])
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_QuantileGPT, drop_last=False)
            print("setting up dataset done") 

            output_list = []
            for x_batch, y_batch in tqdm(test_dataloader):
                # shuffle is off so data comes in order
                with torch.no_grad():
                    
                    attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(past_window+step) # causal mask
                    output_batch = QGPT_model(x_batch, src_mask=attn_mask, src_key_padding_mask=None) # batch_size * past_window * target_quantiles
                    output_list.append(output_batch.numpy())
            
            output_arr = np.array(output_list) # test_set_size-(step_ahead-1) * 1 * past_window+(step_ahead-1) * target_quantiles
            output_arr = np.squeeze(output_arr) # test_set_size-(step_ahead-1) * past_window+(step_ahead-1) * target_quantiles
            predicted_residuals = output_arr[:,-1,median_idx] # test_set_size-(step_ahead-1) * 1 * 1

            if (step+1) < multistep:
                idx_count = 0
                for s in range((step+1), multistep):
                    idx_count += 1
                    # replace true residuals in multi-step range with predicted residuals
                    resid_input_test_list[s][:,-idx_count] = predicted_residuals[:-s+step]

            plot_QGPT_results(saving_dir, prediction_interval_centers[(step+1):], Y_predict[(step+1):], output_arr, past_window+step, target_quantiles, ylim)
        
    def compute_online_TFT_SPCI_single_training(self, saving_dir, stride, past_window, num_prediction_step, max_epoch, additional_training_epoch, 
                                                batch_size, learning_rate, early_stop, standardize=False, pad_value=None):
        '''
        conduct SPCI using TFT as a quantile prediction method
        ''' 

        if standardize:
            if self.standardizer != None:
                print("standardizing data...")
                self.X_train_standardized = torch.Tensor(self.standardizer.transform(self.X_train))
                self.X_validation_standardized = torch.Tensor(self.standardizer.transform(self.X_validation))
                self.X_test_standardized = torch.Tensor(self.standardizer.transform(self.X_test))
                self.X_all_standardized = torch.cat([self.X_train_standardized, self.X_validation_standardized, self.X_test_standardized])
                print("standardizing data done")
            else:
                print("standardizer must be fitted first!")

        n1 = len(self.X_train_standardized)
        s = stride
        feature_dim = self.X_all_standardized.shape[-1]
        # smallT always false on the implementation of SPCI

        out_sample_predict = self.Ensemble_pred_interval_centers
        out_sample_predictSigmaX = self.Ensemble_pred_interval_sigma

        train_size = len(self.X_train_standardized)
        validation_size = len(self.X_validation_standardized)
        test_size = len(self.X_test_standardized)

        resid_training = self.Ensemble_online_resid[:len(self.X_train_standardized)]
        resid_input_train, resid_prediction_input_train, resid_prediction_train = build_strided_residue_QuantileTransformer(resid_training, 
                                                                                                                            past_window, 
                                                                                                                            num_prediction_step, 
                                                                                                                            pad=pad_value)
        feature_input_train, feature_prediction_input_train = build_strided_feature_QuantileTransformer(self.X_train_standardized, past_window, 
                                                                                                        num_prediction_step, pad=pad_value)
        
        resid_validation = self.Ensemble_online_resid[train_size-(past_window+num_prediction_step-1):train_size+validation_size]
        feature_validation = self.X_all_standardized[train_size-(past_window+num_prediction_step-1):train_size+validation_size]
        resid_input_validation, resid_prediction_input_validation, resid_prediction_validation = build_strided_residue_QuantileTransformer(resid_validation, 
                                                                                                                            past_window, 
                                                                                                                            num_prediction_step, 
                                                                                                                            pad=pad_value)
        feature_input_validation, feature_prediction_input_validation = build_strided_feature_QuantileTransformer(feature_validation, past_window, 
                                                                                                        num_prediction_step, pad=pad_value)
        
        resid_test = self.Ensemble_online_resid[train_size+validation_size-(past_window+num_prediction_step-1):train_size+validation_size+test_size]
        feature_test = self.X_all_standardized[train_size+validation_size-(past_window+num_prediction_step-1):train_size+validation_size+test_size]

        resid_input_test, resid_prediction_input_test, resid_prediction_test = build_strided_residue_QuantileTransformer(resid_test, 
                                                                                                                            past_window, 
                                                                                                                            num_prediction_step, 
                                                                                                                            pad=pad_value)
        feature_input_test, feature_prediction_input_test = build_strided_feature_QuantileTransformer(feature_test, past_window, 
                                                                                                        num_prediction_step, pad=pad_value)
        print("setting up dataset...")
        train_dataset = DatasetTFT(resid_input_train, feature_input_train, resid_prediction_train, resid_prediction_input_train, 
                                   feature_prediction_input_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_TFT, drop_last=True)
        validation_dataset = DatasetTFT(resid_input_validation, feature_input_validation, 
                                        resid_prediction_validation, resid_prediction_input_validation, feature_prediction_input_validation)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                         collate_fn=collate_fn_TFT, drop_last=True)
        test_dataset = DatasetTFT(resid_input_test, feature_input_test, resid_prediction_test, resid_prediction_input_test, 
                                  feature_prediction_input_test)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                         collate_fn=collate_fn_TFT, drop_last=False)
        print("setting up dataset done")
        
        print("initialize models...")
        TFT_model = TemporalFusionTransformer(OmegaConf.create(self.configuration))
        Optimizer = torch.optim.Adam(TFT_model.parameters(), lr=learning_rate)
        print("initialize models done")

        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        best_loss = np.inf
        best_epoch = 0
        
        for e in range(max_epoch):

            if early_stop:
                if (e+1-best_epoch) >= early_stop:
                    # if the loss did not decrease for 5 epoch in a row, stop training
                    break

            batch_loss_sum = 0.
            batch_num = len(train_dataloader)

            TFT_model.train()
            for input_batch, targets in tqdm(train_dataloader):
                    
                Optimizer.zero_grad()
                output_batch = TFT_model(input_batch)
                loss_batch, q_risk_batch, losses_array = get_quantiles_loss_and_q_risk(outputs=output_batch["predicted_quantiles"],
                                                                                       targets=targets,
                                        desired_quantiles=torch.Tensor(self.configuration["model"]["output_quantiles"]))
                loss_batch.backward()
                Optimizer.step()
                batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            training_loss_per_epoch.append(epoch_loss)
            print("training loss at epoch {}: {}".format(e+1, epoch_loss))

            TFT_model.eval()
            batch_loss_sum = 0.
            batch_num = len(validation_dataloader)
            for input_batch, targets in tqdm(validation_dataloader):

                with torch.no_grad():
                    output_batch = TFT_model(input_batch)
                    loss_batch, q_risk_batch, losses_array = get_quantiles_loss_and_q_risk(outputs=output_batch["predicted_quantiles"],
                                                                                           targets=targets,
                                                        desired_quantiles=torch.Tensor(self.configuration["model"]["output_quantiles"]))
                    batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            validation_loss_per_epoch.append(epoch_loss)
            print("validation loss at epoch {}: {}".format(e+1, epoch_loss))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = e+1
                best_model = TFT_model.state_dict()

        print("best model at epoch {}".format(best_epoch))
        print("start evaluation on validation dataset...")
        TFT_model.load_state_dict(best_model)

        # additional training with validation dataset
        for e in range(additional_training_epoch):

            batch_loss_sum = 0.
            batch_num = len(validation_dataloader)

            TFT_model.train()
            for input_batch, targets in tqdm(validation_dataloader):
                    
                Optimizer.zero_grad()
                output_batch = TFT_model(input_batch)
                loss_batch, q_risk_batch, losses_array = get_quantiles_loss_and_q_risk(outputs=output_batch["predicted_quantiles"],
                                                                                       targets=targets,
                                        desired_quantiles=torch.Tensor(self.configuration["model"]["output_quantiles"]))
                loss_batch.backward()
                Optimizer.step()
                batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            training_loss_per_epoch.append(epoch_loss)
            print("training loss of additional training at epoch {}: {}".format(e+1, epoch_loss))

        print("start evaluation on test dataset...")
        predicted_qtiles_list = []

        TFT_model.eval()
        for input_single, target in tqdm(test_dataloader):
            # shuffle is off so data comes in order
            with torch.no_grad():
                output_single = TFT_model(input_single)
                predicted_qtiles_list.append(output_single["predicted_quantiles"].numpy())

        saving = saving_dir + "TFT_SPCI_w{}_p{}_results.pkl".format(past_window, num_prediction_step)
        print("saving the results...")
        save_data(saving, predicted_qtiles_list)
    
    def compute_online_TFT_SPCI_single_training_(self, alpha, stride, past_window, max_epoch, batch_size, learning_rate, early_stop, standardize=False):
        '''
        conduct SPCI using TFT as a quantile prediction method
        '''

        if standardize:
            if self.standardizer != None:
                self.X_train = torch.Tensor(self.standardizer.transform(self.X_train))
                self.X_predict = torch.Tensor(self.standardizer.transform(self.X_predict))   
            else:
                print("standardizer must be fitted first!")

        self.alpha = alpha
        n1 = len(self.X_train)
        self.past_window = past_window
        s = stride
        beta_ls = np.linspace(start=0, stop=self.alpha, num=self.bins)
        full_alphas = np.append(beta_ls, 1 - alpha + beta_ls)
        # smallT always false on the implementation of SPCI

        out_sample_predict = self.Ensemble_pred_interval_centers
        out_sample_predictSigmaX = self.Ensemble_pred_interval_sigma

        resid_strided = utils.strided_app(self.Ensemble_online_resid[len(self.X_train)-n1:-1], n1, stride)
        features_strided = utils.get_features_strided(self.X_all.numpy()[len(self.X_train)-n1:-1,:], n1, stride) # sample_size * seq_len * feature_dim
        print("resid_strided shape : {}".format(resid_strided.shape))
        print("single feature_strided shape : {}".format(features_strided[:,:,0].shape)) # two must be the same shape

        num_unique_resid = resid_strided.shape[0] # the same number of X_pred
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)

        # using sequence models, multi-step prediction is possible without training multiple models
        # here only trained a single global model 
        i = 0
        curr_SigmaX = out_sample_predictSigmaX[i].item()
            
        past_resid = resid_strided[i, :]
        past_features = features_strided[i,:,:]
        print("past_resid shape : {}".format(past_resid.shape))
        print("past_features shape : {}".format(past_features.shape))

        n2 = self.past_window
        num = len(past_resid)
        #resid_pred = past_resid[-n2:] dont need here
        #features_pred = past_features[-n2:]
        #print("resid_pred shape: {}".format(resid_pred.shape))
        #print("features_pred shape: {}".format(features_pred.shape))
        residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
        featureX = sliding_window_view(past_features[:num-s+1,:], window_shape=n2, axis=0)
        print("residX shape : {}".format(residX.shape))
        print("featureX shape : {}".format(featureX.shape))
        residY = past_resid[n2:num-(s-1)]
        featureY = past_features[n2:num-(s-1), :]
        print("residY shape: {}".format(residY.shape))
        print("featureY shape: {}".format(featureY.shape))

        print("setting up dataset...")
        training_dataset = DatasetTFT(residX[:-1], featureX[:-1], residY, featureY)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                             collate_fn=collate_fn_TFT, drop_last=True)
        
        # building validation dataset
        validation_residX = []
        validation_featureX = []
        validation_residY = []
        validation_featureY = []

        for i in tqdm(range(num_unique_resid)):
            curr_SigmaX = out_sample_predictSigmaX[i].item()
            
            past_resid = resid_strided[i, :]
            past_features = features_strided[i,:,:]

            n2 = self.past_window
            num = len(past_resid)
            resid_pred = past_resid[-n2:]
            features_pred = past_features[-n2:]

            residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
            featureX = sliding_window_view(past_features[:num-s+1,:], window_shape=n2, axis=0)
            residY = past_resid[n2:num-(s-1)]
            featureY = past_features[n2:num-(s-1), :]

            validation_residX.append(residX[-1])
            validation_featureX.append(featureX[-1])
            validation_residY.append(residY[-1])
            validation_featureY.append(featureY[-1])

        validation_residX = np.array(validation_residX)
        validation_featureX = np.array(validation_featureX)
        validation_residY = np.array(validation_residY)
        validation_featureY = np.array(validation_featureY)

        validation_dataset = DatasetTFT(validation_residX, validation_featureX, validation_residY, validation_featureY)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                             collate_fn=collate_fn_TFT, drop_last=True)
        print("setting up dataset done")

        print("initialize models...")
        TFT_model = TemporalFusionTransformer(OmegaConf.create(self.configuration))
        Optimizer = torch.optim.Adam(TFT_model.parameters(), lr=learning_rate)
        print("initialize models done")

        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        best_loss = np.inf
        best_epoch = 0
        
        for e in range(max_epoch):

            if early_stop:
                if (e+1-best_epoch) >= early_stop:
                    # if the loss did not decrease for 5 epoch in a row, stop training
                    break

            batch_loss_sum = 0.
            batch_num = len(training_dataloader)

            TFT_model.train()
            for input_batch, targets in tqdm(training_dataloader):
                    
                Optimizer.zero_grad()
                output_batch = TFT_model(input_batch)
                loss_batch, q_risk_batch, losses_array = get_quantiles_loss_and_q_risk(outputs=output_batch["predicted_quantiles"],
                                                                                       targets=targets,
                                        desired_quantiles=torch.Tensor(self.configuration["model"]["output_quantiles"]))
                loss_batch.backward()
                Optimizer.step()
                batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            training_loss_per_epoch.append(epoch_loss)
            print("training loss at epoch {}: {}".format(e+1, epoch_loss))

            TFT_model.eval()
            batch_loss_sum = 0.
            batch_num = len(validation_dataloader)
            for input_batch, targets in tqdm(validation_dataloader):

                with torch.no_grad():
                    output_batch = TFT_model(input_batch)
                    loss_batch, q_risk_batch, losses_array = get_quantiles_loss_and_q_risk(outputs=output_batch["predicted_quantiles"],
                                                                                           targets=targets,
                                                        desired_quantiles=torch.Tensor(self.configuration["model"]["output_quantiles"]))
                    batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            validation_loss_per_epoch.append(epoch_loss)
            print("validation loss at epoch {}: {}".format(e+1, epoch_loss))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = e+1

        print("evaluating on {} samples".format(num_unique_resid))
        predicted_qtiles_list = []

        TFT_model.eval()
        for i in tqdm(range(num_unique_resid)):
            curr_SigmaX = out_sample_predictSigmaX[i].item()
            
            past_resid = resid_strided[i, :]
            past_features = features_strided[i,:,:]

            n2 = self.past_window
            num = len(past_resid)
            resid_pred = past_resid[-n2:]
            features_pred = past_features[-n2:]

            residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
            featureX = sliding_window_view(past_features[:num-s+1,:], window_shape=n2, axis=0)
            residY = past_resid[n2:num-(s-1)]
            featureY = past_features[n2:num-(s-1), :]
            pred_single_batch = make_batch_TFT(residX[-1], featureX[-1], featureY[-1])
            with torch.no_grad():
                pred_single_output = TFT_model(pred_single_batch)
            
            predicted_qtiles_list.append(pred_single_output["predicted_quantiles"].flatten().numpy())

        predicted_qtiles_arr = np.array(predicted_qtiles_list) # num_unique_resid * num_qtiles
        Ytest = copy.deepcopy(self.Y_predict.numpy())
        
        lower_PI = copy.deepcopy(self.Ensemble_pred_interval_centers) + predicted_qtiles_arr[:,0]
        upper_PI = copy.deepcopy(self.Ensemble_pred_interval_centers) + predicted_qtiles_arr[:,-1]

        avg_coverage = np.mean((lower_PI <= Ytest) & (upper_PI >= Ytest))
        avg_width = np.mean(upper_PI - lower_PI)

        print('avg coverage : {}'.format(avg_coverage))
        print('avg width : {}'.format(avg_width))

        loss_dict = {"training_loss_per_epoch" : training_loss_per_epoch, "validation_loss_per_epoch" : validation_loss_per_epoch}

        return predicted_qtiles_arr, Ytest, lower_PI, upper_PI, loss_dict

    def compute_online_TFT_SPCI_additional_training(self, alpha, stride, past_window, max_epoch, 
                                                    batch_size, learning_rate, additional_traning_freq, 
                                                    additional_training_epoch, early_stop, standardize=False):
        '''
        conduct SPCI using TFT as a quantile prediction method
        '''

        if standardize:
            if self.standardizer != None:
                self.X_train = torch.Tensor(self.standardizer.transform(self.X_train))
                self.X_predict = torch.Tensor(self.standardizer.transform(self.X_predict))   
            else:
                print("standardizer must be fitted first!")

        self.alpha = alpha
        n1 = len(self.X_train)
        self.past_window = past_window
        s = stride
        beta_ls = np.linspace(start=0, stop=self.alpha, num=self.bins)
        full_alphas = np.append(beta_ls, 1 - alpha + beta_ls)
        # smallT always false on the implementation of SPCI

        out_sample_predict = self.Ensemble_pred_interval_centers
        out_sample_predictSigmaX = self.Ensemble_pred_interval_sigma

        resid_strided = utils.strided_app(self.Ensemble_online_resid[len(self.X_train)-n1:-1], n1, stride)
        features_strided = utils.get_features_strided(self.X_all.numpy()[len(self.X_train)-n1:-1,:], n1, stride) # sample_size * seq_len * feature_dim
        print("resid_strided shape : {}".format(resid_strided.shape))
        print("single feature_strided shape : {}".format(features_strided[:,:,0].shape)) # two must be the same shape

        num_unique_resid = resid_strided.shape[0]
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)

        # using sequence models, multi-step prediction is possible without training multiple models
        # here only trained a single global model 
        i = 0
        curr_SigmaX = out_sample_predictSigmaX[i].item()
            
        past_resid = resid_strided[i, :]
        past_features = features_strided[i,:,:]
        print("past_resid shape : {}".format(past_resid.shape))
        print("past_features shape : {}".format(past_features.shape))

        n2 = self.past_window
        num = len(past_resid)
        resid_pred = past_resid[-n2:]
        features_pred = past_features[-n2:]
        print("resid_pred shape: {}".format(resid_pred.shape))
        print("features_pred shape: {}".format(features_pred.shape))
        residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
        featureX = sliding_window_view(past_features[:num-s+1,:], window_shape=n2, axis=0)
        print("residX shape : {}".format(residX.shape))
        print("featureX shape : {}".format(featureX.shape))
        residY = past_resid[n2:num-(s-1)]
        featureY = past_features[n2:num-(s-1), :]
        print("residY shape: {}".format(residY.shape))
        print("featureY shape: {}".format(featureY.shape))

        print("setting up dataset...")
        training_dataset = DatasetTFT(residX[:-1], featureX[:-1], residY, featureY)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                             collate_fn=collate_fn_TFT, drop_last=True)
        
        # building validation dataset
        validation_residX = []
        validation_featureX = []
        validation_residY = []
        validation_featureY = []

        for i in tqdm(range(num_unique_resid)):
            curr_SigmaX = out_sample_predictSigmaX[i].item()
            
            past_resid = resid_strided[i, :]
            past_features = features_strided[i,:,:]

            n2 = self.past_window
            num = len(past_resid)
            resid_pred = past_resid[-n2:]
            features_pred = past_features[-n2:]

            residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
            featureX = sliding_window_view(past_features[:num-s+1,:], window_shape=n2, axis=0)
            residY = past_resid[n2:num-(s-1)]
            featureY = past_features[n2:num-(s-1), :]

            validation_residX.append(residX[-1])
            validation_featureX.append(featureX[-1])
            validation_residY.append(residY[-1])
            validation_featureY.append(featureY[-1])

        validation_residX = np.array(validation_residX)
        validation_featureX = np.array(validation_featureX)
        validation_residY = np.array(validation_residY)
        validation_featureY = np.array(validation_featureY)

        validation_dataset = DatasetTFT(validation_residX, validation_featureX, validation_residY, validation_featureY)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                             collate_fn=collate_fn_TFT, drop_last=True)

        # creating splitted validation dataset list for sequential additional training
        split_size = int(np.floor(validation_residX.shape[0] / additional_traning_freq))
        print("split size : {}".format(split_size))
        validation_dataset_split_list = []

        for i in range(additional_traning_freq):

            temp_validation_dataset = DatasetTFT(validation_residX[i*split_size:(i+1)*split_size], 
                                                 validation_featureX[i*split_size:(i+1)*split_size], 
                                                 validation_residY[i*split_size:(i+1)*split_size], 
                                                 validation_featureY[i*split_size:(i+1)*split_size])
            print("dataset len : {}".format(len(temp_validation_dataset)))
            validation_dataset_split_list.append(temp_validation_dataset)
        print("setting up dataset done")

        print("initialize models...")
        TFT_model = TemporalFusionTransformer(OmegaConf.create(self.configuration))
        Optimizer = torch.optim.Adam(TFT_model.parameters(), lr=learning_rate)
        print("initialize models done")

        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        best_loss = np.inf
        best_epoch = 0
        
        for e in range(max_epoch):

            if early_stop:
                if (e+1-best_epoch) >= early_stop:
                    # if the loss did not decrease for 5 epoch in a row, stop training
                    break

            batch_loss_sum = 0.
            batch_num = len(training_dataloader)

            TFT_model.train()
            for input_batch, targets in tqdm(training_dataloader):
                    
                Optimizer.zero_grad()
                output_batch = TFT_model(input_batch)
                loss_batch, q_risk_batch, losses_array = get_quantiles_loss_and_q_risk(outputs=output_batch["predicted_quantiles"],
                                                                                       targets=targets,
                                        desired_quantiles=torch.Tensor(self.configuration["model"]["output_quantiles"]))
                loss_batch.backward()
                Optimizer.step()
                batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            training_loss_per_epoch.append(epoch_loss)
            print("training loss at epoch {}: {}".format(e+1, epoch_loss))

            TFT_model.eval()
            batch_loss_sum = 0.
            batch_num = len(validation_dataloader)
            for input_batch, targets in tqdm(validation_dataloader):

                with torch.no_grad():
                    output_batch = TFT_model(input_batch)
                    loss_batch, q_risk_batch, losses_array = get_quantiles_loss_and_q_risk(outputs=output_batch["predicted_quantiles"],
                                                                                           targets=targets,
                                                        desired_quantiles=torch.Tensor(self.configuration["model"]["output_quantiles"]))
                    batch_loss_sum += loss_batch.item()

            epoch_loss = batch_loss_sum/batch_num
            validation_loss_per_epoch.append(epoch_loss)
            print("validation loss at epoch {}: {}".format(e+1, epoch_loss))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = e+1

        print("evaluating on {} samples with additional training".format(num_unique_resid))
        predicted_qtiles_list = []

        for i in tqdm(range(num_unique_resid)):
            #curr_SigmaX = out_sample_predictSigmaX[i].item()
            
            past_resid = resid_strided[i, :]
            past_features = features_strided[i,:,:]

            n2 = self.past_window
            num = len(past_resid)
            resid_pred = past_resid[-n2:]
            features_pred = past_features[-n2:]

            residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
            featureX = sliding_window_view(past_features[:num-s+1,:], window_shape=n2, axis=0)
            residY = past_resid[n2:num-(s-1)]
            featureY = past_features[n2:num-(s-1), :]
            pred_single_batch = make_batch_TFT(residX[-1], featureX[-1], featureY[-1])

            TFT_model.eval()
            with torch.no_grad():
                pred_single_output = TFT_model(pred_single_batch)
            
            predicted_qtiles_list.append(pred_single_output["predicted_quantiles"].flatten().numpy())

            # additional training
            if (i+1) % split_size == 0:
                split_idx = ((i+1) // split_size)-1
                print("split_idx : {}".format(split_idx))
                additional_training_dataloader = DataLoader(validation_dataset_split_list[split_idx], batch_size=batch_size, shuffle=True, 
                                             collate_fn=collate_fn_TFT, drop_last=True)
                
                print("start additional training with validation split {}".format(split_idx+1))
                for e in range(additional_training_epoch):

                    batch_loss_sum = 0.
                    batch_num = len(additional_training_dataloader)

                    TFT_model.train()
                    for input_batch, targets in tqdm(additional_training_dataloader):
                    
                        Optimizer.zero_grad()
                        output_batch = TFT_model(input_batch)
                        loss_batch, q_risk_batch, losses_array = get_quantiles_loss_and_q_risk(outputs=output_batch["predicted_quantiles"],
                                                                                           targets=targets, 
                                                    desired_quantiles=torch.Tensor(self.configuration["model"]["output_quantiles"]))
                        loss_batch.backward()
                        Optimizer.step()
                        batch_loss_sum += loss_batch.item()
                    
                    print("additional training loss at epoch {}: {}".format(e+1, batch_loss_sum/batch_num))

        predicted_qtiles_arr = np.array(predicted_qtiles_list) # num_unique_resid * num_qtiles
        Ytest = copy.deepcopy(self.Y_predict.numpy())
        
        lower_PI = copy.deepcopy(self.Ensemble_pred_interval_centers) + predicted_qtiles_arr[:,0]
        upper_PI = copy.deepcopy(self.Ensemble_pred_interval_centers) + predicted_qtiles_arr[:,-1]

        avg_coverage = np.mean((lower_PI <= Ytest) & (upper_PI >= Ytest))
        avg_width = np.mean(upper_PI - lower_PI)

        print('avg coverage : {}'.format(avg_coverage))
        print('avg width : {}'.format(avg_width))

        loss_dict = {"training_loss_per_epoch" : training_loss_per_epoch, "validation_loss_per_epoch" : validation_loss_per_epoch}

        return predicted_qtiles_arr, Ytest, lower_PI, upper_PI, loss_dict
    
    def compute_PIs_Ensemble_online(self, alpha, stride=1, smallT=True, past_window=100, use_SPCI=False, quantile_regr='RF'):
        '''
            stride: control how many steps we predict ahead
            smallT: if True, we would only start with the last n number of LOO residuals, rather than use the full length T ones. Used in change detection
                NOTE: smallT can be important if time-series is very dynamic, in which case training MORE data may actaully be worse (because quantile longer)
                HOWEVER, if fit quantile regression, set it to be FALSE because we want to have many training pts for the quantile regressor
            use_SPCI: if True, we fit conditional quantile to compute the widths, rather than simply using empirical quantile
        '''
        self.alpha = alpha
        n1 = len(self.X_train)
        self.past_window = past_window # For SPCI, this is the "lag" for predicting quantile
        if smallT:
            # Namely, for special use of EnbPI, only use at most past_window number of LOO residuals.
            n1 = min(self.past_window, len(self.X_train))
        # Now f^b and LOO residuals have been constructed from earlier
        out_sample_predict = self.Ensemble_pred_interval_centers
        out_sample_predictSigmaX = self.Ensemble_pred_interval_sigma
        start = time.time()
        # Matrix, where each row is a UNIQUE slice of residuals with length stride.
        if use_SPCI:
            s = stride
            stride = 1
        # NOTE, NOT ALL rows are actually "observable" in multi-step context, as this is rolling
        resid_strided = utils.strided_app(self.Ensemble_online_resid[len(self.X_train)- n1:-1], n1, stride)
        print(f'Shape of slided residual lists is {resid_strided.shape}')
        num_unique_resid = resid_strided.shape[0]
        width_left = np.zeros(num_unique_resid)
        width_right = np.zeros(num_unique_resid)
        # # NEW, alpha becomes alpha_t. Uncomment things below if we decide to use this upgraded EnbPI
        # alpha_t = alpha
        # errs = []
        # gamma = 0.005
        # method = 'simple'  # 'simple' or 'complex'
        # self.alphas = []
        # NOTE: 'max_features='log2', max_depth=2' make the model "simpler", which improves performance in practice
        self.QRF_ls = []
        self.i_star_ls = []
        for i in range(num_unique_resid):
            curr_SigmaX = out_sample_predictSigmaX[i].item()
            if use_SPCI:
                remainder = i % s 
                if remainder == 0: # update every stride 
                    # Update QRF
                    past_resid = resid_strided[i, :]
                    n2 = self.past_window
                    resid_pred = self.multi_step_QRF(past_resid, i, s, n2)
                # Use the fitted regressor.
                # NOTE, residX is NOT the same as before, as it depends on
                # "past_resid", which has most entries replaced.
                rfqr= self.QRF_ls[remainder]
                i_star = self.i_star_ls[remainder]
                wid_all = rfqr.predict(resid_pred)
                num_mid = int(len(wid_all)/2)
                wid_left = wid_all[i_star]
                wid_right = wid_all[num_mid+i_star]
                width_left[i] = curr_SigmaX * wid_left
                width_right[i] = curr_SigmaX * wid_right
                num_print = int(num_unique_resid / 20)
                if num_print == 0:
                    print(
                            f'Width at test {i} is {width_right[i]-width_left[i]}')
                else:
                    if i % num_print == 0:
                        print(
                            f'Width at test {i} is {width_right[i]-width_left[i]}')
            else:
                past_resid = resid_strided[i, :]
                # Naive empirical quantile, where we use the SAME residuals for multi-step prediction
                # The number of bins will be determined INSIDE binning
                beta_hat_bin = utils.binning(past_resid, alpha)
                # beta_hat_bin = utils.binning(past_resid, alpha_t)
                self.beta_hat_bins.append(beta_hat_bin)
                width_left[i] = curr_SigmaX * np.percentile(
                    past_resid, math.ceil(100 * beta_hat_bin))
                width_right[i] = curr_SigmaX * np.percentile(
                    past_resid, math.ceil(100 * (1 - alpha + beta_hat_bin)))
        print(
            f'Finish Computing {num_unique_resid} UNIQUE Prediction Intervals, took {time.time()-start} secs.')
        Ntest = len(out_sample_predict)
        # This is because |width|=T1/stride.
        width_left = np.repeat(width_left, stride)[:Ntest]
        # This is because |width|=T1/stride.
        width_right = np.repeat(width_right, stride)[:Ntest]
        PIs_Ensemble = pd.DataFrame(np.c_[out_sample_predict + width_left,
                                          out_sample_predict + width_right], columns=['lower', 'upper'])
        self.PIs_Ensemble = PIs_Ensemble
    '''
        Get Multi-step QRF
    '''
    def multi_step_QRF(self, past_resid, i, s, n2):
        '''
            Train multi-step QRF with the most recent residuals
            i: prediction index
            s: num of multi-step, same as stride
            n2: past window w
        '''
        # 1. Get "past_resid" into an auto-regressive fashion
        # This should be more carefully examined, b/c it depends on how long \hat{\eps}_t depends on the past
        # From practice, making it small make intervals wider
        num = len(past_resid)
        resid_pred = past_resid[-n2:].reshape(1, -1)
        residX = sliding_window_view(past_resid[:num-s+1], window_shape=n2)
        for k in range(s):
            residY = past_resid[n2+k:num-(s-k-1)]
            residY2 = sliding_window_view(residY, window_shape=2)
            self.train_QRF(residX, residY)
            if i == 0:
                # Initial training, append QRF to QRF_ls
                self.QRF_ls.append(self.rfqr)
                self.i_star_ls.append(self.i_star)
            else:
                # Retraining, update QRF to QRF_ls
                self.QRF_ls[k] = self.rfqr
                self.i_star_ls[k] = self.i_star
        return resid_pred

    def train_QRF(self, residX, residY):
        alpha = self.alpha
        beta_ls = np.linspace(start=0, stop=alpha, num=self.bins)
        full_alphas = np.append(beta_ls, 1 - alpha + beta_ls)
        self.common_params = dict(n_estimators = self.n_estimators,
                                  max_depth = self.max_d,
                                  criterion = self.criterion,
                                  n_jobs = -1)
        if residX[:-1].shape[0] > 10000:
            # see API ref. https://sklearn-quantile.readthedocs.io/en/latest/generated/sklearn_quantile.RandomForestQuantileRegressor.html?highlight=RandomForestQuantileRegressor#sklearn_quantile.RandomForestQuantileRegressor
            # NOTE, should NOT warm start, as it makes result poor
            self.rfqr = SampleRandomForestQuantileRegressor(
                **self.common_params, q=full_alphas)
        else:
            self.rfqr = RandomForestQuantileRegressor(
                **self.common_params, q=full_alphas)
        # 3. Find best \hat{\beta} via evaluating many quantiles
        # rfqr.fit(residX[:-1], residY)
        sample_weight = None
        if self.weigh_residuals:
            sample_weight = self.c ** np.arange(len(residY), 0, -1)
        if self.T1 is not None:
            self.T1 = min(self.T1, len(residY)) # Sanity check to make sure no errors in training
            self.i_star, _, _, _ = utils.binning_use_RF_quantile_regr(
                self.rfqr, residX[-(self.T1+1):-1], residY[-self.T1:], residX[-1], beta_ls, sample_weight)
        else:
            self.i_star, _, _, _ = utils.binning_use_RF_quantile_regr(
                self.rfqr, residX[:-1], residY, residX[-1], beta_ls, sample_weight)
    '''
        All together
    '''

    def get_results(self, alpha, data_name, itrial, true_Y_predict=[], method='Ensemble'):
        '''
            NOTE: I added a "true_Y_predict" option, which will be used for calibrating coverage under missing data
            In particular, this is needed when the Y_predict we use for training is NOT the same as true Y_predict
        '''
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        train_size = len(self.X_train)
        if method == 'Ensemble':
            PI = self.PIs_Ensemble
        Ytest = self.Y_predict.cpu().detach().numpy()
        coverage = ((np.array(PI['lower']) <= Ytest) & (
            np.array(PI['upper']) >= Ytest)).mean()
        if len(true_Y_predict) > 0: # not used
            coverage = ((np.array(PI['lower']) <= true_Y_predict) & (
                np.array(PI['upper']) >= true_Y_predict)).mean()
        print(f'Average Coverage is {coverage}')
        width = (PI['upper'] - PI['lower']).mean()
        print(f'Average Width is {width}')
        results.loc[len(results)] = [itrial, data_name,
                                     'torch_MLP', method, train_size, coverage, width]
        return results


class MLP(nn.Module):
    def __init__(self, d, sigma=False):
        super(MLP, self).__init__()
        H = 64
        layers = [nn.Linear(d, H), nn.ReLU(), nn.Linear(
            H, H), nn.ReLU(), nn.Linear(H, 1)]
        self.sigma = sigma
        if self.sigma:
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        perturb = 1e-3 if self.sigma else 0
        return self.layers(x) + perturb


#### Competing Methods ####


class QOOB_or_adaptive_CI():
    '''
        Implementation of the QOOB method (Gupta et al., 2021) or the adaptive CI (Gibbs et al., 2022)
    '''

    def __init__(self, fit_func, X_train, X_predict, Y_train, Y_predict):
        self.regressor = fit_func
        self.X_train = X_train
        self.X_predict = X_predict
        self.Y_train = Y_train
        self.Y_predict = Y_predict
    ##############################
    # First on QOOB

    def fit_bootstrap_agg_get_lower_upper(self, B, beta_quantiles):
        '''
          Train B bootstrap estimators from subsets of (X_train, Y_train), compute aggregated predictors, compute scors r_i(X_i,Y_i), and finally get the intervals [l_i(X_n+j),u_i(X_n+j)] for each LOO predictor and the jth prediction in test sample
        '''
        n = len(self.X_train)
        n1 = len(self.X_predict)
        # hold indices of training data for each f^b
        boot_samples_idx = utils.generate_bootstrap_samples(n, n, B)
        # hold lower and upper quantile predictions from each f^b
        boot_predictions_lower = np.zeros((B, (n + n1)), dtype=float)
        boot_predictions_upper = np.zeros((B, (n + n1)), dtype=float)
        # for i^th column, it shows which f^b uses i in training (so exclude in aggregation)
        in_boot_sample = np.zeros((B, n), dtype=bool)
        out_sample_predict_lower = np.zeros((n, n1))
        out_sample_predict_upper = np.zeros((n, n1))
        start = time.time()
        for b in range(B):
            # Fit quantile random forests
            model = self.regressor
            model = model.fit(self.X_train[boot_samples_idx[b], :],
                              self.Y_train[boot_samples_idx[b], ])
            pred_boot = model.predict_quantiles(
                np.r_[self.X_train, self.X_predict], quantiles=beta_quantiles)
            boot_predictions_lower[b] = pred_boot[:, 0]
            boot_predictions_upper[b] = pred_boot[:, 1]
            in_boot_sample[b, boot_samples_idx[b]] = True
        print(
            f'Finish Fitting B Bootstrap models, took {time.time()-start} secs.')
        start = time.time()
        self.QOOB_rXY = []  # the non-conformity scores
        for i in range(n):
            b_keep = np.argwhere(~(in_boot_sample[:, i])).reshape(-1)
            if(len(b_keep) > 0):
                # NOTE: Append these training centers too see their magnitude
                # The reason is sometimes they are TOO close to actual Y.
                quantile_lower = boot_predictions_lower[b_keep, i].mean()
                quantile_upper = boot_predictions_upper[b_keep, i].mean()
                out_sample_predict_lower[i] = boot_predictions_lower[b_keep, n:].mean(
                    0)
                out_sample_predict_upper[i] = boot_predictions_upper[b_keep, n:].mean(
                    0)
            else:  # if aggregating an empty set of models, predict zero everywhere
                print(f'No bootstrap estimator for {i}th LOO estimator')
                quantile_lower = np.percentile(
                    self.Y_train, beta_quantiles[0] * 100)
                quantile_upper = np.percentile(
                    self.Y_train, beta_quantiles[1] * 100)
                out_sample_predict_lower[i] = np.repeat(quantile_lower, n1)
                out_sample_predict_upper[i] = np.repeat(quantile_upper, n1)
            self.QOOB_rXY.append(self.get_rXY(
                self.Y_train[i], quantile_lower, quantile_upper))
        # print('Finish Computing QOOB training' +
        #       r'$\{r_i(X_i,Y_i)\}_{i=1}^N$'+f', took {time.time()-start} secs.')
        # Finally, subtract/add the QOOB_rXY from the predictions
        self.QOOB_rXY = np.array(self.QOOB_rXY)
        out_sample_predict_lower = (
            out_sample_predict_lower.transpose() - self.QOOB_rXY).transpose()
        out_sample_predict_upper = (
            out_sample_predict_upper.transpose() + self.QOOB_rXY).transpose()
        F_minus_i_out_sample = np.r_[
            out_sample_predict_lower, out_sample_predict_upper]
        return F_minus_i_out_sample  # Matrix of shape 2n-by-n1

    def compute_QOOB_intervals(self, data_name, itrial, B, alpha=0.1, get_plots=False):
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        beta_quantiles = [alpha * 2, 1 - alpha * 2]
        # beta_quantiles = [alpha/2, 1-alpha/2]  # Even make thresholds smaller, still not good
        F_minus_i_out_sample = self.fit_bootstrap_agg_get_lower_upper(
            B, beta_quantiles)
        n1 = F_minus_i_out_sample.shape[1]
        PIs = []
        for i in range(n1):
            curr_lower_upper = F_minus_i_out_sample[:, i]
            # print(f'Test point {i}')
            PIs.append(self.get_lower_upper_n_plus_i(curr_lower_upper, alpha))
        PIs = pd.DataFrame(PIs, columns=['lower', 'upper'])
        self.PIs = PIs
        if 'Solar' in data_name:
            PIs['lower'] = np.maximum(PIs['lower'], 0)
        coverage, width = utils.ave_cov_width(PIs, self.Y_predict)
        results.loc[len(results)] = [itrial, data_name,
                                     self.regressor.__class__.__name__, 'QOOB', self.X_train.shape[0], coverage, width]
        if get_plots:
            return [PIs, results]
        else:
            return results
    # QOOB helpers

    def get_rXY(self, Ytrain_i, quantile_lower, quantile_upper):
        # Get r_i(X_i,Y_i) as in Eq. (2) of QOOB
        if Ytrain_i < quantile_lower:
            return quantile_lower - Ytrain_i
        elif Ytrain_i > quantile_upper:
            return Ytrain_i - quantile_upper  # There was a small error here
        else:
            return 0

    # AdaptCI helpers
    def get_Ei(self, Ytrain_i, quantile_lower, quantile_upper):
        return np.max([quantile_lower - Ytrain_i, Ytrain_i - quantile_upper])

    def get_lower_upper_n_plus_i(self, curr_lower_upper, alpha):
        # This implements Algorithm 1 of QOOB
        # See https://github.com/AIgen/QOOB/blob/master/MATLAB/methods/QOOB_interval.m for matlab implementation
        n2 = len(curr_lower_upper)
        n = int(n2 / 2)
        S_ls = np.r_[np.repeat(1, n), np.repeat(0, n)]
        idx_sort = np.argsort(curr_lower_upper)  # smallest to larget
        S_ls = S_ls[idx_sort]
        curr_lower_upper = curr_lower_upper[idx_sort]
        count = 0
        lower_i = np.inf
        upper_i = -np.inf
        threshold = alpha * (n + 1) - 1
        for i in range(n2):
            if S_ls[i] == 1:
                count += 1
                if count > threshold and count - 1 <= threshold and lower_i == np.inf:
                    lower_i = curr_lower_upper[i]
                    # print(f'QOOB lower_end {lower_i}')
            else:
                if count > threshold and count - 1 <= threshold and upper_i == -np.inf:
                    upper_i = curr_lower_upper[i]
                    # print(f'QOOB upper_end {upper_i}')
                count -= 1
        return [lower_i, upper_i]

    ##############################
    # Next on AdaptiveCI

    def compute_AdaptiveCI_intervals(self, data_name, itrial, l, alpha=0.1, get_plots=False):
        results = pd.DataFrame(columns=['itrial', 'dataname', 'muh_fun',
                                        'method', 'train_size', 'coverage', 'width'])
        n = len(self.X_train)
        proper_train = np.arange(l)
        X_train = self.X_train[proper_train, :]
        Y_train = self.Y_train[proper_train]
        X_calibrate = np.delete(self.X_train, proper_train, axis=0)
        Y_calibrate = np.delete(self.Y_train, proper_train)
        # NOTE: below works when the model can takes in MULTIPLE quantiles together (e.g., the RangerForest)
        model = self.regressor
        model = model.fit(X_train, Y_train)
        quantile_pred = model.predict_quantiles(
            np.r_[X_calibrate, self.X_predict], quantiles=[alpha / 2, 1 - alpha / 2])
        # NOTE: below works for sklearn linear quantile: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html#sklearn.linear_model.QuantileRegressor
        # # In particular, it is much slower than the quantile RF with similar results
        # model_l, model_u = self.regressor
        # qpred_l, qpred_u = model_l.fit(X_train, Y_train).predict(np.r_[X_calibrate, self.X_predict]), model_u.fit(
        #     X_train, Y_train).predict(np.r_[X_calibrate, self.X_predict])
        # quantile_pred = np.c_[qpred_l, qpred_u]
        self.quantile_pred = quantile_pred
        Dcal_pred = quantile_pred[:n - l]
        Test_pred = quantile_pred[n - l:]
        # TODO: I guess I can use the QOOB idea, by using "get_rXY"
        Dcal_scores = np.array([self.get_Ei(Ycal, quantile_lower, quantile_upper) for Ycal,
                                quantile_lower, quantile_upper in zip(Y_calibrate, Dcal_pred[:, 0], Dcal_pred[:, 1])])
        self.Escore = Dcal_scores
        # Sequentially get the intervals with adaptive alpha
        alpha_t = alpha
        errs = []
        gamma = 0.005
        method = 'simple'  # 'simple' or 'complex'
        PIs = []
        self.alphas = [alpha_t]
        for t, preds in enumerate(Test_pred):
            lower_pred, upper_pred = preds
            width = np.percentile(Dcal_scores, 100 * (1 - alpha_t))
            # print(f'At test time {t}')
            # print(f'alpha={alpha_t} & width={width}')
            lower_t, upper_t = lower_pred - width, upper_pred + width
            PIs.append([lower_t, upper_t])
            # Check coverage and update alpha_t
            Y_t = self.Y_predict[t]
            err = 1 if Y_t < lower_t or Y_t > upper_t else 0
            errs.append(err)
            alpha_t = utils.adjust_alpha_t(alpha_t, alpha, errs, gamma, method)
            if alpha_t > 1:
                alpha_t = 1
            if alpha_t < 0:
                alpha_t = 0
            self.alphas.append(alpha_t)
        PIs = pd.DataFrame(PIs, columns=['lower', 'upper'])
        if 'Solar' in data_name:
            PIs['lower'] = np.maximum(PIs['lower'], 0)
        self.errs = errs
        self.PIs = PIs
        coverage, width = utils.ave_cov_width(PIs, self.Y_predict)
        results.loc[len(results)] = [itrial, data_name,
                                     self.regressor.__class__.__name__, 'Adaptive_CI', self.X_train.shape[0], coverage, width]
        if get_plots:
            return [PIs, results]
        else:
            return results
        # TODO: I guess I can use the QOOB idea, by using "get_rXY"
        Dcal_scores = np.array([self.get_Ei(Ycal, quantile_lower, quantile_upper) for Ycal,
                                quantile_lower, quantile_upper in zip(Y_calibrate, Dcal_pred[:, 0], Dcal_pred[:, 1])])
        self.Escore = Dcal_scores
        # Sequentially get the intervals with adaptive alpha
        alpha_t = alpha
        errs = []
        gamma = 0.005
        method = 'simple'  # 'simple' or 'complex'
        PIs = []
        self.alphas = [alpha_t]
        for t, preds in enumerate(Test_pred):
            lower_pred, upper_pred = preds
            width = np.percentile(Dcal_scores, 100 * (1 - alpha_t))
            # print(f'At test time {t}')
            # print(f'alpha={alpha_t} & width={width}')
            lower_t, upper_t = lower_pred - width, upper_pred + width
            PIs.append([lower_t, upper_t])
            # Check coverage and update alpha_t
            Y_t = self.Y_predict[t]
            err = 1 if Y_t < lower_t or Y_t > upper_t else 0
            errs.append(err)
            alpha_t = utils.adjust_alpha_t(alpha_t, alpha, errs, gamma, method)
            if alpha_t > 1:
                alpha_t = 1
            if alpha_t < 0:
                alpha_t = 0
            self.alphas.append(alpha_t)
        PIs = pd.DataFrame(PIs, columns=['lower', 'upper'])
        if 'Solar' in data_name:
            PIs['lower'] = np.maximum(PIs['lower'], 0)
        self.errs = errs
        self.PIs = PIs
        coverage, width = utils.ave_cov_width(PIs, self.Y_predict)
        results.loc[len(results)] = [itrial, data_name,
                                     self.regressor.__class__.__name__, 'Adaptive_CI', self.X_train.shape[0], coverage, width]
        if get_plots:
            return [PIs, results]
        else:
            return results


def NEX_CP(X, Y, x, alpha, weights=[], tags=[], seed=1103):
    '''
    # Barber et al. 2022: Nex-CP
    # weights are used for computing quantiles for the prediction interval
    # tags are used as weights in weighted least squares regression
    '''
    n = len(Y)

    if(len(tags) == 0):
        tags = np.ones(n + 1)

    if(len(weights) == 0):
        weights = np.ones(n + 1)
    if(len(weights) == n):
        weights = np.r_[weights, 1]
    weights = weights / np.sum(weights)
    np.random.seed(seed)
    # randomly permute one weight for the regression
    random_ind = int(np.where(np.random.multinomial(1, weights, 1))[1])
    tags[np.c_[random_ind, n]] = tags[np.c_[n, random_ind]]
    XtX = torch.matmul((X.T * tags[:-1]).float(), X) + np.outer(x, x) * tags[-1]
    XY = torch.matmul((X.T * tags[:-1]).float(), Y)

    a = Y - torch.matmul(X, torch.Tensor(np.linalg.solve(XtX, XY)).float())
    b = -torch.matmul(X, torch.Tensor(np.linalg.solve(XtX, x)).float()) * tags[-1]
    a1 = -torch.matmul(x.T, torch.Tensor(np.linalg.solve(XtX, XY)).float())
    b1 =  1 - (torch.matmul(x.T, torch.Tensor(np.linalg.solve(XtX, x)).float()) * tags[-1])
    # if we run weighted least squares on (X[1,],Y[1]),...(X[n,],Y[n]),(x,y)
    # then a + b*y = residuals of data points 1,..,n
    # and a1 + b1*y = residual of data point n+1

    y_knots = np.sort(
        np.unique(np.r_[((a - a1) / (b1 - b))[b1 - b != 0], ((-a - a1) / (b1 + b))[b1 + b != 0]]))
    y_inds_keep = np.where(((np.abs(np.outer(a1 + b1 * y_knots, np.ones(n)))
                             > np.abs(np.outer(np.ones(len(y_knots)), a) + np.outer(y_knots, b))) *
                            weights[:-1]).sum(1) <= 1 - alpha)[0]
    y_PI = np.array([y_knots[y_inds_keep.min()], y_knots[y_inds_keep.max()]])
    if(weights[:-1].sum() <= 1 - alpha):
        y_PI = np.array([-np.inf, np.inf])
    return y_PI

#### Testing functions based on methods above #####


wind_loc = 0  # Can change this to try wind prediction on different locations


def test_EnbPI_or_SPCI(main_condition, results_EnbPI_SPCI, itrial=0):
    '''
    Arguments:

        main_condition: Contain these three below:
            bool. simulation:  True use simulated data. False use solar
                simul_type: int. 1 = simple state-space. 2 = non-statioanry. 3 = heteroskedastic
                The latter 2 follows from case 3 in paper
            bool. use_SPCI: True use `quantile_regr`. False use empirical quatile
            str. quantile_regr:  Which quantile regression to fit residuals (e.g., "RF", "LR")

    Other (not arguments)

        fit_func: None or sklearn module with methods `.fit` & `.predict`. If None, use MLP above

        fit_sigmaX: bool. True if to fit heteroskedastic errors. ONLY activated if fit_func is NONE (i.e. MLP), because errors are unobserved so `.fit()` does not work

        smallT: bool. True if empirical quantile uses not ALL T residual in the past to get quantile (should be tuned as sometimes longer memory causes poor coverage)
            past_window: int. If smallT True, EnbPI uses `past_window` most residuals to get width. FOR quantile_regr of residuals, it determines the dimension of the "feature" that predict new quantile of residuals autoregressively

    Results:
        dict: contains dictionary of coverage and width under different training fraction (fix alpha) under various argument combinations
    '''
    simulation, use_SPCI, quantile_regr, use_NeuralProphet = main_condition
    non_stat_solar, save_dict_rolling = results_EnbPI_SPCI.other_conditions
    train_ls, alpha = results_EnbPI_SPCI.train_ls, results_EnbPI_SPCI.alpha
    univariate, filter_zero = results_EnbPI_SPCI.data_conditions
    result_cov, result_width = [], []
    for train_frac in train_ls:
        print('########################################')
        print(f'Train frac at {train_frac}')
        ''' Get Data '''
        if simulation:
            simul_type = results_EnbPI_SPCI.simul_type  # 1, 2, 3
            fit_sigmaX = True if simul_type == 3 else False  # If we fit variance given X_t
            simul_name_dict = {1: 'simulation_state_space',
                               2: 'simulate_nonstationary', 3: 'simulate_heteroskedastic'}
            data_name = simul_name_dict[simul_type]
            simul_loader = data.simulate_data_loader()
            Data_dict = simul_loader.get_simul_data(simul_type)
            X_full, Y_full = Data_dict['X'].to(
                device), Data_dict['Y'].to(device)
            B = 20
            past_window = 500
            fit_func = None
            # if simul_type == 3:
            #     fit_func = None  # It is MLP above
            # else:
            #     fit_func = RandomForestRegressor(n_estimators=10, criterion='mse',
            #                                      bootstrap=False, n_jobs=-1, random_state=1103+itrial)
        else:
            data_name = results_EnbPI_SPCI.data_name
            dloader = data.real_data_loader()
            solar_args = [univariate, filter_zero, non_stat_solar]
            wind_args = [wind_loc]
            X_full, Y_full = dloader.get_data(data_name, solar_args, wind_args)
            RF_seed = 1103+itrial
            if data_name == 'solar':
                fit_func = RandomForestRegressor(n_estimators=10, criterion='mse',
                                                 bootstrap=False, n_jobs=-1, random_state=RF_seed)
                past_window = 200 if use_SPCI else 300
            if data_name == 'electric':
                fit_func = RandomForestRegressor(n_estimators=10, max_depth=1, criterion='mse',
                                                 bootstrap=False, n_jobs=-1, random_state=RF_seed)
                past_window = 300
            if data_name == 'wind':
                fit_func = RandomForestRegressor(n_estimators=10, max_depth=1, criterion='mse',
                                                 bootstrap=False, n_jobs=-1, random_state=RF_seed)
                past_window = 300
            Y_full, X_full = torch.from_numpy(Y_full).float().to(
                device), torch.from_numpy(X_full).float().to(device)
            fit_sigmaX = False
            B = 25
        N = int(X_full.shape[0] * train_frac)
        X_train, X_predict, Y_train, Y_predict = X_full[:
                                                        N], X_full[N:], Y_full[:N], Y_full[N:]

        ''' Train '''
        EnbPI = SPCI_and_EnbPI(
            X_train, X_predict, Y_train, Y_predict, fit_func=fit_func)
        EnbPI.use_NeuralProphet = use_NeuralProphet
        stride = results_EnbPI_SPCI.stride
        EnbPI.fit_bootstrap_models_online_multistep(
            B, fit_sigmaX=fit_sigmaX, stride=stride)
        # Under cond quantile, we are ALREADY using the last window for prediction so smallT is FALSE, instead, we use ALL residuals in the past (in a sliding window fashion) for training the quantile regressor
        smallT = not use_SPCI
        EnbPI.compute_PIs_Ensemble_online(
            alpha, smallT=smallT, past_window=past_window, use_SPCI=use_SPCI,
            quantile_regr=quantile_regr, stride=stride)
        results = EnbPI.get_results(alpha, data_name, itrial)

        ''' Save results '''
        result_cov.append(results['coverage'].item())
        result_width.append(results['width'].item())
        PI = EnbPI.PIs_Ensemble
        if use_SPCI:
            if use_NeuralProphet:
                results_EnbPI_SPCI.PIs_SPCINeuralProphet = PI
            else:
                results_EnbPI_SPCI.PIs_SPCI = PI
        else:
            results_EnbPI_SPCI.PIs_EnbPI = PI
        Ytest = EnbPI.Y_predict.cpu().detach().numpy()
        results_EnbPI_SPCI.dict_rolling[f'Itrial{itrial}'] = PI
        name = 'SPCI' if use_SPCI else 'EnbPI'
        if use_NeuralProphet:
            name = 'SPCI-NeuralProphet'
        if save_dict_rolling:
            with open(f'{name}_{data_name}_train_frac_{np.round(train_frac,2)}_alpha_{alpha}.p', 'wb') as fp:
                pickle.dump(results_EnbPI_SPCI.dict_rolling, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)
        if simulation:
            # # Examine recovery of F and Sigma
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            ax[0, 0].plot(Data_dict['f(X)'])
            Y_t_hat = EnbPI.Ensemble_pred_interval_centers
            ax[0, 1].plot(Y_t_hat)
            ax[1, 0].plot(Data_dict['Eps'])
            ax[1, 1].plot(EnbPI.Ensemble_online_resid)
            titles = [r'True $f(X)$', r'Est $f(X)$',
                      r'True $\epsilon$', r'Est $\epsilon$']
            fig.tight_layout()
            for i, ax_i in enumerate(ax.flatten()):
                ax_i.set_title(titles[i])
            fig.tight_layout()
            plt.show()
            plt.close()
    results_EnbPI_SPCI.dict_full[name] = np.vstack(
        [result_cov, result_width])
    results_EnbPI_SPCI.Ytest = Ytest
    results_EnbPI_SPCI.train_size = N
    results_EnbPI_SPCI.data_name = data_name
    utils.dict_to_latex(results_EnbPI_SPCI.dict_full, train_ls)
    return results_EnbPI_SPCI


def test_adaptive_CI(results_Adapt_CI, itrial=0):
    train_ls, alpha = results_Adapt_CI.train_ls, results_Adapt_CI.alpha
    non_stat_solar, save_dict_rolling = results_Adapt_CI.other_conditions
    univariate, filter_zero = results_Adapt_CI.data_conditions
    # NOTE: the variance of this method seems high, and I often need to tune a LOT to avoid yielding very very high coverage.
    data_name = results_Adapt_CI.data_name
    cov_ls, width_ls = [], []
    for train_frac in train_ls:
        # As it is split conformal, the result can be random, so we repeat over seed
        seeds = [524, 1103, 1111, 1214, 1228]
        seeds = [seed+itrial+1231 for seed in seeds]
        cov_tmp_ls, width_tmp_ls = [], []
        print('########################################')
        print(f'Train frac at {train_frac} over {len(seeds)} seeds')
        PI_ls = []
        for seed in seeds:
            data_name = results_Adapt_CI.data_name
            dloader = data.real_data_loader()
            solar_args = [univariate, filter_zero, non_stat_solar]
            wind_args = [wind_loc]
            X_full, Y_full = dloader.get_data(data_name, solar_args, wind_args)
            N = int(X_full.shape[0] * train_frac)
            X_train, X_predict, Y_train, Y_predict = X_full[:
                                                            N], X_full[N:], Y_full[:N], Y_full[N:]
            if non_stat_solar:
                # More complex yields wider intervals and more conservative coverage
                fit_func = RangerForestRegressor(
                    n_estimators=5, quantiles=True, seed=seed)
            else:
                fit_func = RangerForestRegressor(
                    n_estimators=10, quantiles=True, seed=seed)
            PI_test_adaptive = QOOB_or_adaptive_CI(
                fit_func, X_train, X_predict, Y_train, Y_predict)
            PI_test_adaptive.compute_AdaptiveCI_intervals(
                data_name, 0, l=int(0.75 * X_train.shape[0]),
                alpha=alpha)
            PIs_AdaptiveCI = PI_test_adaptive.PIs
            PI_ls.append(PIs_AdaptiveCI)
            Ytest = PI_test_adaptive.Y_predict
            coverage = ((np.array(PIs_AdaptiveCI['lower']) <= Ytest)
                        & (np.array(PIs_AdaptiveCI['upper']) >= Ytest))
            width = (
                (np.array(PIs_AdaptiveCI['upper']) - np.array(PIs_AdaptiveCI['lower'])))
            cov_tmp_ls.append(coverage)
            width_tmp_ls.append(width)
        lowers = np.mean([a['lower'] for a in PI_ls], axis=0)
        uppers = np.mean([a['upper'] for a in PI_ls], axis=0)
        PIs_AdaptiveCI = pd.DataFrame(
            np.c_[lowers, uppers], columns=['lower', 'upper'])
        coverage = np.vstack(cov_tmp_ls).mean(axis=0)
        width = np.vstack(width_tmp_ls).mean(axis=0)
        results_Adapt_CI.PIs_AdaptiveCI = PIs_AdaptiveCI
        results_Adapt_CI.dict_rolling[f'Itrial{itrial}'] = PIs_AdaptiveCI
        if save_dict_rolling:
            with open(f'AdaptiveCI_{data_name}_train_frac_{np.round(train_frac,2)}_alpha_{alpha}.p', 'wb') as fp:
                pickle.dump(results_Adapt_CI.dict_rolling, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)
        cov_ls.append(np.mean(coverage))
        width_ls.append(np.mean(width))
    results_Adapt_CI.dict_full['AdaptiveCI'] = np.vstack(
        [cov_ls, width_ls])
    utils.dict_to_latex(results_Adapt_CI.dict_full, train_ls)
    return results_Adapt_CI


def test_NEX_CP(results_NEX_CP, itrial=0):
    train_ls, alpha = results_NEX_CP.train_ls, results_NEX_CP.alpha
    non_stat_solar, save_dict_rolling = results_NEX_CP.other_conditions
    univariate, filter_zero = results_NEX_CP.data_conditions
    cov, width = [], []
    data_name = results_NEX_CP.data_name
    dloader = data.real_data_loader()
    solar_args = [univariate, filter_zero, non_stat_solar]
    wind_args = [wind_loc]
    X_full, Y_full = dloader.get_data(data_name, solar_args, wind_args)
    N = len(X_full)
    for train_frac in train_ls:
        train_size = int(train_frac * N)
        PI_nexCP_WLS = np.zeros((N, 2))
        for n in np.arange(train_size, N):
            # weights and tags (parameters for new methods)
            rho = 0.99
            rho_LS = 0.99
            weights = rho**(np.arange(n, 0, -1))
            tags = rho_LS**(np.arange(n, -1, -1))
            PI_nexCP_WLS[n, :] = NEX_CP(X_full[:n, :], Y_full[:n], X_full[n, :], alpha,
                                        weights=weights, tags=tags, seed=1103+itrial)
            inc = int((N - train_size) / 20)
            if (n - train_size) % inc == 0:
                print(
                    f'NEX-CP WLS width at {n-train_size} is: {PI_nexCP_WLS[n,1] - PI_nexCP_WLS[n,0]}')
        cov_nexCP_WLS = (PI_nexCP_WLS[train_size:, 0] <= Y_full[train_size:N]) *\
            (PI_nexCP_WLS[train_size:, 1] >= Y_full[train_size:N])
        PI_width_nexCP_WLS = PI_nexCP_WLS[train_size:,
                                          1] - PI_nexCP_WLS[train_size:, 0]
        PI_nexCP_WLS = PI_nexCP_WLS[train_size:]
        PI_nexCP_WLS = pd.DataFrame(PI_nexCP_WLS, columns=['lower', 'upper'])
        cov.append(np.mean(cov_nexCP_WLS))
        width.append(np.mean(PI_width_nexCP_WLS))
        print(
            f'At {train_frac} tot data \n cov: {cov[-1]} & width: {width[-1]}')
        # Rolling coverage and width
        # cov_moving = utils.rolling_avg(cov_nexCP_WLS)
        # width_moving = utils.rolling_avg(PI_width_nexCP_WLS)
        results_NEX_CP.PI_nexCP_WLS = PI_nexCP_WLS
        results_NEX_CP.dict_rolling[f'Itrial{itrial}'] = PI_nexCP_WLS
        if save_dict_rolling:
            with open(f'NEXCP_{data_name}_train_frac_{np.round(train_frac,2)}_alpha_{alpha}.p', 'wb') as fp:
                pickle.dump(results_NEX_CP.dict_rolling, fp,
                            protocol=pickle.HIGHEST_PROTOCOL)
    results_NEX_CP.dict_full['NEXCP'] = np.vstack([cov, width])
    utils.dict_to_latex(results_NEX_CP.dict_full, train_ls)
    return results_NEX_CP

