# 尝试大网络线性模拟


# FILE: sequential_simulation_k2_nonlinear_ic_search_jupyter.py
#
# >>> MODIFIED by Gemini to:
# 1. Fix all U+00A0 (non-printable character) syntax errors.
# 2. Implement a full search path for Lambda (La = Lb).
# 3. For each lambda, calculate and log NLL, AIC, BIC, and HBIC.
# 4. Print the full Information Criteria (IC) table for each simulation run.
# 5. Add a new benchmark "NetworkNetFixed" that uses the user's known-good lambda=0.1.
# 6. *** REMOVED `if __name__ == '__main__'` for direct Jupyter cell execution. ***
# 7. *** ENABLED `tqdm.notebook` for all loops for Jupyter monitoring. ***
# 8. *** ADDED `print` statements for step-by-step progress monitoring. ***
#

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from typing import Tuple, List, Dict
import pandas as pd
import time
import warnings
# *** 确保使用 tqdm.notebook ***
from tqdm import tqdm
import math
from sklearn.linear_model import LassoCV
from sklearn.exceptions import ConvergenceWarning
import traceback # Import traceback for detailed error reporting

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore')


# --- Prerequisite Code (Copied Directly - NO CHANGES HERE) ---

# --- MLE Params (Robust Version) ---
def estimate_mle_params(Y: np.ndarray, n_epochs: int = 10000, lr: float = 0.05, verbose: bool = False, device='cpu') -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimates MLE parameters, returns None if unstable."""
    n = Y.shape[0]; Y_tensor = torch.FloatTensor(Y).to(device)
    alpha = nn.Parameter(torch.zeros(n, 1, device=device)); beta = nn.Parameter(torch.zeros(n, 1, device=device))
    optimizer = optim.Adam([alpha, beta], lr=lr)
    try:
        # *** 使用 tqdm.notebook (如果 verbose=True) ***
        epoch_iterator = tqdm(range(n_epochs), desc="MLE Estimation", leave=False) if verbose else range(n_epochs)
        for epoch in epoch_iterator:
            optimizer.zero_grad();
            with torch.no_grad(): beta.data -= torch.mean(beta.data)
            alpha_clamped=torch.clamp(alpha,-20,20); beta_clamped=torch.clamp(beta,-20,20)
            lambda_ij=torch.exp(alpha_clamped.T+beta_clamped); lambda_ij=torch.clamp(lambda_ij,1e-6,1e8)
            log_lambda_ij=alpha_clamped.T+beta_clamped; mask=~torch.eye(n,dtype=torch.bool,device=device)
            loss=torch.sum((lambda_ij-Y_tensor*log_lambda_ij)*mask);
            if torch.isnan(loss) or torch.isinf(loss): return None, None
            loss.backward();
            grad_norm_alpha=torch.norm(alpha.grad) if alpha.grad is not None else 0; grad_norm_beta=torch.norm(beta.grad) if beta.grad is not None else 0
            if torch.isnan(grad_norm_alpha).any() or torch.isnan(grad_norm_beta).any() or torch.isinf(grad_norm_alpha).any() or torch.isinf(grad_norm_beta).any(): return None, None
            optimizer.step()
    except Exception as e: print(f"MLE Error: {e}"); return None, None
    alpha_np=alpha.data.cpu().numpy().flatten(); beta_np=beta.data.cpu().numpy().flatten();
    if np.isnan(alpha_np).any() or np.isnan(beta_np).any(): return None, None
    return alpha_np, beta_np

# --- CountNetworkLassoNet (No Change) ---
class CountNetworkLassoNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.1):
        super().__init__(); self.input_dim = input_dim;
        self.skip = nn.Linear(input_dim, 1, bias=False); self.first_layer = nn.Linear(input_dim, hidden_dims[0])
        layers = [self.first_layer, nn.ReLU(), nn.Dropout(dropout)]; prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]: layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]); prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1, bias=False)); self.network = nn.Sequential(*layers)
    def forward(self, x): return self.skip(x) + self.network(x)
    def get_selected_features(self, threshold=1e-4):
        if not hasattr(self.skip,'weight') or self.skip.weight is None: return np.array([],dtype=int)
        try:
            imp=torch.abs(self.skip.weight.data).squeeze().cpu().numpy();
            if np.isnan(imp).any(): return np.array([],dtype=int)
            if imp.ndim==0: imp=np.array([imp]);
            idx=np.where(imp > threshold)[0]; return idx[idx < self.input_dim]
        except Exception: return np.array([],dtype=int)

# --- Hierarchical Prox (No Change) ---
def _hierarchical_prox(model: nn.Module, lambda_: float, step_size: float, M: int):
    with torch.no_grad():
        if not hasattr(model.skip,'weight') or model.skip.weight is None or not hasattr(model.first_layer,'weight') or model.first_layer.weight is None: return
        theta, W1 = model.skip.weight.data, model.first_layer.weight.data;
        if theta.ndim==1: theta = theta.unsqueeze(0);
        input_dim=model.input_dim; theta_cols=theta.shape[1] if theta.ndim>1 else 0; w1_cols=W1.shape[1] if W1.ndim>1 else 0; eff_dim=min(input_dim,theta_cols,w1_cols);
        for j in range(eff_dim):
            theta_j_val=theta[0,j].item(); w_j=W1[:,j]; thresh=lambda_*step_size;
            if abs(theta_j_val) <= thresh: theta[0,j]=0.0; W1[:,j]=0.0;
            else:
                shrunk_theta_j=torch.sign(theta[0,j])*(abs(theta_j_val)-thresh); theta[0,j]=shrunk_theta_j;
                max_w=torch.max(torch.abs(w_j)); cur_theta_abs=abs(shrunk_theta_j.item());
                if max_w>1e-8 and cur_theta_abs>1e-8 and max_w>M*cur_theta_abs: W1[:,j]*=(M*cur_theta_abs/max_w);

# --- Data Generation (No Change) ---
def generate_simulation_data(n_samples: int, n_features: int, signal_strength: float, relationship_type: str = 'linear') -> tuple:
    if n_features < 10: raise ValueError("n_features must be >= 10");
    if relationship_type not in ['linear', 'nonlinear']: raise ValueError("relationship_type must be 'linear' or 'nonlinear'")
    # --- Nonlinear needs positive X for log ---
    X_data = np.random.uniform(low=0.5, high=1.5, size=(n_samples, n_features))
    feature_names = [f'X{i+1}' for i in range(n_features)]
    X = pd.DataFrame(X_data, columns=feature_names); c = signal_strength
    true_idx_alpha = np.arange(5); true_idx_beta = np.arange(5, 10)
    if relationship_type == 'linear':
        true_alpha = c * (X['X1'] + X['X2'] + X['X3'] + X['X4'] + X['X5']); 
        true_beta  = c * (X['X6'] + X['X7'] + X['X8'] + X['X9'] + X['X10'])
    elif relationship_type == 'nonlinear':
       # Using the slightly different non-linear function from previous user code
       # Ensure X values are positive before taking log
       true_alpha = c * ( np.abs(X['X1']) + np.abs(X['X2']) + np.log(X['X3']) + np.log(X['X4']) + np.abs(X['X5']) )
       true_beta  = c * ( np.abs(X['X6']) + np.abs(X['X7']) + np.log(X['X8']) + np.log(X['X9']) + np.abs(X['X10']) )
    true_alpha_np = true_alpha.to_numpy(); true_beta_np = true_beta.to_numpy(); true_beta_np -= true_beta_np.mean()
    if np.isnan(true_alpha_np).any() or np.isnan(true_beta_np).any() or \
       np.isinf(true_alpha_np).any() or np.isinf(true_beta_np).any():
        raise ValueError("NaN or Inf generated in true alpha/beta. Check non-linear function and X range.")
    log_lambda_ij = true_alpha_np[:, None] + true_beta_np[None, :]; true_lambda = np.exp(np.clip(log_lambda_ij, -100, 80))
    Y = np.random.poisson(true_lambda); np.fill_diagonal(Y, 0); return Y, X, true_alpha_np, true_beta_np, true_lambda, true_idx_alpha, true_idx_beta

# --- Prepare Edge Data (No Change) ---
def prepare_edge_data(Y: torch.Tensor, device: str):
    n = Y.shape[0]; r_idx = torch.arange(n, device=device).repeat_interleave(n); c_idx = torch.arange(n, device=device).repeat(n)
    off_diag = r_idx != c_idx; return r_idx[off_diag], c_idx[off_diag], Y.flatten()[off_diag]

# --- Evaluate Alpha Results (No Change) ---
def evaluate_alpha_results(true_alpha, est_alpha, true_idx_alpha, selected_alpha_indices):
    if est_alpha is None or np.isnan(est_alpha).any(): return {'rmse_alpha': np.nan, 'n_sel_alpha': np.nan, 'prec_alpha': np.nan, 'rec_alpha': np.nan, 'f1_alpha': np.nan}
    results = {}; true_alpha_c = true_alpha - true_alpha.mean(); est_alpha_c = est_alpha - est_alpha.mean()
    results['rmse_alpha'] = np.sqrt(np.mean((true_alpha_c - est_alpha_c)**2))
    selected = set(selected_alpha_indices); true = set(true_idx_alpha); tp = len(selected & true); fp = len(selected - true); fn = len(true - selected)
    precision = tp/(tp+fp) if (tp+fp)>0 else 0; recall = tp/(tp+fn) if (tp+fn)>0 else 0; f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    results['n_sel_alpha'] = len(selected); results['prec_alpha'] = precision; results['rec_alpha'] = recall; results['f1_alpha'] = f1
    return results

# --- Evaluate Beta Results (No Change) ---
def evaluate_beta_results(true_beta, est_beta, true_idx_beta, selected_beta_indices):
    if est_beta is None or np.isnan(est_beta).any(): return {'rmse_beta': np.nan, 'n_sel_beta': np.nan, 'prec_beta': np.nan, 'rec_beta': np.nan, 'f1_beta': np.nan}
    results = {}; true_beta_c = true_beta - true_beta.mean(); est_beta_c = est_beta - est_beta.mean()
    results['rmse_beta'] = np.sqrt(np.mean((true_beta_c - est_beta_c)**2))
    selected = set(selected_beta_indices); true = set(true_idx_beta); tp = len(selected & true); fp = len(selected - true); fn = len(true - selected)
    precision = tp/(tp+fp) if (tp+fp)>0 else 0; recall = tp/(tp+fn) if (tp+fn)>0 else 0; f1 = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    results['n_sel_beta'] = len(selected); results['prec_beta'] = precision; results['rec_beta'] = recall; results['f1_beta'] = f1
    return results

# --- Helper Function for NetworkNet Training Step ---
def train_network_step(network, optimizer, X_t, y_v, r_idx, c_idx,
                       fixed_param_e, sum_fixed_param,
                       n_epochs, lr, lambda_penalty, M, gamma_balance,
                       is_alpha_update, desc=""):
    """Performs PGD training for one network (alpha or beta) for n_epochs."""
    network_name = "Alpha" if is_alpha_update else "Beta";
    # *** 启用 TQDM.NOTEBOOK ***
    epoch_iterator = tqdm(range(n_epochs), desc=f"Training {network_name} ({desc})", leave=False)
    
    for epoch in epoch_iterator:
        network.train(); optimizer.zero_grad(); current_all = network(X_t).squeeze();
        if is_alpha_update:
            alpha_e = current_all[r_idx]; log_lambda = alpha_e + fixed_param_e; bal = gamma_balance * (torch.sum(current_all) - sum_fixed_param)**2
        else:
            beta_e = current_all[c_idx]; log_lambda = fixed_param_e + beta_e; bal = gamma_balance * (sum_fixed_param - torch.sum(current_all))**2
        log_lambda = torch.clamp(log_lambda, -20, 20); p_loss = torch.mean(torch.exp(log_lambda) - y_v * log_lambda); loss = p_loss + bal;
        if torch.isnan(loss) or torch.isinf(loss): print(f"\nNaN/Inf loss ({network_name}) epoch {epoch}. Stop."); return False
        loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0); optimizer.step(); _hierarchical_prox(network, lambda_penalty, lr, M);
    return True

# --- Helper Function for MLELassoNet Training Step ---
def train_mle_lassonet_step(network, optimizer, X_t, target_t,
                            n_epochs, lr, lambda_penalty, M, desc=""):
    """Performs PGD training for CountNetworkLassoNet with MSE loss."""
    criterion = nn.MSELoss() # Use MSE loss for regression on MLE targets
    # *** 启用 TQDM.NOTEBOOK ***
    epoch_iterator = tqdm(range(n_epochs), desc=f"Training MLELassoNet ({desc})", leave=False)
    
    for epoch in epoch_iterator:
        network.train(); optimizer.zero_grad(); predictions = network(X_t).squeeze()
        loss = criterion(predictions, target_t) # MSE loss
        if torch.isnan(loss) or torch.isinf(loss): print(f"\nNaN/Inf loss (MLELassoNet) epoch {epoch}. Stop."); return False
        loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0); optimizer.step(); _hierarchical_prox(network, lambda_penalty, lr, M);
    return True

# --- Helper Function for BIC Calculation (MSE) (No Change) ---
def calculate_bic_mse(target_y, predicted_y, n_samples, n_selected_features):
    """Calculates BIC for a regression model based on RSS."""
    if n_selected_features == 0: n_selected_features = 1 # Avoid issues with k=0
    rss = np.sum((target_y - predicted_y)**2);
    if rss < 1e-9: rss = 1e-9
    if n_samples <= 0: return np.inf 
    bic = n_selected_features * np.log(n_samples) + n_samples * np.log(rss / n_samples)
    return bic if np.isfinite(bic) else np.inf

# --- *** NEW: Helper Function for Information Criteria (Poisson) *** ---
def calculate_information_criteria(y_v_tensor: torch.Tensor, 
                                   log_lambda_v_tensor: torch.Tensor, 
                                   n_edges: int, 
                                   n_selected_features_total: int,
                                   n_features: int) -> dict:
    """
    Calculates NLL, AIC, BIC, and HBIC for the Poisson network model.
    Assumes NLL is based on the Poisson loss (ignoring constant log(y!) term).
    """
    with torch.no_grad():
        # Clamp for stability, mirroring the training loss
        log_lambda_v_tensor = torch.clamp(log_lambda_v_tensor, -20, 20)
        
        # 1. Total Negative Log-Likelihood (NLL)
        total_neg_log_lik = torch.mean(torch.exp(log_lambda_v_tensor) - y_v_tensor * log_lambda_v_tensor)
        
        nll = total_neg_log_lik.item()
        k = float(n_selected_features_total)
        n = float(n_edges)
        p = float(n_features)

        if n <= 0: return {'nll': nll, 'aic': np.inf, 'bic': np.inf, 'hbic': np.inf}
        
        log_n = np.log(n)
        log_p = np.log(p) if p > 1 else 1.0 
        
        # 2. AIC = 2 * NLL + 2 * k
        aic = 2 * nll + 2 * k
        
        # 3. BIC = 2 * NLL + k * log(n)
        bic_penalty = k * log_n
        bic = 2 * nll + bic_penalty
        
        # 4. HBIC = 2 * NLL + k * log(n) * log(p)
        hbic_penalty = bic_penalty * log_p   
        hbic = 2 * nll + hbic_penalty * 200
        
        return {
            'nll': nll,
            'aic': aic,
            'bic': bic,
            'hbic': hbic,
            'n_selected': k
        }



def run_networknet_k_iterations(
    lambda_val, k_iterations, n_features, fixed_nn_params, learning_rate, n_epochs_per_step,
    X_t, y_v, r_idx, c_idx, a_mle_t, b_mle_t, device, desc_prefix=""
):
    """
    Runs the K-iteration training loop for a *single* lambda value.
    Returns the final models, predictions, and success status.
    """
    # 1. Re-initialize models and optimizers
    net_alpha = CountNetworkLassoNet(input_dim=n_features, hidden_dims=fixed_nn_params['hidden_dims'], dropout=fixed_nn_params['dropout']).to(device)
    net_beta = CountNetworkLassoNet(input_dim=n_features, hidden_dims=fixed_nn_params['hidden_dims'], dropout=fixed_nn_params['dropout']).to(device)
    optimizer_alpha = optim.SGD(net_alpha.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_beta = optim.SGD(net_beta.parameters(), lr=learning_rate, momentum=0.9)
    
    a_curr_t = a_mle_t.clone() 
    b_curr_t = b_mle_t.clone()
    nn_success = True
    
    # 2. Run the K=2 iterations for this lambda
    for k in range(k_iterations):
        # Beta update
        fixed_alpha_e_k = a_curr_t.detach()[r_idx]; sum_alpha_k = torch.sum(a_curr_t.detach())
        beta_update_success = train_network_step(
            net_beta, optimizer_beta, X_t, y_v, r_idx, c_idx, 
            fixed_alpha_e_k, sum_alpha_k, 
            n_epochs_per_step, learning_rate, 
            lambda_val, 
            fixed_nn_params['M'], fixed_nn_params['gamma_balance'], 
            is_alpha_update=False, 
            desc=f"{desc_prefix} Iter {k+1} Beta"
        )
        if not beta_update_success: nn_success = False; break
        with torch.no_grad(): net_beta.eval(); b_curr_t = net_beta(X_t).squeeze(); b_curr_t = torch.nan_to_num(b_curr_t, nan=0.0)
        
        # Alpha update
        fixed_beta_e_k = b_curr_t.detach()[c_idx]; sum_beta_k = torch.sum(b_curr_t.detach())
        alpha_update_success = train_network_step(
            net_alpha, optimizer_alpha, X_t, y_v, r_idx, c_idx, 
            fixed_beta_e_k, sum_beta_k, 
            n_epochs_per_step, learning_rate, 
            lambda_val, 
            fixed_nn_params['M'], fixed_nn_params['gamma_balance'], 
            is_alpha_update=True, 
            desc=f"{desc_prefix} Iter {k+1} Alpha"
        )
        if not alpha_update_success: nn_success = False; break
        with torch.no_grad(): net_alpha.eval(); a_curr_t = net_alpha(X_t).squeeze(); a_curr_t = torch.nan_to_num(a_curr_t, nan=0.0)

    return net_alpha, net_beta, a_curr_t, b_curr_t, nn_success


# ---------------------------------------------------------------------------- #
#     *** Function for Running One Simulation Instance w/ Benchmarks *** #
# ---------------------------------------------------------------------------- #
def run_single_simulation_with_benchmarks(sim_id, dgp_params, fixed_nn_params, fit_nn_params, k_iterations, device):
    """Runs one full simulation instance including benchmarks."""

    # *** MONITOR: Print run start ***
    print(f"\n--- [Run {sim_id+1}] 正在启动 ---")

    current_seed_run = SEED + sim_id
    np.random.seed(current_seed_run)
    torch.manual_seed(current_seed_run)
    if device == 'cuda': torch.cuda.manual_seed_all(current_seed_run)

    n_features = dgp_params['n_features']
    learning_rate = fit_nn_params['lr']
    n_epochs_per_step = fit_nn_params['n_epochs_per_step']
    mle_epochs = fit_nn_params['mle_epochs']
    mle_lr = fit_nn_params['mle_lr']
    lambda_fixed_alpha = fit_nn_params['lambda_alpha']
    lambda_fixed_beta = fit_nn_params['lambda_beta']
    all_final_metrics = {}

    # --- 1. Generate Data ---
    # *** MONITOR: Print step ***
    print(f"--- [Run {sim_id+1}] 1. 正在生成数据 (Type: {dgp_params['relationship_type']}) ---")
    try:
        Y, X_df, true_alpha, true_beta, _, true_idx_alpha, true_idx_beta = generate_simulation_data(**dgp_params)
        X = X_df.to_numpy()
        n_samples = X.shape[0]
    except ValueError as e:
        print(f"[{sim_id+1}] Error generating data: {e}. Skipping run.")
        return {}

    # --- 2. Estimate MLE ---
    # *** MONITOR: Print step ***
    print(f"--- [Run {sim_id+1}] 2. 正在估算 MLE (Epochs: {mle_epochs}) ---")
    alpha_mle, beta_mle = estimate_mle_params(Y, n_epochs=mle_epochs, lr=mle_lr, device=device, verbose=True) # Set verbose=True
    if alpha_mle is None or beta_mle is None:
        print(f"[{sim_id+1}] MLE Failed. Skipping run.")
        return {}
    print(f"--- [Run {sim_id+1}]    MLE 估算完成 ---")

    alpha_mle_c = alpha_mle - alpha_mle.mean(); beta_mle_c = beta_mle - beta_mle.mean()
    true_alpha_c = true_alpha - true_alpha.mean(); true_beta_c = true_beta - true_beta.mean()
    all_final_metrics['rmse_alpha_mle'] = np.sqrt(np.mean((true_alpha_c - alpha_mle_c)**2))
    all_final_metrics['rmse_beta_mle'] = np.sqrt(np.mean((true_beta_c - beta_mle_c)**2))

    # --- 3. Prepare Tensors ---
    X_t = torch.FloatTensor(X).to(device)
    Y_t = torch.FloatTensor(Y).to(device)
    r_idx, c_idx, y_v = prepare_edge_data(Y_t, device)
    n_edges = len(y_v) 
    a_mle_t = torch.FloatTensor(alpha_mle).to(device)
    b_mle_t = torch.FloatTensor(beta_mle).to(device)


    # =======================================================
    #     METHOD 1: NetworkNetIC (IC Selection)
    # =======================================================
    # *** MONITOR: Print step ***
    print(f"--- [Run {sim_id+1}] 3. 方法 1: NetworkNetIC (IC 搜索) ---")
    try:
        if n_edges == 0: raise ValueError("No edges found for IC calculation.")
        
        lambda_search_path = [0.2,0.4,0.8,0.9,0.95,1,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.6] ##################################################################
        ic_search_results = []
        
        for current_lambda in lambda_search_path:
            # *** MONITOR: Print sub-step ***
            print(f"--- [Run {sim_id+1}]    IC 搜索: 正在运行 Lambda = {current_lambda} ---")
            net_alpha_cv, net_beta_cv, a_curr_t, b_curr_t, nn_cv_success = \
                run_networknet_k_iterations(
                    current_lambda, k_iterations, n_features, fixed_nn_params, 
                    learning_rate, n_epochs_per_step,
                    X_t, y_v, r_idx, c_idx, a_mle_t, b_mle_t, device,
                    desc_prefix=f"Run {sim_id+1} IC-Search L={current_lambda}"
                )
            
            if nn_cv_success:
                with torch.no_grad():
                    est_alpha_cv_t = a_curr_t; est_beta_cv_t = b_curr_t
                    sel_alpha_cv_idx = net_alpha_cv.get_selected_features()
                    sel_beta_cv_idx = net_beta_cv.get_selected_features()
                    n_sel_total = len(sel_alpha_cv_idx) + len(sel_beta_cv_idx)
                    log_lambda_v = est_alpha_cv_t[r_idx] + est_beta_cv_t[c_idx]
                    
                    ic_dict = calculate_information_criteria(
                        y_v, log_lambda_v, n_edges, n_sel_total, n_features 
                    )
                    
                    ic_search_results.append({
                        'lambda': current_lambda,
                        'nll': ic_dict['nll'],
                        'aic': ic_dict['aic'],
                        'bic': ic_dict['bic'],
                        'hbic': ic_dict['hbic'],
                        'n_selected': ic_dict['n_selected'],
                        'est_alpha_np': est_alpha_cv_t.cpu().numpy(),
                        'est_beta_np': est_beta_cv_t.cpu().numpy(),
                        'sel_alpha_idx': sel_alpha_cv_idx,
                        'sel_beta_idx': sel_beta_cv_idx
                    })
        
        if ic_search_results:
            ic_search_df = pd.DataFrame(ic_search_results).set_index('lambda')
            
            # *** MONITOR: Print IC Table ***
            print(f"\n--- [Run {sim_id+1}] 信息准则 (IC) 搜索路径结果 ---")
            print(ic_search_df[['nll', 'n_selected', 'aic', 'bic', 'hbic']].to_string(float_format="%.2f"))
            
            best_lambda_nll = ic_search_df['nll'].idxmin()
            best_lambda_aic = ic_search_df['aic'].idxmin()
            best_lambda_bic = ic_search_df['bic'].idxmin()
            best_lambda_hbic = ic_search_df['hbic'].idxmin()
            
            # *** MONITOR: Print IC Selections ***
            print(f"  Best Lambda (NLL):  {best_lambda_nll:.4f} (n_sel={ic_search_df.loc[best_lambda_nll]['n_selected']:.0f})")
            print(f"  Best Lambda (AIC):  {best_lambda_aic:.4f} (n_sel={ic_search_df.loc[best_lambda_aic]['n_selected']:.0f})")
            print(f"  Best Lambda (BIC):  {best_lambda_bic:.4f} (n_sel={ic_search_df.loc[best_lambda_bic]['n_selected']:.0f})")
            print(f"  Best Lambda (HBIC): {best_lambda_hbic:.4f} (n_sel={ic_search_df.loc[best_lambda_hbic]['n_selected']:.0f})")
            print("-"*(len(f"  Best Lambda (HBIC): {best_lambda_hbic:.4f} (n_sel={ic_search_df.loc[best_lambda_hbic]['n_selected']:.0f})")+2))
            
            best_entry_ic = ic_search_df.loc[best_lambda_hbic].to_dict()
            
            alpha_eval = evaluate_alpha_results(true_alpha, best_entry_ic['est_alpha_np'], true_idx_alpha, best_entry_ic['sel_alpha_idx'])
            all_final_metrics.update({f"{k}_NetworkNetIC": v for k, v in alpha_eval.items()})
            beta_eval = evaluate_beta_results(true_beta, best_entry_ic['est_beta_np'], true_idx_beta, best_entry_ic['sel_beta_idx'])
            all_final_metrics.update({f"{k}_NetworkNetIC": v for k, v in beta_eval.items()})
            all_final_metrics['best_lambda_AIC'] = best_lambda_aic
            all_final_metrics['best_lambda_BIC'] = best_lambda_bic
            all_final_metrics['best_lambda_HBIC'] = best_lambda_hbic
        
        else:
            raise ValueError("NetworkNetIC path finding failed (all lambdas failed).")

    except Exception as e:
        print(f"[{sim_id+1}] Error NetworkNetIC: {e}"); traceback.print_exc()
        for metric in ['rmse_alpha','n_sel_alpha','prec_alpha','rec_alpha','f1_alpha','rmse_beta','n_sel_beta','prec_beta','rec_beta','f1_beta', 'best_lambda_AIC', 'best_lambda_BIC', 'best_lambda_HBIC']: 
            all_final_metrics[f"{metric}_NetworkNetIC"] = np.nan
            
    # =======================================================
    #     METHOD 2: MLELasso (LassoCV)
    # =======================================================
    # *** MONITOR: Print step ***
    print(f"--- [Run {sim_id+1}] 4. 方法 2: MLELasso (LassoCV) ---")
    try:
        if alpha_mle is not None and beta_mle is not None:
            lasso_alpha = LassoCV(cv=5, random_state=current_seed_run, n_jobs=1, max_iter=10000, tol=5e-4, n_alphas=100)
            X_cont = np.ascontiguousarray(X); lasso_alpha.fit(X_cont, alpha_mle)
            est_alpha_lasso = lasso_alpha.predict(X_cont); sel_alpha_lasso = np.where(np.abs(lasso_alpha.coef_) > 1e-6)[0]
            alpha_eval = evaluate_alpha_results(true_alpha, est_alpha_lasso, true_idx_alpha, sel_alpha_lasso); all_final_metrics.update({f"{k}_MLELasso": v for k, v in alpha_eval.items()})

            lasso_beta = LassoCV(cv=5, random_state=current_seed_run, n_jobs=1, max_iter=10000, tol=5e-4, n_alphas=100)
            lasso_beta.fit(X_cont, beta_mle)
            est_beta_lasso = lasso_beta.predict(X_cont); sel_beta_lasso = np.where(np.abs(lasso_beta.coef_) > 1e-6)[0]
            beta_eval = evaluate_beta_results(true_beta, est_beta_lasso, true_idx_beta, sel_beta_lasso); all_final_metrics.update({f"{k}_MLELasso": v for k, v in beta_eval.items()})
        else:
            for metric in ['rmse_alpha','n_sel_alpha','prec_alpha','rec_alpha','f1_alpha','rmse_beta','n_sel_beta','prec_beta','rec_beta','f1_beta']: all_final_metrics[f"{metric}_MLELasso"] = np.nan
    except Exception as e:
        print(f"[{sim_id+1}] Error MLELasso: {e}"); traceback.print_exc()
        for metric in ['rmse_alpha','n_sel_alpha','prec_alpha','rec_alpha','f1_alpha','rmse_beta','n_sel_beta','prec_beta','rec_beta','f1_beta']: all_final_metrics[f"{metric}_MLELasso"] = np.nan

    # ===============================================================================================
    #     METHOD 3: MLELassoNet (Internal Implementation)
    # ===============================================================================================
    # *** MONITOR: Print step ***
    print(f"--- [Run {sim_id+1}] 5. 方法 3: MLELassoNet (BIC 路径) ---")
    mleln_success = False
    try:
        if alpha_mle is not None and beta_mle is not None:
            lambda_start_ln = 0.5; lambda_multiplier_ln = 1.2; max_lambda_ln = 5.0
            n_epochs_ln_step = 50; lr_ln = 0.0005

            # --- Fit Alpha Path ---
            print(f"--- [Run {sim_id+1}]    MLELassoNet: 正在拟合 Alpha 路径 ---")
            path_results_alpha = []
            current_lambda_ln = lambda_start_ln
            net_ln_alpha = CountNetworkLassoNet(input_dim=n_features, hidden_dims=fixed_nn_params['hidden_dims'], dropout=fixed_nn_params['dropout']).to(device)
            optimizer_ln_alpha = optim.SGD(net_ln_alpha.parameters(), lr=lr_ln, momentum=0.9)
            alpha_mle_t = torch.FloatTensor(alpha_mle).to(device)
            while current_lambda_ln < max_lambda_ln:
                success = train_mle_lassonet_step(net_ln_alpha, optimizer_ln_alpha, X_t, alpha_mle_t, n_epochs=n_epochs_ln_step, lr=lr_ln, lambda_penalty=current_lambda_ln, M=fixed_nn_params['M'], desc=f"MLELN Alpha L={current_lambda_ln:.3f}")
                if not success: break
                with torch.no_grad(): net_ln_alpha.eval(); pred_alpha = net_ln_alpha(X_t).squeeze().cpu().numpy()
                sel_alpha = net_ln_alpha.get_selected_features()
                bic_alpha = calculate_bic_mse(alpha_mle, pred_alpha, n_samples, len(sel_alpha))
                path_results_alpha.append({'lambda': current_lambda_ln, 'bic': bic_alpha, 'n_sel': len(sel_alpha), 'state_dict': {k: v.detach().clone().cpu() for k, v in net_ln_alpha.state_dict().items()}})
                if len(sel_alpha) == 0 and current_lambda_ln > lambda_start_ln: break
                current_lambda_ln *= lambda_multiplier_ln
            if path_results_alpha:
                best_entry_alpha = min(path_results_alpha, key=lambda x: x['bic'])
                best_net_ln_alpha = CountNetworkLassoNet(input_dim=n_features, hidden_dims=fixed_nn_params['hidden_dims'], dropout=fixed_nn_params['dropout']).to(device)
                best_net_ln_alpha.load_state_dict({k: v.to(device) for k, v in best_entry_alpha['state_dict'].items()})
                with torch.no_grad(): best_net_ln_alpha.eval(); est_alpha_ln = best_net_ln_alpha(X_t).squeeze().cpu().numpy()
                sel_alpha_ln = best_net_ln_alpha.get_selected_features()
                alpha_eval = evaluate_alpha_results(true_alpha, est_alpha_ln, true_idx_alpha, sel_alpha_ln); all_final_metrics.update({f"{k}_MLELassoNet": v for k, v in alpha_eval.items()})
            else: raise ValueError("MLELN Alpha path failed.")

            # --- Fit Beta Path ---
            print(f"--- [Run {sim_id+1}]    MLELassoNet: 正在拟合 Beta 路径 ---")
            path_results_beta = []
            current_lambda_ln = lambda_start_ln
            net_ln_beta = CountNetworkLassoNet(input_dim=n_features, hidden_dims=fixed_nn_params['hidden_dims'], dropout=fixed_nn_params['dropout']).to(device)
            optimizer_ln_beta = optim.SGD(net_ln_beta.parameters(), lr=lr_ln, momentum=0.9)
            beta_mle_t = torch.FloatTensor(beta_mle).to(device)
            while current_lambda_ln < max_lambda_ln:
                success = train_mle_lassonet_step(net_ln_beta, optimizer_ln_beta, X_t, beta_mle_t, n_epochs=n_epochs_ln_step, lr=lr_ln, lambda_penalty=current_lambda_ln, M=fixed_nn_params['M'], desc=f"MLELN Beta L={current_lambda_ln:.3f}")
                if not success: break
                with torch.no_grad(): net_ln_beta.eval(); pred_beta = net_ln_beta(X_t).squeeze().cpu().numpy()
                sel_beta = net_ln_beta.get_selected_features()
                bic_beta = calculate_bic_mse(beta_mle, pred_beta, n_samples, len(sel_beta))
                path_results_beta.append({'lambda': current_lambda_ln, 'bic': bic_beta, 'n_sel': len(sel_beta), 'state_dict': {k: v.detach().clone().cpu() for k, v in net_ln_beta.state_dict().items()}})
                if len(sel_beta) == 0 and current_lambda_ln > lambda_start_ln: break
                current_lambda_ln *= lambda_multiplier_ln
            if path_results_beta:
                best_entry_beta = min(path_results_beta, key=lambda x: x['bic'])
                best_net_ln_beta = CountNetworkLassoNet(input_dim=n_features, hidden_dims=fixed_nn_params['hidden_dims'], dropout=fixed_nn_params['dropout']).to(device)
                best_net_ln_beta.load_state_dict({k: v.to(device) for k, v in best_entry_beta['state_dict'].items()})
                with torch.no_grad(): best_net_ln_beta.eval(); est_beta_ln = best_net_ln_beta(X_t).squeeze().cpu().numpy()
                sel_beta_ln = best_net_ln_beta.get_selected_features()
                beta_eval = evaluate_beta_results(true_beta, est_beta_ln, true_idx_beta, sel_beta_ln); all_final_metrics.update({f"{k}_MLELassoNet": v for k, v in beta_eval.items()})
                mleln_success = True
            else: raise ValueError("MLELN Beta path failed.")
        else: pass 
    except Exception as e:
        print(f"[{sim_id+1}] Error MLELassoNet: {e}"); traceback.print_exc()
        mleln_success = False
    if not mleln_success:
        for metric in ['rmse_alpha','n_sel_alpha','prec_alpha','rec_alpha','f1_alpha','rmse_beta','n_sel_beta','prec_beta','rec_beta','f1_beta']: all_final_metrics[f"{metric}_MLELassoNet"] = np.nan
        
    # =======================================================
    #     *** NEW *** METHOD 4: NetworkNetFixed (L=0.1)
    # =======================================================
    # *** MONITOR: Print step ***
    print(f"--- [Run {sim_id+1}] 6. 方法 4: NetworkNetFixed (L={lambda_fixed_alpha}) ---")
    try:
        net_alpha_fixed, net_beta_fixed, a_curr_t, b_curr_t, nn_fixed_success = \
            run_networknet_k_iterations(
                lambda_fixed_alpha, k_iterations, n_features, fixed_nn_params, 
                learning_rate, n_epochs_per_step,
                X_t, y_v, r_idx, c_idx, a_mle_t, b_mle_t, device,
                desc_prefix=f"Run {sim_id+1} Fixed L={lambda_fixed_alpha}"
            )
        
        if nn_fixed_success:
            est_alpha_fixed_np = a_curr_t.cpu().numpy()
            est_beta_fixed_np = b_curr_t.cpu().numpy()
            sel_alpha_fixed_idx = net_alpha_fixed.get_selected_features()
            sel_beta_fixed_idx = net_beta_fixed.get_selected_features()
            
            alpha_eval = evaluate_alpha_results(true_alpha, est_alpha_fixed_np, true_idx_alpha, sel_alpha_fixed_idx)
            all_final_metrics.update({f"{k}_NetworkNetFixed": v for k, v in alpha_eval.items()})
            beta_eval = evaluate_beta_results(true_beta, est_beta_fixed_np, true_idx_beta, sel_beta_fixed_idx)
            all_final_metrics.update({f"{k}_NetworkNetFixed": v for k, v in beta_eval.items()})
        else:
             raise ValueError("NetworkNetFixed training failed.")
            
    except Exception as e:
        print(f"[{sim_id+1}] Error NetworkNetFixed: {e}"); traceback.print_exc()
        for metric in ['rmse_alpha','n_sel_alpha','prec_alpha','rec_alpha','f1_alpha','rmse_beta','n_sel_beta','prec_beta','rec_beta','f1_beta']: 
            all_final_metrics[f"{metric}_NetworkNetFixed"] = np.nan

    # *** MONITOR: Print run end ***
    print(f"--- [Run {sim_id+1}] 运行完成, 正在返回指标 ---")
    return all_final_metrics


# ---------------------------------------------------------------------------- #
#                          *** Main Execution Block *** #
#           (已移除 'if __name__ == __main__':'以便在Jupyter中运行)
# ---------------------------------------------------------------------------- #

print("="*80)
print("脚本已加载。开始执行主模拟...")
print("="*80)

# --- Simulation Setup ---
N_SIMULATIONS = 100 # Repeat 3 times sequentially
K_ITERATIONS = 2  # K=2 iterations for NetworkNet
SEED = 19210406 # Base seed

# --- Device Setup ---
device = "cuda"
print(f"开始模拟. 次数: {N_SIMULATIONS}, K={K_ITERATIONS}, 设备: {device.upper()}")

# --- Parameter Setting (User provided - NONLINEAR) ---
dgp_params = {
    'relationship_type': 'linear', # <<< NONLINEAR
    'n_samples': 100,
    'n_features': 50,
    'signal_strength': 2
}

# --- Fixed & Fit Params (User Provided) ---
fixed_nn_params = {
    'hidden_dims': (64, ),
    'dropout': 0.0,
    'gamma_balance': 1.0, # Used by NetworkNet
    'M': 1              # User updated M to 20
}
fit_nn_params = {
    # Lambdas are now found by search, so these are not used for final model
    'lambda_beta': 0.15,         # Not used in OracleK5, but placeholder
    'lambda_alpha': 0.15,        # Not used in OracleK5, but placeholder
    'lr': 0.005,                # For NetworkNet PGD steps
    'n_epochs_per_step': 2000,  # User updated epochs
    'mle_lr': 0.005,            # User updated MLE LR
    'mle_epochs': 2000          # User updated MLE epochs
}

all_results_list = [] # Store results from all runs

# --- Sequential Simulation Loop ---
print(f"\n--- Running Setting: Type={dgp_params['relationship_type']}, n={dgp_params['n_samples']}, p={dgp_params['n_features']} ---")

# *** 启用 TQDM.NOTEBOOK 
for sim_id in tqdm(range(N_SIMULATIONS), desc="Overall Simulation Progress"):
    current_seed = SEED + sim_id
    np.random.seed(current_seed); torch.manual_seed(current_seed)
    if device == 'cuda': torch.cuda.manual_seed_all(current_seed) # Seed GPU if used

    result = run_single_simulation_with_benchmarks(
                     sim_id=sim_id, dgp_params=dgp_params,
                     fixed_nn_params=fixed_nn_params, fit_nn_params=fit_nn_params,
                     k_iterations=K_ITERATIONS, device=device
                 )
    if result: all_results_list.append(result)
    else: print(f"Warning: Simulation run {sim_id+1} failed.")
    results_df = pd.DataFrame(all_results_list)
    summary = results_df.agg(['mean', 'std']).T
    summary['std'] = summary['std'].fillna(0)
    summary['mean (std)'] = summary.apply(lambda row: f"{row['mean']:.4f} ({row['std']:.4f})" if pd.notna(row['mean']) else "N/A", axis=1)
    final_summary = summary[['mean (std)']]
    
    metric_order = [
        'rmse_alpha_mle', 'rmse_beta_mle',
        'rmse_alpha_NetworkNetIC', 'n_sel_alpha_NetworkNetIC', 'prec_alpha_NetworkNetIC', 'rec_alpha_NetworkNetIC', 'f1_alpha_NetworkNetIC',
        'rmse_beta_NetworkNetIC', 'n_sel_beta_NetworkNetIC', 'prec_beta_NetworkNetIC', 'rec_beta_NetworkNetIC', 'f1_beta_NetworkNetIC',
        'best_lambda_AIC', 'best_lambda_BIC', 'best_lambda_HBIC',
        'rmse_alpha_NetworkNetFixed', 'n_sel_alpha_NetworkNetFixed', 'prec_alpha_NetworkNetFixed', 'rec_alpha_NetworkNetFixed', 'f1_alpha_NetworkNetFixed',
        'rmse_beta_NetworkNetFixed', 'n_sel_beta_NetworkNetFixed', 'prec_beta_NetworkNetFixed', 'rec_beta_NetworkNetFixed', 'f1_beta_NetworkNetFixed',
        'rmse_alpha_MLELasso', 'n_sel_alpha_MLELasso', 'prec_alpha_MLELasso', 'rec_alpha_MLELasso', 'f1_alpha_MLELasso',
        'rmse_beta_MLELasso', 'n_sel_beta_MLELasso', 'prec_beta_MLELasso', 'rec_beta_MLELasso', 'f1_beta_MLELasso',
        'rmse_alpha_MLELassoNet', 'n_sel_alpha_MLELassoNet', 'prec_alpha_MLELassoNet', 'rec_alpha_MLELassoNet', 'f1_alpha_MLELassoNet',
        'rmse_beta_MLELassoNet', 'n_sel_beta_MLELassoNet', 'prec_beta_MLELassoNet', 'rec_beta_MLELassoNet', 'f1_beta_MLELassoNet',
    ]
    available_metrics = final_summary.index.tolist()
    final_order = [m for m in metric_order if m in available_metrics] + [m for m in available_metrics if m not in metric_order]
    final_summary = final_summary.reindex(final_order).fillna("N/A")
    
    pd.set_option('display.max_rows', None); pd.set_option('display.width', 200)
    print("\n\n" + "="*100); print(f"         最终模拟结果汇总 (NetworkNetIC Search, {dgp_params['relationship_type']}, K={K_ITERATIONS}, N={N_SIMULATIONS}, n={dgp_params['n_samples']}, p={dgp_params['n_features']})"); print("="*100)
    print(final_summary)

# --- Aggregate and Save Final Results ---
print("\n--- 所有模拟运行完毕, 正在聚合结果... ---")

if not all_results_list:
    print("\nSimulation complete, but NO valid results were collected.")
else:
    results_df = pd.DataFrame(all_results_list)
    summary = results_df.agg(['mean', 'std']).T
    summary['std'] = summary['std'].fillna(0)
    summary['mean (std)'] = summary.apply(lambda row: f"{row['mean']:.4f} ({row['std']:.4f})" if pd.notna(row['mean']) else "N/A", axis=1)
    final_summary = summary[['mean (std)']]
    
    metric_order = [
        'rmse_alpha_mle', 'rmse_beta_mle',
        'rmse_alpha_NetworkNetIC', 'n_sel_alpha_NetworkNetIC', 'prec_alpha_NetworkNetIC', 'rec_alpha_NetworkNetIC', 'f1_alpha_NetworkNetIC',
        'rmse_beta_NetworkNetIC', 'n_sel_beta_NetworkNetIC', 'prec_beta_NetworkNetIC', 'rec_beta_NetworkNetIC', 'f1_beta_NetworkNetIC',
        'best_lambda_AIC', 'best_lambda_BIC', 'best_lambda_HBIC',
        'rmse_alpha_NetworkNetFixed', 'n_sel_alpha_NetworkNetFixed', 'prec_alpha_NetworkNetFixed', 'rec_alpha_NetworkNetFixed', 'f1_alpha_NetworkNetFixed',
        'rmse_beta_NetworkNetFixed', 'n_sel_beta_NetworkNetFixed', 'prec_beta_NetworkNetFixed', 'rec_beta_NetworkNetFixed', 'f1_beta_NetworkNetFixed',
        'rmse_alpha_MLELasso', 'n_sel_alpha_MLELasso', 'prec_alpha_MLELasso', 'rec_alpha_MLELasso', 'f1_alpha_MLELasso',
        'rmse_beta_MLELasso', 'n_sel_beta_MLELasso', 'prec_beta_MLELasso', 'rec_beta_MLELasso', 'f1_beta_MLELasso',
        'rmse_alpha_MLELassoNet', 'n_sel_alpha_MLELassoNet', 'prec_alpha_MLELassoNet', 'rec_alpha_MLELassoNet', 'f1_alpha_MLELassoNet',
        'rmse_beta_MLELassoNet', 'n_sel_beta_MLELassoNet', 'prec_beta_MLELassoNet', 'rec_beta_MLELassoNet', 'f1_beta_MLELassoNet',
    ]
    available_metrics = final_summary.index.tolist()
    final_order = [m for m in metric_order if m in available_metrics] + [m for m in available_metrics if m not in metric_order]
    final_summary = final_summary.reindex(final_order).fillna("N/A")
    
    pd.set_option('display.max_rows', None); pd.set_option('display.width', 200)
    print("\n\n" + "="*100); print(f"         最终模拟结果汇总 (NetworkNetIC Search, {dgp_params['relationship_type']}, K={K_ITERATIONS}, N={N_SIMULATIONS}, n={dgp_params['n_samples']}, p={dgp_params['n_features']})"); print("="*100)
    print(final_summary)
    
    output_filename = f"NEWBigNetsimulation_summary_Linear_{dgp_params['relationship_type']}_k{K_ITERATIONS}_n{dgp_params['n_samples']}_p{dgp_params['n_features']}_R{N_SIMULATIONS}_NetworkNetIC_Search.csv"
    final_summary.to_csv(output_filename, encoding='utf-8-sig')
    print(f"\n汇总表格已保存至: {output_filename}")

print("\n脚本执行完毕。")

