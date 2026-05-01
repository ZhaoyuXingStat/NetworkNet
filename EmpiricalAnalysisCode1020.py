import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import time
import re
import random

warnings.filterwarnings('ignore')

def estimate_mle_params(Y: np.ndarray, n_epochs: int = 1000, lr: float = 0.05, verbose: bool = False, device='cpu') -> tuple[np.ndarray, np.ndarray]:
    n = Y.shape[0]
    Y_tensor = torch.FloatTensor(Y).to(device)
    alpha = nn.Parameter(torch.zeros(n, 1, device=device))
    beta = nn.Parameter(torch.zeros(n, 1, device=device))
    optimizer = optim.Adam([alpha, beta], lr=lr)
    if verbose: print(f"  [MLE Init] Starting MLE on {device}...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        with torch.no_grad(): beta.data -= torch.mean(beta.data)
        lambda_ij = torch.exp(alpha.T + beta)
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        loss = torch.sum((lambda_ij - Y_tensor * (alpha.T + beta)) * mask)
        if torch.isnan(loss): break
        loss.backward()
        optimizer.step()
    if verbose: print("  [MLE Init] MLE finished.")
    return alpha.data.cpu().numpy().flatten(), beta.data.cpu().numpy().flatten()

class CountNetworkLassoNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(32, 16), dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.skip = nn.Linear(input_dim, 1, bias=False)
        self.first_layer = nn.Linear(input_dim, hidden_dims[0])
        layers = [self.first_layer, nn.ReLU(), nn.Dropout(dropout)]
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1, bias=False))
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.skip(x) + self.network(x)
    def l1_regularization(self): return torch.norm(self.skip.weight, p=1)
    def get_selected_features(self, threshold=1e-4):
        importance = torch.abs(self.skip.weight.data).squeeze().cpu().numpy()
        if importance.ndim == 0: importance = np.array([importance])
        return np.where(importance > threshold)[0]

def _hierarchical_prox(model: nn.Module, lambda_: float, step_size: float, M: int):
    with torch.no_grad():
        theta, W1 = model.skip.weight.data, model.first_layer.weight.data
        for j in range(model.input_dim):
            theta_j, w_j = theta[0, j].item(), W1[:, j]
            threshold = lambda_ * step_size
            if abs(theta_j) <= threshold:
                theta[0, j], W1[:, j] = 0.0, 0.0
            else:
                theta[0, j] = torch.sign(theta[0, j]) * (abs(theta_j) - threshold)
                max_w = torch.max(torch.abs(w_j))
                if max_w > M * abs(theta[0, j]): W1[:, j] *= (M * abs(theta[0, j]) / max_w)

class AlternatingCountNetworkModel:
    def __init__(self, input_dim: int, hidden_dims: tuple, lambda_l1_alpha: float,
                 lambda_l1_beta: float, gamma_balance: float, M: int, device: str):
        self.input_dim, self.device, self.M, self.gamma_balance = input_dim, device, M, gamma_balance
        self.net_alpha = CountNetworkLassoNet(input_dim, hidden_dims).to(device)
        self.net_beta = CountNetworkLassoNet(input_dim, hidden_dims).to(device)
        self.lambda_l1_alpha, self.lambda_l1_beta = lambda_l1_alpha, lambda_l1_beta
        self.alpha_estimates_, self.beta_estimates_ = None, None
    def _prepare_edge_data(self, Y: torch.Tensor):
        n = Y.shape[0]
        r_idx = torch.arange(n, device=self.device).repeat_interleave(n)
        c_idx = torch.arange(n, device=self.device).repeat(n)
        off_diag = r_idx != c_idx
        return r_idx[off_diag], c_idx[off_diag], Y.flatten()[off_diag]
    def _update_alpha_network(self, X_t, y_v, r_idx, c_idx, fixed_beta, n_epochs, lr):
        opt = optim.Adam(self.net_alpha.parameters(), lr=lr)
        beta_e, sum_beta = fixed_beta[c_idx], torch.sum(fixed_beta)
        for _ in range(n_epochs):
            self.net_alpha.train()
            opt.zero_grad()
            alpha_all = self.net_alpha(X_t).squeeze()
            alpha_e = alpha_all[r_idx]
            log_lambda = alpha_e + beta_e
            p_loss = torch.mean(torch.exp(log_lambda) - y_v * log_lambda)
            l1 = self.lambda_l1_alpha * self.net_alpha.l1_regularization()
            bal = self.gamma_balance * (torch.sum(alpha_all) - sum_beta)**2
            loss = p_loss + l1 + bal
            loss.backward()
            opt.step()
            _hierarchical_prox(self.net_alpha, self.lambda_l1_alpha, lr, self.M)
    def _update_beta_network(self, X_t, y_v, r_idx, c_idx, fixed_alpha, n_epochs, lr):
        opt = optim.Adam(self.net_beta.parameters(), lr=lr)
        alpha_e, sum_alpha = fixed_alpha[r_idx], torch.sum(fixed_alpha)
        for _ in range(n_epochs):
            self.net_beta.train()
            opt.zero_grad()
            beta_all = self.net_beta(X_t).squeeze()
            beta_e = beta_all[c_idx]
            log_lambda = alpha_e + beta_e
            p_loss = torch.mean(torch.exp(log_lambda) - y_v * log_lambda)
            l1 = self.lambda_l1_beta * self.net_beta.l1_regularization()
            bal = self.gamma_balance * (sum_alpha - torch.sum(beta_all))**2
            loss = p_loss + l1 + bal
            loss.backward()
            opt.step()
            _hierarchical_prox(self.net_beta, self.lambda_l1_beta, lr, self.M)
    def fit(self, Y, X, alpha_init=None, beta_init=None, max_iters=10, n_epochs_per_iter=20, lr=0.01, tol=1e-4, verbose=False):
        Y_t, X_t = torch.FloatTensor(Y).to(self.device), torch.FloatTensor(X).to(self.device)
        r_idx, c_idx, y_v = self._prepare_edge_data(Y_t)
        a_curr, b_curr = (torch.FloatTensor(alpha_init).to(self.device), torch.FloatTensor(beta_init).to(self.device))
        for it in range(max_iters):
            a_old = a_curr.clone()
            self._update_beta_network(X_t, y_v, r_idx, c_idx, a_curr, n_epochs_per_iter, lr)
            with torch.no_grad(): b_curr = self.net_beta(X_t).squeeze()
            b_old = b_curr.clone()
            self._update_alpha_network(X_t, y_v, r_idx, c_idx, b_curr, n_epochs_per_iter, lr)
            with torch.no_grad(): a_curr = self.net_alpha(X_t).squeeze()
            a_chg = torch.norm(a_curr - a_old)/(torch.norm(a_old) + 1e-8)
            b_chg = torch.norm(b_curr - b_old)/(torch.norm(b_old) + 1e-8)
            if verbose and (it+1) % 5 == 0: print(f"  Iter {it+1}: Rel. Change (α, β) = ({a_chg:.2e}, {b_chg:.2e})")
            if a_chg < tol and b_chg < tol: break
        self.alpha_estimates_, self.beta_estimates_ = a_curr.cpu().detach().numpy(), b_curr.cpu().detach().numpy()
        return self
    def predict(self, X):
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            self.net_alpha.eval()
            self.net_beta.eval()
        return self.net_alpha(X_t).squeeze().cpu().detach().numpy(), self.net_beta(X_t).squeeze().cpu().detach().numpy()
    def get_selected_features(self):
        return self.net_alpha.get_selected_features(), self.net_beta.get_selected_features()

print("="*80)
print("1. 加载数据")
print("="*80)
try:
    features_df = pd.read_csv("paper_info_8090Author-Keywords-Features.csv", index_col=0)
    edges_df = pd.read_csv("paper_info_8090Countvalued_Author_author_CitationNet.txt", sep=" ")
except FileNotFoundError as e:
    print(f"错误: {e}\n请确保数据文件在当前目录")
    exit()

if 'Author_uni_ID' not in features_df.columns:
    features_df.reset_index(inplace=True)
    features_df.rename(columns={features_df.columns[0]: 'Author_uni_ID'}, inplace=True)

authors_list = features_df['Author_uni_ID'].unique().tolist()
n_nodes = len(authors_list)
author_to_idx = {author_id: i for i, author_id in enumerate(authors_list)}
print(f"节点数: {n_nodes}")

numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
feature_cols = [col for col in numeric_cols if col != 'Author_uni_ID']
n_features = len(feature_cols)
print(f"特征维度: {n_features}")

X_df = features_df.set_index('Author_uni_ID').reindex(authors_list)[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X_df.to_numpy(dtype=np.float32))

print("\n构建邻接矩阵...")
Y = np.zeros((n_nodes, n_nodes), dtype=np.float32)
edges_df = edges_df[edges_df['from'].isin(authors_list) & edges_df['to'].isin(authors_list)]
for _, row in tqdm(edges_df.iterrows(), total=edges_df.shape[0], desc="构建网络"):
    from_idx, to_idx = author_to_idx.get(row['from']), author_to_idx.get(row['to'])
    if from_idx is not None and to_idx is not None: Y[from_idx, to_idx] = row['weight']

def calculate_marginal_effects(model, X_df_unscaled, scaler_obj, feature_cols, selected_features_map, parameter_type):
    if not selected_features_map:
        return {}, {}
    
    print(f"  Calculating marginal effects for {parameter_type}...")
    marginal_effects_pos = {}
    marginal_effects_neg = {}
    
    X_scaled = scaler_obj.transform(X_df_unscaled.to_numpy(dtype=np.float32))
    original_alpha, original_beta = model.predict(X_scaled)
    base_prediction = original_alpha if parameter_type == 'alpha' else original_beta
    
    for feature_name in tqdm(selected_features_map, desc=f"Marginal Effects ({parameter_type})"):
        non_zero_mask = X_df_unscaled[feature_name] > 0
        if not non_zero_mask.any(): 
            marginal_effects_pos[feature_name] = 0
            marginal_effects_neg[feature_name] = 0
            continue

        X_df_pos = X_df_unscaled.copy()
        X_df_pos.loc[non_zero_mask, feature_name] += 1
        X_pos_scaled = scaler_obj.transform(X_df_pos.to_numpy(dtype=np.float32)) 
        pos_alpha, pos_beta = model.predict(X_pos_scaled)
        pos_prediction = pos_alpha if parameter_type == 'alpha' else pos_beta
        avg_pos_effect = (pos_prediction - base_prediction)[non_zero_mask].mean()
        marginal_effects_pos[feature_name] = avg_pos_effect
        
        X_df_neg = X_df_unscaled.copy()
        X_df_neg.loc[non_zero_mask, feature_name] -= 1
        X_df_neg[feature_name] = X_df_neg[feature_name].clip(lower=0) 
        X_neg_scaled = scaler_obj.transform(X_df_neg.to_numpy(dtype=np.float32)) 
        neg_alpha, neg_beta = model.predict(X_neg_scaled)
        neg_prediction = neg_alpha if parameter_type == 'alpha' else neg_beta
        avg_neg_effect = (neg_prediction - base_prediction)[non_zero_mask].mean()
        marginal_effects_neg[feature_name] = avg_neg_effect
        
    return marginal_effects_pos, marginal_effects_neg

def find_best_lambda_1d(parameter_type, target_k, fixed_lambda_other, fixed_model_params, fit_params, Y, X):
    other_type = 'beta' if parameter_type == 'alpha' else 'alpha'
    print(f"    -- 搜索 '{parameter_type}' (固定 {other_type} = {fixed_lambda_other:.4f})...")
    
    lambda_min, lambda_max = 0.01, 10.0
    best_lambda, min_dist = 0.01, float('inf')
    n_sel = 0

    for i in range(15):
        lambda_mid = (lambda_min + lambda_max) / 2
        
        l1_a = lambda_mid if parameter_type == 'alpha' else fixed_lambda_other
        l1_b = lambda_mid if parameter_type == 'beta' else fixed_lambda_other
        
        model = AlternatingCountNetworkModel(lambda_l1_alpha=l1_a, lambda_l1_beta=l1_b, **fixed_model_params)
        model.fit(Y, X, **fit_params, verbose=False)
        
        sel_a, sel_b = model.get_selected_features()
        n_sel = len(sel_a) if parameter_type == 'alpha' else len(sel_b)
        dist = abs(n_sel - target_k)
        
        if dist < min_dist:
            min_dist, best_lambda = dist, lambda_mid
        
        if n_sel > target_k + 1:
            lambda_min = lambda_mid
        else:
            lambda_max = lambda_mid
            
        if abs(n_sel - target_k) <= 1:
            break
        if (lambda_max - lambda_min) < 1e-4:
            break
            
    print(f"    -- '{parameter_type}' 搜索完成. 最佳 Lambda: {best_lambda:.4f} (选中 {n_sel} 个)")
    return best_lambda

def run_analysis_for_architecture(config_tuple, Y_data, X_data, X_df_unscaled_data, scaler_obj, feature_cols_list, device):
    name, config = config_tuple
    print(f"开始处理架构: {name} on device: {device}")
    
    TARGET_K = 20
    
    fixed_model_params = {'input_dim': X_data.shape[1], 'hidden_dims': config, 'gamma_balance': 1.0, 'M': 10, 'device': device}
    fit_params = {'max_iters': 2, 'n_epochs_per_iter': 2000, 'lr': 0.05, 'tol': 1e-4}

    alpha_mle, beta_mle = estimate_mle_params(Y_data, device=device, verbose=False)
    fit_params['alpha_init'], fit_params['beta_init'] = alpha_mle, beta_mle
    
    print(f"  为 {name} 开始交替搜索 Lambda (目标: {TARGET_K} 个特征)...")
    la, lb = 0.1, 0.1
    la_prev, lb_prev = 0.0, 0.0
    
    for search_iter in range(10):
        print(f"  -- 交替搜索第 {search_iter + 1} 轮:")
        la = find_best_lambda_1d('alpha', TARGET_K, lb, fixed_model_params, fit_params, Y_data, X_data)
        lb = find_best_lambda_1d('beta', TARGET_K, la, fixed_model_params, fit_params, Y_data, X_data)
        
        if abs(la - la_prev) < 1e-3 and abs(lb - lb_prev) < 1e-3:
            print(f"  -- Lambda 搜索收敛于: (Alpha: {la:.4f}, Beta: {lb:.4f})")
            break
        la_prev, lb_prev = la, lb
        
    best_lambda_alpha = la
    best_lambda_beta = lb
    print(f"  最终 Lambda 组合: (Alpha: {best_lambda_alpha:.4f}, Beta: {best_lambda_beta:.4f})")
    
    print(f"  为 {name} 训练最终模型...")
    final_model = AlternatingCountNetworkModel(lambda_l1_alpha=best_lambda_alpha, lambda_l1_beta=best_lambda_beta, **fixed_model_params)
    final_model.fit(Y_data, X_data, **fit_params, verbose=False)
    
    sel_alpha_idx, sel_beta_idx = final_model.get_selected_features()
    
    alpha_features_to_eval = [feature_cols_list[i] for i in sel_alpha_idx]
    beta_features_to_eval = [feature_cols_list[i] for i in sel_beta_idx]
    
    me_pos_a, me_neg_a = calculate_marginal_effects(final_model, X_df_unscaled_data, scaler_obj, feature_cols_list, alpha_features_to_eval, 'alpha')
    me_pos_b, me_neg_b = calculate_marginal_effects(final_model, X_df_unscaled_data, scaler_obj, feature_cols_list, beta_features_to_eval, 'beta')

    alpha_df = pd.DataFrame({'Feature_Name': alpha_features_to_eval})
    if not alpha_df.empty:
        alpha_df['Marginal_Effect_Pos'] = alpha_df['Feature_Name'].map(me_pos_a)
        alpha_df['Marginal_Effect_Neg'] = alpha_df['Feature_Name'].map(me_neg_a)
        alpha_df['Sort_Key'] = alpha_df['Marginal_Effect_Pos'].abs()
        alpha_df = alpha_df.sort_values(by='Sort_Key', ascending=False).drop(columns='Sort_Key').head(TARGET_K)
    
    beta_df = pd.DataFrame({'Feature_Name': beta_features_to_eval})
    if not beta_df.empty:
        beta_df['Marginal_Effect_Pos'] = beta_df['Feature_Name'].map(me_pos_b)
        beta_df['Marginal_Effect_Neg'] = beta_df['Feature_Name'].map(me_neg_b)
        beta_df['Sort_Key'] = beta_df['Marginal_Effect_Pos'].abs()
        beta_df = beta_df.sort_values(by='Sort_Key', ascending=False).drop(columns='Sort_Key').head(TARGET_K)
    
    return name, {'alpha': alpha_df, 'beta': beta_df}

if __name__ == '__main__':
    print("\n" + "="*80)
    print("3. 串行对比不同神经网络架构")
    print("="*80)
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f"检测到 CUDA 设备。将在 GPU 上串行处理 3 个架构...")
    else:
        DEVICE = 'cpu'
        print(f"未检测到 CUDA。将在 CPU 上串行处理 3 个架构...")
    
    hidden_dims_options = {
        "Small_Wide (64, 32)": (64, 32),
        "Deep_Narrow (32, 16, 8, 8)": (32, 16, 16, 16),
        "Wide_Shallow (128, 64)": (128, 64)
    }
    
    random.seed(2025)
    print("开始串行处理...")
    for name, config in hidden_dims_options.items():
        
        print("\n" + "="*80)
        print(f"开始处理架构: {name}")
        print("="*80)
        
        arch_name, result_dict = run_analysis_for_architecture(
            (name, config), 
            Y, X, X_df, scaler, feature_cols, DEVICE
        )
        
        print(f"\n--- 正在格式化和保存 {name} 的结果 ---")
        model_df = pd.DataFrame({'Rank': range(1, 21)})
        
        alpha_df = result_dict['alpha'].reset_index(drop=True).reindex(range(20)).fillna('')
        beta_df = result_dict['beta'].reset_index(drop=True).reindex(range(20)).fillna('')
        
        model_df[f'Alpha_Feature'] = alpha_df.apply(
            lambda row: row['Feature_Name'] if row['Feature_Name'] != '' else '', axis=1)
        model_df[f'Alpha_ME_Pos'] = alpha_df.apply(
            lambda row: row['Marginal_Effect_Pos'] if row['Feature_Name'] != '' else '', axis=1)
        model_df[f'Alpha_ME_Neg'] = alpha_df.apply(
            lambda row: row['Marginal_Effect_Neg'] if row['Feature_Name'] != '' else '', axis=1)
        
        model_df[f'Beta_Feature'] = beta_df.apply(
            lambda row: row['Feature_Name'] if row['Feature_Name'] != '' else '', axis=1)
        model_df[f'Beta_ME_Pos'] = beta_df.apply(
            lambda row: row['Marginal_Effect_Pos'] if row['Feature_Name'] != '' else '', axis=1)
        model_df[f'Beta_ME_Neg'] = beta_df.apply(
            lambda row: row['Marginal_Effect_Neg'] if row['Feature_Name'] != '' else '', axis=1)
        model_df = model_df.set_index('Rank')

        safe_name = re.sub(r'[^\w\d_()]+', '_', name).strip('_')
        output_filename = f"1020NEW_results_{safe_name}.csv"
        
        model_df.to_csv(output_filename, encoding='utf-8-sig')
        print(f"结果已保存到: {output_filename}")

        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', 1000)
        print(f"\n--- {name} 结果预览 (Top 20) ---")
        print(model_df.to_string())

    print("\n" + "="*80)
    print("所有架构处理完毕。")
    print("="*80)