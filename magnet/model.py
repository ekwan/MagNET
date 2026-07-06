import torch
import pytorch_lightning as pl
from .eqV2 import EquiformerV2_NMR

class MagNET_Lightning(pl.LightningModule):
    def __init__(self, params, precision = 32, filter_solvent_edges = None, solvent_edge_radius = None):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.precision = precision
        
        # if filter_solvent_edges and solvent_edge_radius are not provided as arguments, read from params file
            # this is convenient, because we don't need to constantly provide filter_solvent_edges and solvent_edge_radius
            # as arguments if they're already specified in params, while still allowing us to override their previous 
            # values if we need to (e.g., when loading in a pretrained checkpoint)
        if filter_solvent_edges is None:
            assert solvent_edge_radius is None
            filter_solvent_edges = False
            solvent_edge_radius = params['max_radius']
            if 'filter_solvent_edges' in params:
                filter_solvent_edges = params['filter_solvent_edges']
            if 'solvent_edge_radius' in params:
                solvent_edge_radius = params['solvent_edge_radius']
        self.filter_solvent_edges = filter_solvent_edges
        self.solvent_edge_radius = solvent_edge_radius
        
        self.lr = params['lr']
            
        if 'min_lr' in params:
            self.min_lr = params['min_lr']
        else:
            self.min_lr = self.lr
        
        # number of exponential decay steps (= # batches) from self.lr to self.min_lr
        if 'lr_steps' in params:
            self.lr_steps = params['lr_steps']
        else:
            self.lr_steps = 300000 
        
        self.model_type = params['model_type'] # in ['EquiformerV2']
        
        evidential_regression = False
        if 'evidential_regression' in params:
            evidential_regression = params['evidential_regression']
        self.evidential_regression = evidential_regression
        
        if self.model_type == 'EquiformerV2':
            assert self.precision == 32
            
            self.model = EquiformerV2_NMR(
                None, None, None,
                use_pbc=False, regress_forces=False,
                
                otf_graph = True,
                max_neighbors = params['max_neighbors'],
                max_radius = params['max_radius'],
                max_num_elements = params['max_num_elements'],
            
                num_layers = params['num_layers'], 
                sphere_channels = params['sphere_channels'],
                attn_hidden_channels = params['attn_hidden_channels'],
                num_heads = params['num_heads'], 
                attn_alpha_channels = params['attn_alpha_channels'],
                attn_value_channels = params['attn_value_channels'],
                ffn_hidden_channels = params['ffn_hidden_channels'],
                
                norm_type='layer_norm_sh',
                
                lmax_list = params['lmax_list'],
                mmax_list = params['mmax_list'],
                grid_resolution = params['grid_resolution'], 
            
                num_sphere_samples=128,
                
                edge_channels=128,
                use_atom_edge_embedding=True,
                share_atom_edge_embedding=False,
                use_m_share_rad=False,
                distance_function="gaussian",
                num_distance_basis=512, # not used; hard-coded by Equiformer-V2 to 600
            
                attn_activation='silu',
                use_s2_act_attn=False, 
                use_attn_renorm=True,
                ffn_activation='silu',
                use_gate_act=False,
                use_grid_mlp=True, 
                use_sep_s2_act=True,
            
                alpha_drop=0.0,
                drop_path_rate=0.0, 
                proj_drop=0.0, 
            
                weight_init = params['weight_init'],
                
                evidential_regression = self.evidential_regression,
                filter_solvent_edges = self.filter_solvent_edges,
                solvent_edge_radius = self.solvent_edge_radius,
            )
    
    def forward(self, data):
        if 'atomic_numbers' not in data.keys():
            data.atomic_numbers = data.x 
        if 'natoms' not in data.keys():
            data.natoms = torch.unique_consecutive(data.batch, return_counts = True)[1]
        
        if self.model_type == 'EquiformerV2':
            if self.evidential_regression:
                mu,v,alpha,beta = self.model(data).squeeze(1).chunk(4, dim = 1)
                v = torch.nn.functional.softplus(v) + 1e-8
                alpha = torch.nn.functional.softplus(alpha) + 1. + 1e-8
                beta = torch.nn.functional.softplus(beta) + 1e-8
                return torch.cat([mu,v,alpha,beta], dim = 1)
            else:
                return self.model(data).squeeze(1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        
        # exponential lr decay from self.lr to self.min_lr in self.lr_steps steps
        gamma = (self.min_lr / self.lr) ** (1.0 / self.lr_steps)
        func = lambda step: max(gamma**(step), self.min_lr / self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = func)
        
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": False,
            "name": None,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch
        data.atomic_numbers = data.x 
        data.natoms = torch.unique_consecutive(data.batch, return_counts = True)[1] 

        out = self.forward(data)
        
        if self.precision == 64:
            y = data.shieldings.double()
        else:
            y = data.shieldings
            
        if self.evidential_regression:
            lam = 0.2
            mu,v,alpha,beta = out.chunk(4, dim = 1)
            mu = mu.squeeze(dim=1)[data.atom_type_mask]
            v = v.squeeze(dim=1)[data.atom_type_mask]
            alpha = alpha.squeeze(dim=1)[data.atom_type_mask]
            beta = beta.squeeze(dim=1)[data.atom_type_mask]
            
            loss = 0.5 * (torch.log(torch.pi / v)) \
                   - alpha * torch.log(2.*beta*(1. + v)) \
                   + (alpha + 0.5) * torch.log(v*(y[data.atom_type_mask] - mu)**2. + 2.*beta*(1. + v)) \
                   + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) \
                   + lam * (torch.abs(y[data.atom_type_mask] - mu) * (2. * v + alpha) - 1e-4)
            loss = loss.mean()
        else:
            loss = torch.mean(((out.squeeze(1) - y)[data.atom_type_mask])**2.0)
        
        self.log('train_loss', loss, batch_size = data.atom_type_mask.sum().item())
        
        # does not account for symmetric atoms 
        return loss

    def validation_step(self, val_batch, batch_idx):
        data = val_batch
        data.atomic_numbers = data.x 
        data.natoms = torch.unique_consecutive(data.batch, return_counts = True)[1] 
        
        out = self.forward(data)
        
        if self.precision == 64:
            y = data.shieldings.double()
        else:
            y = data.shieldings
        
        if self.evidential_regression:
            mu,v,alpha,beta = out.chunk(4, dim = 1)
            mu = mu.squeeze(dim=1)[data.atom_type_mask]
            v = v.squeeze(dim=1)[data.atom_type_mask]
            alpha = alpha.squeeze(dim=1)[data.atom_type_mask]
            beta = beta.squeeze(dim=1)[data.atom_type_mask]
            
            lam = 0.2
            loss = 0.5 * (torch.log(torch.pi / v)) \
                   - alpha * torch.log(2.*beta*(1. + v)) \
                   + (alpha + 0.5) * torch.log(v*(y[data.atom_type_mask] - mu)**2. + 2.*beta*(1. + v)) \
                   + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) \
                   + lam * (torch.abs(y[data.atom_type_mask] - mu) * (2. * v + alpha) - 1e-4)
            loss = loss.mean()
            mae = torch.mean(torch.abs(mu - y[data.atom_type_mask]))
        else:
            loss = torch.mean(((out.squeeze(1) - y)[data.atom_type_mask])**2.0)
            mae = torch.mean(torch.abs(((out.squeeze(1) - y)[data.atom_type_mask])))
            
        self.log_dict({"val_loss": loss, "val_mae": mae}, batch_size = data.atom_type_mask.sum().item())
        return loss
    
    def test_step(self, test_batch, batch_idx):
        data = test_batch
        data.atomic_numbers = data.x 
        data.natoms = torch.unique_consecutive(data.batch, return_counts = True)[1] 
        
        out = self.forward(data)
        
        if self.precision == 64:
            y = data.shieldings.double()
        else:
            y = data.shieldings
        
        if self.evidential_regression:
            lam = 0.2
            mu,v,alpha,beta = out.chunk(4, dim = 1)
            mu = mu.squeeze(dim=1)[data.atom_type_mask]
            v = v.squeeze(dim=1)[data.atom_type_mask]
            alpha = alpha.squeeze(dim=1)[data.atom_type_mask]
            beta = beta.squeeze(dim=1)[data.atom_type_mask]
            
            loss = 0.5 * (torch.log(torch.pi / v)) \
                   - alpha * torch.log(2.*beta*(1. + v)) \
                   + (alpha + 0.5) * torch.log(v*(y[data.atom_type_mask] - mu)**2. + 2.*beta*(1. + v)) \
                   + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5) \
                   + lam * (torch.abs(y[data.atom_type_mask] - mu) * (2. * v + alpha) - 1e-4)
            loss = loss.mean()
            mae = torch.mean(torch.abs(mu - y[data.atom_type_mask]))
        else:
            loss = torch.mean(((out.squeeze(1) - y)[data.atom_type_mask])**2.0)
            mae = torch.mean(torch.abs(((out.squeeze(1) - y)[data.atom_type_mask])))

        self.log_dict({"test_loss": loss, "test_mae": mae}, batch_size = data.atom_type_mask.sum().item())
        return loss
