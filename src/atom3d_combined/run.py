import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import datetime
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore",
                        message="Setting attributes on ParameterList is not supported.")

from sacred import Experiment
from sacred.observers import FileStorageObserver
ex = Experiment()
ex.observers.append(FileStorageObserver('unified_dev'))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tasks.lba.train_eqv_lba import train_eqv_lba
train_eqv_lba = ex.capture(train_eqv_lba)
from tasks.lep.train_eqv_lep import train_eqv_lep
train_eqv_lep = ex.capture(train_eqv_lep)
from tasks.msp.train_eqv_msp import train_eqv_msp
train_eqv_msp = ex.capture(train_eqv_msp)
from tasks.ppi.train_eqv_ppi import train_eqv_ppi
train_eqv_ppi = ex.capture(train_eqv_ppi)
from tasks.psr.train_eqv_psr import train_eqv_psr
train_eqv_psr = ex.capture(train_eqv_psr)
from tasks.res.train_eqv_res import train_eqv_res
train_eqv_res = ex.capture(train_eqv_res)
from tasks.rsr.train_eqv_rsr import train_eqv_rsr
train_eqv_rsr = ex.capture(train_eqv_rsr)


TASKS = ['ppi', 'lba', 'lep', 'msp', 'rsr', 'res', 'psr']
NOT_SUPPORTED_TASKS = ['smp', 'mam', 'scn']
MODELS = ['gert']
MODES = ['train', 'test']
TASK_TO_DATA = {
    'ppi': {
        'data_dir': '/oak/stanford/groups/rondror/projects/atom3d/protein_protein_interfaces/DIPS-split/',
        'labels_dir': '',
    },
    'lba': {
        'data_dir': '/oak/stanford/groups/rondror/projects/atom3d/ligand_binding_affinity/lba-split-by-sequence-identity-30',
        'labels_dir': '/oak/stanford/groups/rondror/projects/atom3d/supporting_files/ligand_binding_affinity/pdbbind_files/pdbbind_refined_set_labels.csv',
    },
    'lep': {
        'data_dir': '/oak/stanford/groups/rondror/projects/atom3d/ligand_efficacy_prediction/split/',
        'labels_dir': '',
    },
    'msp': {
        'data_dir': '/oak/stanford/groups/rondror/projects/atom3d/mutation_stability_prediction/split/',
        'labels_dir': '',
    },
    'psr': {
        'data_dir': '/oak/stanford/groups/rondror/projects/atom3d/protein_structure_ranking/decoy-split',
        'labels_dir': '',
    },
    'res': {
        'data_dir': '/oak/stanford/groups/rondror/projects/atom3d/residue_identity/environments-split/',
        'labels_dir': '',
    },
    'rsr': {
        'data_dir': '/oak/stanford/groups/rondror/projects/atom3d/rna_structure_ranking/candidates-split',
        'labels_dir': '',
    },
}

# Choose the configuration we want
@ex.config
def config():
    task = 'lba'
    model = 'gert'
    use_attention = True
    mode = 'test'
    seed = 40
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')
    data_dir = TASK_TO_DATA[task]['data_dir']
    labels_dir = TASK_TO_DATA[task]['labels_dir']
    log_dir = os.path.join('logs', model, task, mode, now)
    checkpoint = None
    note = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 5
    batch_size = 1
    lmax = 3
    num_heads = 1
    N_hidden = 1
    learning_rate = 1e-4
    workers = 0
    betas = (0.9, 0.999)
    eps = 1e-8
    max_train_iter = 256000
    max_test_iter = 64000
    print_frequency = 1000
    num_dense = 1
    encoder_layers = 1
    max_radius = 10.0
    number_of_basis = 15
    h = 100
    L = 1


@ex.automain
def run(task, model, mode, seed, log_dir, use_attention):
    assert task in TASKS
    assert model in MODELS
    assert mode in MODES

    print('Task: {} \tModel: {} \tAttention: {} \tMode: {} \tSeed: {}'.format(task, model, use_attention, mode, seed))

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    test_mode = (mode == 'test')

    fnname = 'train_eqv_{}'.format(task)
    print(fnname)
    eval(fnname + '(ex, test_mode={})'.format(test_mode))