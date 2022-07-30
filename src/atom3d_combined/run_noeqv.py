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

from tasks.lba.train_lba import train_noneqv_lba
train_noneqv_lba = ex.capture(train_noneqv_lba)
from tasks.lep.train_lep import train_noneqv_lep
train_noneqv_lep = ex.capture(train_noneqv_lep)
from tasks.msp.train_msp import train_noneqv_msp
train_noneqv_msp = ex.capture(train_noneqv_msp)
from tasks.ppi.train_ppi import train_noneqv_ppi
train_noneqv_ppi = ex.capture(train_noneqv_ppi)
from tasks.psr.train_psr import train_noneqv_psr
train_noneqv_psr = ex.capture(train_noneqv_psr)
from tasks.res.train_res import train_noneqv_res
train_noneqv_res = ex.capture(train_noneqv_res)
from tasks.rsr.train_rsr import train_noneqv_rsr
train_noneqv_rsr = ex.capture(train_noneqv_rsr)


TASKS = ['ppi', 'lba', 'lep', 'msp', 'rsr', 'res', 'psr']
NOT_SUPPORTED_TASKS = ['smp', 'mam', 'scn']
MODELS = ['gert_noeqv']
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
PROT_ATOMS = ('C', 'O', 'N', 'S', 'P', 'H')

# Choose the configuration we want
@ex.config
def config():
    task = 'lba'
    model = 'gert_noeqv'
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
    batch_size = 8
    hidden_dim = 64
    learning_rate = 1e-4
    workers = 0
    betas = (0.9, 0.999)
    eps = 1e-8
    max_train_iter = 256000
    max_test_iter = 64000
    print_frequency = 1000
    num_heads = 4
    d_ff = 64
    d_atom = len(PROT_ATOMS) + 1
    eta = 0.4
    max_radius = 10.0
    num_atoms = len(PROT_ATOMS) + 1

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

    fnname = 'train_noneqv_{}'.format(task)
    print(fnname)
    eval(fnname + '(ex, test_mode={})'.format(test_mode))