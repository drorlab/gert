# equivariant attention
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=True num_epochs=40 lmax=2 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=True num_epochs=40 lmax=2 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=True num_epochs=40 lmax=2 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=42
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=True num_epochs=40 lmax=3 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=True num_epochs=40 lmax=3 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=True num_epochs=40 lmax=3 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=42
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=True num_epochs=40 lmax=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=True num_epochs=40 lmax=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=True num_epochs=40 lmax=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=42

# equivariant no attention
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=False num_epochs=40 lmax=2 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=False num_epochs=40 lmax=2 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=False num_epochs=40 lmax=2 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=42
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=False num_epochs=40 lmax=3 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=False num_epochs=40 lmax=3 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=False num_epochs=40 lmax=3 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=42
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=False num_epochs=40 lmax=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=False num_epochs=40 lmax=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=lep use_attention=False num_epochs=40 lmax=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=42

# non-equivariant, attention
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=lep use_attention=True num_epochs=40 num_heads=1 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=lep use_attention=True num_epochs=40 num_heads=1 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=lep use_attention=True num_epochs=40 num_heads=1 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=42
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=lep use_attention=True num_epochs=40 num_heads=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=lep use_attention=True num_epochs=40 num_heads=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=lep use_attention=True num_epochs=40 num_heads=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=42

# non-equivariant, no attention
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=lep use_attention=False num_epochs=40 num_heads=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=lep use_attention=False num_epochs=40 num_heads=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=lep use_attention=False num_epochs=40 num_heads=4 max_radius=8.0 max_train_iter=256000 max_test_iter=64000 seed=42