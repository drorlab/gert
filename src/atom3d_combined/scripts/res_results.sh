# equivariant attention
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=True num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=2 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=True num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=2 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=True num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=2 seed=42
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=True num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=3 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=True num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=3 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=True num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=3 seed=42
sbatch sbatches/ppi.sbatch python3 run.py with task=res use_attention=True num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=4 seed=40
sbatch sbatches/ppi.sbatch python3 run.py with task=res use_attention=True num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=4 seed=41
sbatch sbatches/ppi.sbatch python3 run.py with task=res use_attention=True num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=4 seed=42

# equivariant no attention
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=2 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=2 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=2 seed=42
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=3 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=3 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=3 seed=42
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=4 seed=40
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=4 seed=41
sbatch sbatches/exp.sbatch python3 run.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 lmax=4 seed=42

# non-equivariant, attention
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=res use_attention=True num_heads=1 num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=res use_attention=True num_heads=1 num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=res use_attention=True num_heads=1 num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 seed=42
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=res use_attention=True num_heads=4 num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=res use_attention=True num_heads=4 num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=res use_attention=True num_heads=4 num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 seed=42

# non-equivariant, no attention
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 seed=40
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 seed=41
sbatch sbatches/exp.sbatch python3 run_noeqv.py with task=res use_attention=False num_epochs=3 max_radius=10.0 max_train_iter=256000 max_test_iter=64000 seed=42