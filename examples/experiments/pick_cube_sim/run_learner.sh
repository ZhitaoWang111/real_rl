export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=first_run \
    --demo_path=/home/wzt/wzt/mycode/hil-serl-sim/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_2_demos_2025-11-08_17-00-25.pkl \
    --learner \