export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \


TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
BASE_DIR="/home/zhou/wzt/real_rl/ckpt"
CKPT_DIR="$BASE_DIR/run_$TIMESTAMP"

# CKPT_DIR="/home/zhou/wzt/real_rl/ckpt/run_2025-11-14_11-17-29"

python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --actor \
    --checkpoint_path=$CKPT_DIR \
    # --eval_checkpoint_step=50000 \
    # --eval_n_trajs=10 \


# python ../../train_rlpd.py "$@" \
#     --exp_name=pick_cube_sim \
#     --checkpoint_path=first_run \
#     --actor \