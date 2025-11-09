export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \


TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
BASE_DIR="/home/zhou/wzt/real_rl/ckpt"
CKPT_DIR="$BASE_DIR/run_$TIMESTAMP"

python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --actor \
    --checkpoint_path="$CKPT_DIR" \


# python ../../train_rlpd.py "$@" \
#     --exp_name=pick_cube_sim \
#     --checkpoint_path=first_run \
#     --actor \