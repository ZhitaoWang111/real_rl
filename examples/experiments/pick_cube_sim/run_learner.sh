export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6 && \


TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
BASE_DIR="/home/zhou/wzt/real_rl/ckpt"
CKPT_DIR="$BASE_DIR/run_$TIMESTAMP"

# CKPT_DIR="/home/zhou/wzt/real_rl/ckpt/run_2025-11-14_11-17-29"


source /opt/ros/noetic/setup.bash
source /home/zhou/pika_ros/install/setup.bash && \
python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --demo_path=/home/zhou/wzt/real_rl/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_20_demos_2025-11-14_15-52-10.pkl \
    --checkpoint_path=$CKPT_DIR \
    --learner \