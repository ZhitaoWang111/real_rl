export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
source /opt/ros/noetic/setup.bash
source /home/zhou/pika_ros/install/setup.bash && \
python ../../train_rlpd.py "$@" \
    --exp_name=pick_cube_sim \
    --demo_path=/home/zhou/wzt/real_rl/examples/experiments/pick_cube_sim/demo_data/pick_cube_sim_2_demos_2025-11-10_13-13-06.pkl \
    --learner \

