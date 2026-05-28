## training

```bash
CUDA_VISIBLE_DEVICES=0 /home/lsy/.conda/envs/panda_rl/bin/python -u train/train_sac.py   --env FrankaPickAndPlaceWindowInsertSparse-v0   --timesteps 1000000   --exp-name insert_posealign_noher_residual_seed1   --seed 1   --device cuda   --no-her   --dense-reward-shaping   --dense-reward-style insert   --task-progress-features   --residual-guidance   --residual-action-scale 0.015   --expert-demo-episodes 64   --expert-demo-style insert   --expert-hint-style insert   --bc-epochs 30   --demo-prefill-passes 8   --learning-starts 0   --action-noise-std 0.01   --ent-coef auto_0.02   --target-entropy -0.2   --min-ent-coef 0.001   --batch-size 512   --gradient-steps 8   --n-envs 1   --no-normalize-env
```

## evaluate with video

```bash
CUDA_VISIBLE_DEVICES=0 /home/lsy/.conda/envs/panda_rl/bin/python -u evaluate/evaluate_with_video.py   --exp-dir outputs/insert_posealign_noher_residual_seed1_expertfix   --model-name best_model   --episodes 50   --seed-start 0   --env FrankaPickAndPlaceWindowInsertSparse-v0   --task-progress-features   --residual-guidance   --expert-hint-style insert   --residual-action-scale 0.015   --no-vecnormalize   --no-stop-on-success   --max-steps 200   --device cuda
```
```bash
CUDA_VISIBLE_DEVICES=0 /home/lsy/.conda/envs/panda_rl/bin/python -u evaluate/evaluate_with_video.py   --model-path outputs/insert_posealign_noher_residual_seed1_rewardfix/models/checkpoints/sac_FrankaPickAndPlaceWindowInsertSparse-v0_600000_steps.zip   --episodes 50   --seed-start 0   --env FrankaPickAndPlaceWindowInsertSparse-v0   --task-progress-features   --residual-guidance   --expert-hint-style insert   --residual-action-scale 0.015   --no-vecnormalize   --max-steps 200   --device cuda
```