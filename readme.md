## training

# expert hint + residual
```bash
CUDA_VISIBLE_DEVICES=0 /home/lsy/.conda/envs/panda_rl/bin/python -u train/train_sac.py   --env FrankaPickAndPlaceWindowInsertSparse-v0   --timesteps 1000000   --exp-name insert_posealign_noher_residual_seed1   --seed 1   --device cuda   --no-her   --dense-reward-shaping   --dense-reward-style insert   --task-progress-features   --residual-guidance   --residual-action-scale 0.015   --expert-demo-episodes 64   --expert-demo-style insert   --expert-hint-style insert   --bc-epochs 30   --demo-prefill-passes 8   --learning-starts 0   --action-noise-std 0.01   --ent-coef auto_0.02   --target-entropy -0.2   --min-ent-coef 0.001   --batch-size 512   --gradient-steps 8   --n-envs 1   --no-normalize-env
```

# warm-start only, no runtime expert
```bash
CUDA_VISIBLE_DEVICES=0 /home/lsy/.conda/envs/panda_rl/bin/python -u train/train_sac.py   --env FrankaPickAndPlaceWindowInsertSparse-v0   --timesteps 1000000   --exp-name insert_warmstart_only_rewardfix_seed1   --seed 1   --device cuda   --no-her   --dense-reward-shaping   --dense-reward-style insert   --task-geometry-features   --no-task-progress-features   --no-residual-guidance   --expert-warmstart-only   --expert-demo-episodes 512   --expert-demo-style insert   --expert-hint-style insert   --bc-epochs 200   --demo-prefill-passes 32   --learning-starts 0   --action-noise-std 0.01   --ent-coef auto_0.02   --target-entropy -0.2   --min-ent-coef 0.001   --batch-size 512   --gradient-steps 8   --n-envs 4   --no-normalize-env
```

## evaluate with video

# expert hint + residual
```bash
CUDA_VISIBLE_DEVICES=0 /home/lsy/.conda/envs/panda_rl/bin/python -u evaluate/evaluate_with_video.py   --exp-dir outputs/insert_posealign_noher_residual_seed1_expertfix   --model-name best_model   --episodes 50   --seed-start 0   --env FrankaPickAndPlaceWindowInsertSparse-v0   --task-progress-features   --residual-guidance   --expert-hint-style insert   --residual-action-scale 0.015   --no-vecnormalize   --no-stop-on-success   --max-steps 200   --device cuda
```
```bash
CUDA_VISIBLE_DEVICES=0 /home/lsy/.conda/envs/panda_rl/bin/python -u evaluate/evaluate_with_video.py   --model-path outputs/insert_posealign_noher_residual_seed1_rewardfix/models/checkpoints/sac_FrankaPickAndPlaceWindowInsertSparse-v0_600000_steps.zip   --episodes 50   --seed-start 0   --env FrankaPickAndPlaceWindowInsertSparse-v0   --task-progress-features   --residual-guidance   --expert-hint-style insert   --residual-action-scale 0.015   --no-vecnormalize   --max-steps 200   --device cuda
```

# warm-start only, no runtime expert
```bash
CUDA_VISIBLE_DEVICES=0 /home/lsy/.conda/envs/panda_rl/bin/python -u evaluate/evaluate_with_video.py   --exp-dir outputs/insert_warmstart_only_seed1   --model-name best_model   --episodes 10   --seed-start 0   --env FrankaPickAndPlaceWindowInsertSparse-v0   --task-geometry-features   --no-task-progress-features   --no-residual-guidance   --expert-warmstart-only   --expert-hint-style insert   --no-vecnormalize   --no-stop-on-success   --max-steps 200   --device cuda
```

```bash
MUJOCO_GL=egl \
/home/lsy/.conda/envs/panda_rl/bin/python \
  evaluate/evaluate_window_assembly_pipeline.py \
  --model outputs/window_fine_align/sac-20260721-205122-seed0/models/checkpoints/window_fine_align_sac_150000_steps.zip \
  --episodes 5 \
  --seed 0 \
  --deterministic \
  --record-video \
  --output-dir outputs/window_assembly_visualization/model1500
```

```bash
MUJOCO_GL=egl \
python capture_stage_snapshots.py \
  --model outputs/window_fine_align/sac-20260721-205122-seed0/models/checkpoints/window_fine_align_sac_150000_steps.zip \
  --seed 0 \
  --output-dir outputs/stage_snapshots_seed0
```