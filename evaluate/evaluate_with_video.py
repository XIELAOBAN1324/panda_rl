#!/usr/bin/env python3
"""
모델 평가 및 비디오 생성 통합 스크립트
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List
from stable_baselines3.common.evaluation import evaluate_policy

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import panda_mujoco_gym

from utils import (
    load_model, save_json, get_experiment_info,
    find_latest_experiment, create_vec_env,
)
from video_recorder import StageVideoRecorder


def _predict_action(model, obs, env):
    if model is None:
        raw_action = env.action_space.sample()
        return np.array([raw_action])
    action, _ = model.predict(obs, deterministic=True)
    return action


def evaluate_model_performance(model, env, num_episodes: int = 50) -> Dict:
    """모델 또는 랜덤 정책 성능 평가.

    evaluate_policy()는 성공률이나 개별 rollout 통계를 직접 제공하지 않으므로,
    여기서는 모델/랜덤 정책 모두 동일한 수동 rollout 경로로 평가한다.
    """
    policy_name = '랜덤 정책' if model is None else '모델'
    print(f"\n📊 {policy_name} 성능 평가 중... ({num_episodes}개 에피소드)")

    rewards = []
    lengths = []
    success_count = 0

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        length = 0

        while not done:
            action = _predict_action(model, obs, env)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            total_reward += float(reward[0])
            length += 1

            if done and infos[0].get('is_success', False):
                success_count += 1

        rewards.append(total_reward)
        lengths.append(length)

    results = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'mean_length': float(np.mean(lengths)),
        'success_rate': success_count / num_episodes,
        'num_episodes': num_episodes,
        'all_rewards': [float(r) for r in rewards],
        'all_lengths': [int(l) for l in lengths],
        'rollout_mean_reward': float(np.mean(rewards)),
        'rollout_mean_length': float(np.mean(lengths)),
    }

    print(f"   평균 보상: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   성공률: {results['success_rate']:.3f}")
    print(f"   평균 에피소드 길이: {results['mean_length']:.1f}")

    return results


def _get_vecnormalize_path(models_dir: str, model_name: str):
    stage_vecnorm = os.path.join(models_dir, f"{model_name}_vecnormalize.pkl")
    final_vecnorm = os.path.join(models_dir, 'vec_normalize.pkl')
    if os.path.exists(stage_vecnorm):
        return stage_vecnorm
    if os.path.exists(final_vecnorm):
        return final_vecnorm
    return None


def _build_stage_model_list(available_models: List[str], stages: List[str] = None) -> List[str]:
    if stages is None or 'all' in stages:
        selected = [m for m in available_models if m.startswith('stage_')]
        if 'best_model' in available_models:
            selected.append('best_model')
        if 'final_model' in available_models:
            selected.append('final_model')
    else:
        selected = []
        for stage in stages:
            normalized_stage = stage.lower()
            if normalized_stage == 'best' and 'best_model' in available_models:
                selected.append('best_model')
                continue
            if normalized_stage == 'final' and 'final_model' in available_models:
                selected.append('final_model')
                continue

            model_name = stage if stage.startswith('stage_') else f'stage_{stage}'
            if model_name in available_models:
                selected.append(model_name)

    def sort_key(name: str):
        if name == 'best_model':
            return (998, name)
        if name == 'final_model':
            return (999, name)
        if name.startswith('stage_'):
            stage_label = name.replace('stage_', '', 1)
            try:
                stage_num = int(stage_label.split('_', 1)[0])
            except ValueError:
                stage_num = 997
            return (stage_num, name)
        return (997, name)

    return sorted(dict.fromkeys(selected), key=sort_key)


def evaluate_experiment(
    exp_dir: str,
    stages: List[str] = None,
    num_eval_episodes: int = 50,
    num_video_episodes: int = 3,
    record_video: bool = True,
    create_highlights: bool = True,
) -> Dict:
    print('=' * 60)
    print('🔍 실험 평가 시작')
    print(f'📁 실험 디렉토리: {exp_dir}')
    print('=' * 60)

    exp_info = get_experiment_info(exp_dir)
    if not exp_info:
        raise ValueError(f'실험 정보를 찾을 수 없습니다: {exp_dir}')

    env_name = exp_info.get('env_name', 'FrankaSlideDense-v0')
    algorithm = exp_info.get('algorithm', 'SAC')
    reward_scale = exp_info.get('config', {}).get('reward_scale', 0.1)

    print('\n📋 실험 정보:')
    print(f'   환경: {env_name}')
    print(f'   알고리즘: {algorithm}')
    print(f'   보상 스케일: {reward_scale}')
    print(f"   총 학습 스텝: {exp_info.get('total_timesteps', 'Unknown')}")

    models_dir = os.path.join(exp_dir, 'models')
    available_models = exp_info.get('available_models', [])
    if not available_models:
        raise ValueError(f'모델을 찾을 수 없습니다: {models_dir}')

    stage_models = _build_stage_model_list(available_models, stages)
    print(f'\n📦 평가할 모델 ({len(stage_models)}개):')
    for model_name in stage_models:
        print(f'   - {model_name}')

    evaluation_dir = os.path.join(exp_dir, 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)

    video_recorder = None
    if record_video:
        videos_dir = os.path.join(evaluation_dir, 'videos')
        video_recorder = StageVideoRecorder(videos_dir, fps=10)

    all_results = {
        'experiment_info': exp_info,
        'evaluation_config': {
            'num_eval_episodes': num_eval_episodes,
            'num_video_episodes': num_video_episodes,
            'stages_evaluated': stage_models,
        },
        'model_performances': {},
        'video_recordings': {},
    }

    for model_name in stage_models:
        print(f"\n{'=' * 50}")
        print(f'🎯 평가 중: {model_name}')
        print(f"{'=' * 50}")

        model_path = os.path.join(models_dir, f'{model_name}.zip')
        vec_normalize_path = _get_vecnormalize_path(models_dir, model_name)

        eval_env = create_vec_env(
            env_name,
            n_envs=1,
            normalize=True,
            reward_scale=reward_scale,
            vec_normalize_path=vec_normalize_path,
            training=False,
            render_mode=None,
        )

        if model_name == 'stage_0_random':
            model = None
            print('   🎲 Random Policy 사용')
        else:
            model = load_model(model_path, algorithm=algorithm, env=eval_env)
            print('   ✅ 모델 로드 완료')
            if vec_normalize_path:
                print(f'   📦 정규화 통계: {os.path.basename(vec_normalize_path)}')

        performance = evaluate_model_performance(model, eval_env, num_eval_episodes)
        all_results['model_performances'][model_name] = performance
        eval_env.close()

        if record_video and video_recorder is not None:
            video_env = create_vec_env(
                env_name,
                n_envs=1,
                normalize=True,
                reward_scale=reward_scale,
                vec_normalize_path=vec_normalize_path,
                training=False,
                render_mode='rgb_array',
            )
            stage_name = model_name.replace('stage_', '') if model_name.startswith('stage_') else model_name
            video_results = video_recorder.record_stage_episodes(
                stage_name, model, video_env, num_video_episodes
            )
            all_results['video_recordings'][model_name] = {
                'num_videos': len(video_results),
                'episodes': video_results,
            }
            video_env.close()

    if record_video and create_highlights and video_recorder is not None:
        video_recorder.create_highlight_reel()

    if record_video and video_recorder is not None:
        video_summary = video_recorder.save_evaluation_summary()
        all_results['video_summary'] = video_summary

    results_path = os.path.join(evaluation_dir, 'evaluation_results.json')
    save_json(all_results, results_path)

    print('\n' + '=' * 60)
    print('✅ 평가 완료!')
    print(f'📊 결과 저장: {results_path}')
    if record_video:
        print(f"🎥 비디오 저장: {os.path.join(evaluation_dir, 'videos')}")
    print('=' * 60)
    return all_results


def main():
    parser = argparse.ArgumentParser(description='모델 평가 및 비디오 생성')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exp-dir', type=str, help='실험 디렉토리 경로')
    group.add_argument('--latest', action='store_true', help='가장 최근 실험 평가')

    parser.add_argument('--stages', type=str, nargs='+', default=None,
                        help="평가할 단계 (예: 0_random 1_20percent final) 또는 'all'")
    parser.add_argument('--num-eval', type=int, default=50,
                        help='성능 평가 에피소드 수 (기본: 50)')
    parser.add_argument('--num-video', type=int, default=3,
                        help='각 단계별 비디오 에피소드 수 (기본: 3)')
    parser.add_argument('--no-video', action='store_true',
                        help='비디오 녹화 비활성화')
    parser.add_argument('--no-highlights', action='store_true',
                        help='하이라이트 생성 비활성화')

    args = parser.parse_args()

    if args.latest:
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            print('❌ 최근 실험을 찾을 수 없습니다.')
            return
        print(f'📁 최신 실험 발견: {exp_dir}')
    else:
        exp_dir = args.exp_dir
        if not os.path.exists(exp_dir):
            print(f'❌ 실험 디렉토리를 찾을 수 없습니다: {exp_dir}')
            return

    try:
        results = evaluate_experiment(
            exp_dir=exp_dir,
            stages=args.stages,
            num_eval_episodes=args.num_eval,
            num_video_episodes=args.num_video,
            record_video=not args.no_video,
            create_highlights=not args.no_highlights,
        )

        print('\n📈 평가 결과 요약:')
        for model_name, perf in results['model_performances'].items():
            if 'mean_reward' in perf:
                print(
                    f"   {model_name}: 보상 {perf['mean_reward']:.2f} ± {perf['std_reward']:.2f}, "
                    f"성공률 {perf['success_rate']:.3f}"
                )

    except Exception as e:
        print(f'❌ 평가 중 오류 발생: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
