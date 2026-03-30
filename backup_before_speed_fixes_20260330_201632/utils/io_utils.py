"""
파일 입출력 유틸리티 함수들
"""

import os
import json
from typing import Dict, Any, Optional
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import VecNormalize


ALGORITHM_CLASSES = {
    'SAC': SAC,
    'PPO': PPO,
    'TD3': TD3,
}


def load_model(model_path: str, algorithm: str = 'SAC', env=None):
    """저장된 모델 로드"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    model_class = ALGORITHM_CLASSES.get(algorithm.upper())
    if model_class is None:
        raise ValueError(f"지원하지 않는 알고리즘: {algorithm}")

    return model_class.load(model_path, env=env)


def load_vec_normalize(vec_normalize_path: str, vec_env):
    """VecNormalize 통계 로드"""
    if not os.path.exists(vec_normalize_path):
        raise FileNotFoundError(f"VecNormalize 파일을 찾을 수 없습니다: {vec_normalize_path}")

    return VecNormalize.load(vec_normalize_path, vec_env)


def save_json(data: Dict[str, Any], filepath: str):
    """JSON 파일 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[str, Any]:
    """JSON 파일 로드"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def get_experiment_info(exp_dir: str) -> Dict[str, Any]:
    """실험 디렉토리에서 정보 추출"""
    info: Dict[str, Any] = {}

    summary_candidates = [
        os.path.join(exp_dir, 'training_summary.json'),
        os.path.join(exp_dir, 'logs', 'training_summary.json'),
    ]
    for summary_path in summary_candidates:
        if os.path.exists(summary_path):
            info.update(load_json(summary_path))
            info['summary_path'] = summary_path
            break

    models_dir = os.path.join(exp_dir, 'models')
    available_models = set()
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.zip'):
                available_models.add(file.replace('.zip', ''))

    # stage stats 만 존재하는 단계(예: stage_0_random)도 평가 대상에 포함
    logs_dir = os.path.join(exp_dir, 'logs')
    if os.path.exists(logs_dir):
        for file in os.listdir(logs_dir):
            if file.startswith('stage_') and file.endswith('_stats.npz'):
                available_models.add(file.replace('_stats.npz', ''))

    if available_models:
        info['available_models'] = sorted(available_models)

    checkpoint_candidates = [
        os.path.join(exp_dir, 'models', 'checkpoints'),
        os.path.join(exp_dir, 'checkpoints'),
    ]
    for checkpoints_dir in checkpoint_candidates:
        if os.path.exists(checkpoints_dir):
            checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.zip')]
            info['checkpoints'] = sorted(checkpoints)
            info['checkpoints_dir'] = checkpoints_dir
            break

    return info


def find_latest_experiment(base_dir: str = 'outputs', env_name: str = None) -> Optional[str]:
    """가장 최근 실험 디렉토리 찾기"""
    if not os.path.exists(base_dir):
        return None

    experiments = []
    for exp_dir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, exp_dir)
        if os.path.isdir(full_path):
            if env_name and env_name not in exp_dir:
                continue
            experiments.append((full_path, os.path.getmtime(full_path)))

    if experiments:
        experiments.sort(key=lambda x: x[1], reverse=True)
        return experiments[0][0]

    return None
