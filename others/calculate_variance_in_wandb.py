import wandb
import numpy as np

# wandb API 초기화
api = wandb.Api()

# 프로젝트 이름 (대시보드에서 확인)
project_name = "albertruaz-korea-advanced-institute-of-science-and-techn/TinyHanabi"

# 10개의 런 ID를 리스트에 추가 (대시보드에서 복사)
run_ids = [
    "as9r2yfx",
    "9e01khhv",
    "iho84uen",
    "u1m2rqx7",
    "gjzxpqe7",
    "v2riejft",
    "d5xxb8rr",
    "30scok3f",
    "yq9hrgbx",
    "dvb0amm4"
]

# 각 런에 대한 분산 저장 리스트
run_means = []
run_variances = []
all_rewards = []

for run_id in run_ids:
    run = api.run(f"{project_name}/{run_id}")
    
    history = run.history(keys=["reward"])
    rewards = history["reward"].dropna().to_numpy()
    
    mean = np.mean(rewards)
    run_means.append(mean)
    variance = np.var(rewards)
    run_variances.append(variance)
    
    # 전체 rewards 데이터에 추가
    all_rewards.extend(rewards)

# 전체 reward 데이터에 대한 분산 계산
total_mean = np.mean(all_rewards)
total_variance = np.var(all_rewards)

# 결과 출력
for i, variance in enumerate(run_variances):
    print(f"Run (ID: {run_ids[i]})| mean: {run_means[i]}, variance: {run_variances[i]}")
print(f"Whole data| mean: {total_mean}, variance: {total_variance}")
