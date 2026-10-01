# 论文实验执行命令归档

说明：下面命令均按论文表格中的学习率整理。随机种子使用 42、43、44。

运行位置：进入你的代码根目录后执行。若你的目录不同，只需要修改开头的 `cd ...`。

GPU 写法：`CUDA_VISIBLE_DEVICES=1` 后，程序内部仍使用 `--device cuda:0`。

## Table 4：消融实验命令

对应论文：Table IV，CIFAR-10 + AlexNet。

固定层默认使用主实验中最常被选中的全连接层：`fc3,fc2,fc1`。

```bash
cd /path/to/钱开苹ISPA2026/code
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ./logs/paper_table4_ablation ./results/paper_table4_ablation

run_ablation() {
  local tag="$1"
  local seed="$2"
  local selection_strategy="$3"
  local k_strategy="$4"
  local fixed_k="$5"
  local min_k="$6"
  local max_k="$7"
  local hessian_estimates="$8"
  local fixed_layers="$9"

  echo "========== Table4 ablation | ${tag} | seed ${seed} =========="
  python main.py \
    --dataset cifar10 \
    --model alexnet \
    --partition iid \
    --algos dynahybrid \
    --rounds 100 \
    --clients 10 \
    --local_steps 40 \
    --batch_size 128 \
    --test_batch_size 256 \
    --train_subset 50000 \
    --test_subset 10000 \
    --seed "${seed}" \
    --data_dir ./datasets \
    --fedavg_lr 0.05 \
    --fedprox_mu 0.01 \
    --max_grad_norm 0 \
    --lr_schedule constant \
    --lr_milestones 50,75 \
    --lr_decay_gamma 0.2 \
    --sophia_lr 0.01 \
    --sophia_lr_schedule constant \
    --weight_decay 0.0001 \
    --image_size 224 \
    --beta1 0.9 \
    --beta2 0.95 \
    --epsilon 1e-8 \
    --rho 0.003 \
    --tau 10 \
    --coverage_threshold 0.6 \
    --first_score_round 2 \
    --score_interval 10 \
    --min_k "${min_k}" \
    --max_k "${max_k}" \
    --selection_strategy "${selection_strategy}" \
    --k_strategy "${k_strategy}" \
    --fixed_k "${fixed_k}" \
    --fixed_layers "${fixed_layers}" \
    --hybrid_hessian_scope selected \
    --hybrid_hessian_estimates "${hessian_estimates}" \
    --hybrid_track_m_for_all_layers true \
    --hybrid_warmup_hessian_round1 false \
    --print_every_round true \
    --save_csv true \
    --output_dir "./results/paper_table4_ablation/${tag}_seed_${seed}" \
    --device cuda:0 \
    --grouping_mode 8groups \
    --num_workers 2 \
    --target_accuracy 0.84 \
    2>&1 | tee "./logs/paper_table4_ablation/${tag}_seed_${seed}.log"
}

# Ranking: fixed layers, M=3, seed 42
run_ablation ranking_fixed_layers_m3 42 fixed fixed 3 3 3 1 fc3,fc2,fc1

# Ranking: dynamic scores, fixed M=3, seed 42
run_ablation ranking_dynamic_scores_m3 42 dynamic fixed 3 3 3 1 ""

# Count: fixed M=3, seeds 42--44
for seed in 42 43 44; do
  run_ablation count_fixed_m3 "${seed}" dynamic fixed 3 3 3 1 ""
done

# Count: dynamic M_t, seeds 42--44
for seed in 42 43 44; do
  run_ablation count_dynamic_mt "${seed}" dynamic dynamic 3 2 4 1 ""
done

# Refresh: 1 estimate, seed 42
run_ablation refresh_1_estimate 42 dynamic dynamic 3 2 4 1 ""

# Refresh: 40 estimates, seed 42
run_ablation refresh_40_estimates 42 dynamic dynamic 3 2 4 40 ""
```
