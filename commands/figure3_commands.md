# 论文实验执行命令归档

说明：下面命令均按论文表格中的学习率整理。随机种子使用 42、43、44。

运行位置：进入你的代码根目录后执行。若你的目录不同，只需要修改开头的 `cd ...`。

GPU 写法：`CUDA_VISIBLE_DEVICES=1` 后，程序内部仍使用 `--device cuda:0`。

## Figure 3：Hessian estimator 适用性对照实验

对应论文：Figure 3。图 3 抽取三组代表实验：MNIST+ResNet18、CIFAR-10+AlexNet、CIFAR-100+AlexNet。

### FedSophia + DynaHybrid-Sophia

```bash
cd /path/to/钱开苹ISPA2026/code
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ./logs/paper_fig3_sophia ./results/paper_fig3_sophia

for seed in 42 43 44
do
  echo "========== fedsophia,dynahybrid | resnet18 + mnist | seed ${seed} =========="
  python main.py \
    --dataset mnist \
    --model resnet18 \
    --partition iid \
    --algos fedsophia,dynahybrid \
    --rounds 100 \
    --clients 10 \
    --local_steps 40 \
    --batch_size 128 \
    --test_batch_size 256 \
    --train_subset 60000 \
    --test_subset 10000 \
    --seed "${seed}" \
    --data_dir ./datasets \
    --fedavg_lr 0.10 \
    --fedprox_mu 0.01 \
    --max_grad_norm 0 \
    --lr_schedule constant \
    --lr_milestones 50,75 \
    --lr_decay_gamma 0.2 \
    --sophia_lr 0.05 \
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
    --min_k 1 \
    --max_k 3 \
    --selection_strategy dynamic \
    --k_strategy dynamic \
    --fixed_k 3 \
    --hybrid_hessian_scope selected \
    --hybrid_hessian_estimates 1 \
    --hybrid_track_m_for_all_layers true \
    --hybrid_warmup_hessian_round1 false \
    --print_every_round true \
    --save_csv true \
    --output_dir "./results/paper_fig3_sophia/mnist_resnet18_seed_${seed}" \
    --device cuda:0 \
    --grouping_mode 7groups \
    --num_workers 2 \
    --target_accuracy 0.98 \
    2>&1 | tee "./logs/paper_fig3_sophia/mnist_resnet18_seed_${seed}.log"

  echo "========== fedsophia,dynahybrid | alexnet + cifar10 | seed ${seed} =========="
  python main.py \
    --dataset cifar10 \
    --model alexnet \
    --partition iid \
    --algos fedsophia,dynahybrid \
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
    --min_k 2 \
    --max_k 4 \
    --selection_strategy dynamic \
    --k_strategy dynamic \
    --fixed_k 3 \
    --hybrid_hessian_scope selected \
    --hybrid_hessian_estimates 1 \
    --hybrid_track_m_for_all_layers true \
    --hybrid_warmup_hessian_round1 false \
    --print_every_round true \
    --save_csv true \
    --output_dir "./results/paper_fig3_sophia/cifar10_alexnet_seed_${seed}" \
    --device cuda:0 \
    --grouping_mode 8groups \
    --num_workers 2 \
    --target_accuracy 0.84 \
    2>&1 | tee "./logs/paper_fig3_sophia/cifar10_alexnet_seed_${seed}.log"

  echo "========== fedsophia,dynahybrid | alexnet + cifar100 | seed ${seed} =========="
  python main.py \
    --dataset cifar100 \
    --model alexnet \
    --partition iid \
    --algos fedsophia,dynahybrid \
    --rounds 100 \
    --clients 10 \
    --local_steps 40 \
    --batch_size 128 \
    --test_batch_size 256 \
    --train_subset 50000 \
    --test_subset 10000 \
    --seed "${seed}" \
    --data_dir ./datasets \
    --fedavg_lr 0.10 \
    --fedprox_mu 0.001 \
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
    --min_k 2 \
    --max_k 4 \
    --selection_strategy dynamic \
    --k_strategy dynamic \
    --fixed_k 3 \
    --hybrid_hessian_scope selected \
    --hybrid_hessian_estimates 1 \
    --hybrid_track_m_for_all_layers true \
    --hybrid_warmup_hessian_round1 false \
    --print_every_round true \
    --save_csv true \
    --output_dir "./results/paper_fig3_sophia/cifar100_alexnet_seed_${seed}" \
    --device cuda:0 \
    --grouping_mode 8groups \
    --num_workers 2 \
    --target_accuracy 0.50 \
    2>&1 | tee "./logs/paper_fig3_sophia/cifar100_alexnet_seed_${seed}.log"

done
```

### DynaHybrid-AdaHessian

```bash
cd /path/to/钱开苹ISPA2026/code
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ./logs/paper_fig3_adahessian ./results/paper_fig3_adahessian

for seed in 42 43 44
do
  echo "========== dynahybrid_adahessian | resnet18 + mnist | seed ${seed} =========="
  python main.py \
    --dataset mnist \
    --model resnet18 \
    --partition iid \
    --algos dynahybrid_adahessian \
    --rounds 100 \
    --clients 10 \
    --local_steps 40 \
    --batch_size 128 \
    --test_batch_size 256 \
    --train_subset 60000 \
    --test_subset 10000 \
    --seed "${seed}" \
    --data_dir ./datasets \
    --fedavg_lr 0.10 \
    --fedprox_mu 0.01 \
    --max_grad_norm 1.0 \
    --lr_schedule constant \
    --lr_milestones 50,75 \
    --lr_decay_gamma 0.2 \
    --sophia_lr 0.05 \
    --sophia_lr_schedule constant \
    --weight_decay 0.0001 \
    --image_size 224 \
    --beta1 0.9 \
    --beta2 0.95 \
    --epsilon 1e-8 \
    --rho 0.003 \
    --tau 10 \
    --adahessian_hutchinson_samples 1 \
    --coverage_threshold 0.6 \
    --first_score_round 2 \
    --score_interval 10 \
    --min_k 1 \
    --max_k 3 \
    --selection_strategy dynamic \
    --k_strategy dynamic \
    --fixed_k 3 \
    --hybrid_hessian_scope selected \
    --hybrid_hessian_estimates 1 \
    --hybrid_track_m_for_all_layers true \
    --hybrid_warmup_hessian_round1 false \
    --print_every_round true \
    --save_csv true \
    --output_dir "./results/paper_fig3_adahessian/mnist_resnet18_seed_${seed}" \
    --device cuda:0 \
    --grouping_mode 7groups \
    --num_workers 2 \
    --target_accuracy 0.98 \
    2>&1 | tee "./logs/paper_fig3_adahessian/mnist_resnet18_seed_${seed}.log"

  echo "========== dynahybrid_adahessian | alexnet + cifar10 | seed ${seed} =========="
  python main.py \
    --dataset cifar10 \
    --model alexnet \
    --partition iid \
    --algos dynahybrid_adahessian \
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
    --max_grad_norm 1.0 \
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
    --adahessian_hutchinson_samples 1 \
    --coverage_threshold 0.6 \
    --first_score_round 2 \
    --score_interval 10 \
    --min_k 2 \
    --max_k 4 \
    --selection_strategy dynamic \
    --k_strategy dynamic \
    --fixed_k 3 \
    --hybrid_hessian_scope selected \
    --hybrid_hessian_estimates 1 \
    --hybrid_track_m_for_all_layers true \
    --hybrid_warmup_hessian_round1 false \
    --print_every_round true \
    --save_csv true \
    --output_dir "./results/paper_fig3_adahessian/cifar10_alexnet_seed_${seed}" \
    --device cuda:0 \
    --grouping_mode 8groups \
    --num_workers 2 \
    --target_accuracy 0.84 \
    2>&1 | tee "./logs/paper_fig3_adahessian/cifar10_alexnet_seed_${seed}.log"

  echo "========== dynahybrid_adahessian | alexnet + cifar100 | seed ${seed} =========="
  python main.py \
    --dataset cifar100 \
    --model alexnet \
    --partition iid \
    --algos dynahybrid_adahessian \
    --rounds 100 \
    --clients 10 \
    --local_steps 40 \
    --batch_size 128 \
    --test_batch_size 256 \
    --train_subset 50000 \
    --test_subset 10000 \
    --seed "${seed}" \
    --data_dir ./datasets \
    --fedavg_lr 0.10 \
    --fedprox_mu 0.001 \
    --max_grad_norm 1.0 \
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
    --adahessian_hutchinson_samples 1 \
    --coverage_threshold 0.6 \
    --first_score_round 2 \
    --score_interval 10 \
    --min_k 2 \
    --max_k 4 \
    --selection_strategy dynamic \
    --k_strategy dynamic \
    --fixed_k 3 \
    --hybrid_hessian_scope selected \
    --hybrid_hessian_estimates 1 \
    --hybrid_track_m_for_all_layers true \
    --hybrid_warmup_hessian_round1 false \
    --print_every_round true \
    --save_csv true \
    --output_dir "./results/paper_fig3_adahessian/cifar100_alexnet_seed_${seed}" \
    --device cuda:0 \
    --grouping_mode 8groups \
    --num_workers 2 \
    --target_accuracy 0.50 \
    2>&1 | tee "./logs/paper_fig3_adahessian/cifar100_alexnet_seed_${seed}.log"

done
```
