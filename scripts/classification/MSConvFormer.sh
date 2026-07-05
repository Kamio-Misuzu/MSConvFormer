# # loss>f1>auroc>acc
# # 消融实验（超参与本文件对齐）：见 scripts/classification/MSConvFormer_ablation.sh
CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/APAVA/ \
  --model_id APAVA-Indep \
  --model MSConvFormer \
  --data APAVA \
  --seq_len 256 \
  --pred_len 0 \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --n_heads 8 \
  --d_ff 256 \
  --dropout 0.05 \
  --use_mixup True \
  --mixup_alpha 0.4 \
  --mixup_prob 0.5 \
  --weight_decay 0.0001 \
  --cls_early_stop_metric loss \
  --des 'Exp_APAVA-Indep' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 150 \
  --patience 12 \
  --gpu 0




# # e_layers:4>6
# # batch_size:32>64

# CUDA_VISIBLE_DEVICES=6 python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/ADFTD/ \
#   --model_id ADFTD-Indep \
#   --model MSConvFormer \
#   --data ADFTD \
#   --seq_len 256 \
#   --e_layers 4 \
#   --batch_size 32 \
#   --d_model 128 \
#   --n_heads 8 \
#   --d_ff 256 \
#   --dropout 0.4 \
#   --use_mixup True \
#   --mixup_alpha 0.25 \
#   --mixup_prob 0.4 \
#   --des 'Exp_ADFTD-Indep' \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 120 \
#   --patience 10\
#   --gpu 0

## Subject-Dependent (Dep)
# 可切换早停指标：f1 / loss / auroc / acc / precision / recall / auprc / mean6

# CUDA_VISIBLE_DEVICES=6 python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/ADFTD/ \
#   --model_id ADFTD-Dep \
#   --model MSConvFormer \
#   --data ADFTD-Dependent \
#   --seq_len 256 \
#   --e_layers 4 \
#   --batch_size 16 \
#   --d_model 128 \
#   --n_heads 8 \
#   --d_ff 256 \
#   --dropout 0.35 \
#   --use_mixup True \
#   --mixup_alpha 0.25 \
#   --mixup_prob 0.4 \
#   --cls_early_stop_metric loss \
#   --des 'Exp_ADFTD_Dep' \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 120 \
#   --patience 10 \
#   --gpu 0


# CUDA_VISIBLE_DEVICES=6 python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/PTB/ \
#   --model_id PTB-Indep \
#   --model MSConvFormer \
#   --data PTB \
#   --seq_len 300 \
#   --pred_len 0 \
#   --e_layers 6 \
#   --batch_size 128 \
#   --d_model 256 \
#   --n_heads 8 \
#   --d_ff 1024 \
#   --dropout 0.15 \
#   --use_subject_norm True \
#   --use_mixup True \
#   --mixup_alpha 0.3 \
#   --mixup_prob 0.45 \
#   --weight_decay 0.0001 \
#   --cls_early_stop_metric auroc \
#   --des 'Exp_PTB-Indep' \
#   --itr 5 \
#   --learning_rate 0.0003 \
#   --train_epochs 150 \
#   --patience 12 \
#   --gpu 0


# ===============================PTB-XL===============================
# CUDA_VISIBLE_DEVICES=7 python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/PTB-XL/ \
#   --model_id PTB-XL_MSConvFormer_PTB_XL_auroc_opt_mean6_dropout_0.28\
#   --model MSConvFormer \
#   --data PTB-XL \
#   --seq_len 300 \
#   --pred_len 0 \
#   --e_layers 6 \
#   --batch_size 64 \
#   --d_model 256 \
#   --n_heads 8 \
#   --d_ff 1024 \
#   --dropout 0.28 \
#   --use_subject_norm True \
#   --use_mixup True \
#   --mixup_alpha 0.3 \
#   --mixup_prob 0.4 \
#   --weight_decay 0.0001 \
#   --cls_early_stop_metric mean6 \
#   --des 'Exp_MedDG_PTB_XL' \
#   --itr 5 \
#   --learning_rate 0.0002 \
#   --train_epochs 150 \
#   --patience 10 \
#   --gpu 0