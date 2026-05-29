experiment_name,cv_rmse,leaderboard_score,changes,result
baseline_cpu,0.03715,90.09,initial catboost,good
cv_cpu_best,0.03481,90.71142,5-fold CV best model,best
gpu_same_params,0.03483,89.32,gpu training,worse
depth_7,0.03554,90.03,depth=7,worse
lr_002_iter3000,0.03477,89.52,lr=0.02 iterations=3000,worse
l2_leaf_reg_5,0.03496,89.29,l2_leaf_reg=5,worse