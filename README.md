# genenas_cv_implement
# Execute inference only (CLI):
python /path/to/folder/geneNas/main_genenas_cv_multi_obj_no_train.py --train_batch_size 1024 --task_name cifar10 --gpus 1 --eval_batch_size 1024  --h_main 6 --h_adf 4 --hidden_shape 64 32 32 --input_size 32 --max_arity 2 --N 2 --num_workers 4
#Execute RWE (CLI):
python -W ignore /path/to/folder/geneNas/main_cv_rwe_multi_obj.py --gpus 1 --task_name cifar10 --max_epochs 10 --popsize 10 --num_iter 20 --train_batch_size 512 --eval_batch_size 2048 --h_main 6 --h_adf 4 --num_workers 2 --N 1 --hidden_shape 32 32 32 --k-folds 2
#Train a choromosome (CLI):
python /path/to/folder/geneNas/train_chromosome.py --file_name '/path/to/chromosome/file' --task_name cifar10 --num_workers 2 --gpus 1 --N 2 --hidden_shape 64 32 32 --train_batch_size 256 --eval_batch_size 1028 --h_main 6 --h_adf 4 --max_epochs 20
