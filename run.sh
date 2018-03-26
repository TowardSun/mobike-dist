nohup python run.py --train_cities bj --test_cities nb --model_choice 0 --y_scale --epochs 200 > ./logs/bj_nb_scale_cnn.log 2>&1 &
nohup python run.py --train_cities bj --test_cities nb --model_choice 0 --epochs 200 > ./logs/bj_nb_cnn.log 2>&1 &
nohup python run.py --train_cities bj --test_cities nb --model_choice 1 --y_scale --epochs 200 > ./logs/bj_nb_scale_dense_cnn.log 2>&1 &
nohup python run.py --train_cities bj --test_cities nb --model_choice 1 --epochs 200 > ./logs/bj_nb_dense_cnn.log 2>&1 &

nohup python run.py --train_cities sh --test_cities nb --model_choice 0 --y_scale --epochs 200 --gpu 1 > ./logs/sh_nb_scale_cnn.log 2>&1 &
nohup python run.py --train_cities sh --test_cities nb --model_choice 0 --epochs 200 --gpu 1 > ./logs/sh_nb_cnn.log 2>&1 &
nohup python run.py --train_cities sh --test_cities nb --model_choice 1 --y_scale --epochs 200 --gpu 1 > ./logs/sh_nb_scale_dense_cnn.log 2>&1 &
nohup python run.py --train_cities sh --test_cities nb --model_choice 1 --epochs 200 --gpu 1 > ./logs/sh_nb_debse_cnn.log 2>&1 &

nohup python run.py --train_cities sh --test_cities bj --model_choice 0 --y_scale --epochs 200 --gpu 1 > ./logs/sh_bj_scale_cnn.log 2>&1 &
nohup python run.py --train_cities sh --test_cities bj --model_choice 0 --epochs 200 --gpu 1 > ./logs/sh_bj_cnn.log 2>&1 &
nohup python run.py --train_cities sh --test_cities bj --model_choice 1 --y_scale --epochs 200 --gpu 1 > ./logs/sh_bj_scale_dense_cnn.log 2>&1 &
nohup python run.py --train_cities sh --test_cities bj --model_choice 1 --epochs 200 --gpu 1 > ./logs/sh_bj_debse_cnn.log 2>&1 &
