# nohup python run.py --train_cities bj --test_cities nb --model_choice 0 --feature_choice 0 --epochs 200 > ./logs/bj_nb_cnn.log 2>&1 &
# nohup python run.py --train_cities bj --test_cities nb --model_choice 1 --feature_choice 0 --epochs 200 --gpu 1 > ./logs/bj_nb_dense_cnn.log 2>&1 &

# nohup python run.py --train_cities sh --test_cities nb --model_choice 0 --feature_choice 0 --epochs 200 --gpu 1 > ./logs/sh_nb_cnn.log 2>&1 &
# nohup python run.py --train_cities sh --test_cities nb --model_choice 1 --feature_choice 0 --epochs 200 --gpu 1 > ./logs/sh_nb_dense_cnn.log 2>&1 &

# nohup python run.py --train_cities sh --test_cities bj --model_choice 0 --feature_choice 0 --epochs 200 --gpu 1 > ./logs/sh_bj_cnn.log 2>&1 &
# nohup python run.py --train_cities sh --test_cities bj --model_choice 1 --feature_choice 0 --epochs 200 --gpu 1 > ./logs/sh_bj_dense_cnn.log 2>&1 &

# nohup python run.py --train_cities bj --test_cities sh --model_choice 0 --feature_choice 0 --epochs 200 --gpu 1 > ./logs/bj_sh_cnn.log 2>&1 &
# nohup python run.py --train_cities bj --test_cities sh --model_choice 1 --feature_choice 0 --epochs 200 --gpu 1 > ./logs/bj_sh_dense_cnn.log 2>&1 &

# multi-source feature test
# nohup python run.py --train_cities sh --test_cities nb --model_choice 0 --feature_choice 1 --epochs 200 --gpu 0 > ./logs/bj_nb_cnn_f1.log 2>&1 &
# nohup python run.py --train_cities sh --test_cities nb --model_choice 0 --feature_choice 2 --epochs 200 --gpu 0 > ./logs/bj_nb_cnn_f2.log 2>&1 &
# nohup python run.py --train_cities sh --test_cities nb --model_choice 0 --feature_choice 3 --epochs 200 --gpu 0 > ./logs/bj_nb_cnn_f3.log 2>&1 &

# nohup python run.py --train_cities sh --test_cities bj --model_choice 0 --feature_choice 1 --epochs 200 --gpu 1 > ./logs/bj_nb_cnn_f1.log 2>&1 &
# nohup python run.py --train_cities sh --test_cities bj --model_choice 0 --feature_choice 2 --epochs 200 --gpu 1 > ./logs/bj_nb_cnn_f2.log 2>&1 &
# nohup python run.py --train_cities sh --test_cities bj --model_choice 0 --feature_choice 3 --epochs 200 --gpu 1 > ./logs/bj_nb_cnn_f3.log 2>&1 &

# multi-source city
# nohup python run.py --train_cities sh,bj --test_cities nb --model_choice 0 --feature_choice 0 --epochs 200 --gpu 1 > ./logs/sh_bj_nb_cnn.log 2>&1 &
# nohup python run.py --train_cities sh,nb --test_cities bj --model_choice 0 --feature_choice 0 --epochs 200 > ./logs/sh_nb_bj_cnn.log 2>&1 &
# nohup python run.py --train_cities bj,nb --test_cities sh --model_choice 0 --feature_choice 0 --epochs 200 > ./logs/bj_nb_sh_cnn.log 2>&1 &

## multi-source city ensemble
nohup python run.py --train_cities sh,bj --test_cities nb --model_choice 0 --feature_choice 0 --epochs 200 --ensemble --gpu 1 > ./logs/sh_bj_nb_cnn.log 2>&1 &
# nohup python run.py --train_cities sh,nb --test_cities bj --model_choice 0 --feature_choice 0 --epochs 200 --ensemble > ./logs/sh_nb_bj_cnn.log 2>&1 &
nohup python run.py --train_cities bj,nb --test_cities sh --model_choice 0 --feature_choice 0 --epochs 200 --ensemble --gpu 0 > ./logs/bj_nb_sh_cnn.log 2>&1 &
