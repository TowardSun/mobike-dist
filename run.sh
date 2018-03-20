nohup python main.py --train_cities bj --test_cities nb --model_choice 0 --y_scale --epochs 200 > ./logs/bj_nb_scale_cnn.log 2>&1 & 
nohup python main.py --train_cities bj --test_cities nb --model_choice 0 --epochs 200 > ./logs/bj_nb_cnn.log 2>&1 &
