# -*- coding: utf-8 -*_

import argparse
import os
from const import ModelChoice, ReducerChoice, FeatureChoice, ScaleChoice
from main import run
from evaluation.metrics import entropy_evaluation

if __name__ == '__main__':
    data_param_config = dict(
        n_components=list(range(2, 31, 2)),
        # n_components=[2],
	reducer_choice=[ReducerChoice.pca, ReducerChoice.fa]
    )
    model_param_config = {
        ModelChoice.cnn: dict(
            lr=[0.001, 0.0005],
            dropout=[0.0, 0.2, 0.5],
            bn=[True]
        ),
        ModelChoice.dense_cnn: dict(
            lr=[0.001, 0.0001],
            dropout=[0.2, 0.5],
            first_filter=[8, 16],
            nb_dense_block_layers=[(2,), (4,)],
            growth_rate=[4, 6], compression=[0.5]
        )
    }

    parser = argparse.ArgumentParser(description='mobike dist')
    parser.add_argument('--train_cities', required=True, default='bj', type=str)
    parser.add_argument('--test_cities', required=True, default='nb', type=str)
    parser.add_argument('--feature_choice', default=0, type=int,
                        help='feature choice: 0 -> all, 1 -> poi feature, 2 -> street features, 3 -> engineer features')
    parser.add_argument('--scale_choice', default=0, type=int,
                        help='target scale choice: 0 -> no scale, 1 -> std scale, 2 -> min max scale')
    parser.add_argument('--model_choice', default=0, type=int, help='model choice: 0 -> cnn, 1-> dense cnn',
                        choices=[0, 1])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--ensemble', action='store_true', help='whether ensemble the results of the train cities')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    train_city = args.train_cities.split(',')
    test_city = args.test_cities.split(',')

    if args.ensemble:
        assert len(train_city) > 1
        test_city_real, test_city_pred1, val_rmse1 = run(
            train_cities=[train_city[0]], test_cities=test_city,
            data_param_grid=data_param_config,
            model_param_dict=model_param_config, scale_choice=ScaleChoice(args.scale_choice), epochs=args.epochs,
            model_choice=ModelChoice(args.model_choice), feature_choice=FeatureChoice(args.feature_choice, ),
            test_origin=True
        )
        val_rmse1 = 1.0 / val_rmse1

        _, test_city_pred2, val_rmse2 = run(
            train_cities=[train_city[1]], test_cities=test_city,
            data_param_grid=data_param_config,
            model_param_dict=model_param_config, scale_choice=ScaleChoice(args.scale_choice), epochs=args.epochs,
            model_choice=ModelChoice(args.model_choice), feature_choice=FeatureChoice(args.feature_choice),
            test_origin=True
        )
        val_rmse2 = 1.0 / val_rmse2

        w1 = val_rmse1 / (val_rmse1 + val_rmse2)
        w2 = val_rmse2 / (val_rmse1 + val_rmse2)

        test_city_pred = w1 * test_city_pred1 + w2 * test_city_pred2

        t_kl, t_rmlse = entropy_evaluation(ModelChoice.cnn.name, test_city_real, test_city_pred)
    else:
        run(
            train_cities=train_city, test_cities=test_city, data_param_grid=data_param_config,
            model_param_dict=model_param_config, scale_choice=ScaleChoice(args.scale_choice), epochs=args.epochs,
            model_choice=ModelChoice(args.model_choice), feature_choice=FeatureChoice(args.feature_choice),
            test_origin=True
        )
