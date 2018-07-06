# Mobike Distribution Inference (TKDE)

> Inferring dockless shared bike distributions in new cities by transfering the knowledge from the cities popular with the dockless shared bikes.

Keras implementation for the paper **'Inferring Dockless Shared Bike Distribution in New cities'**, which was publised on the WSDM 2018.

Foreknowing the dockless shared bike distributions in a new city is of great significance for the design of the bike delivering strategy, government regulations, renewing the traffic rules, and so on. In this paper, based on the multi-source data and known bike distirbutions in the delivered cities, we employed the convolutional neural network to model the interaction between geo-related information and the bike distributions, and then applied this learnt knowledge to the target city to infer the potential dockless shared bike distributions.


## DataSet

We collect muliple geo-related data from differet sources, including:
 
 * Mobike data (by crawler);
 * POI data from ([Baidu Map](http://lbsyun.baidu.com)); 
 * Satellite light ([Light Polution](https://www.lightpollutionmap.info));  
 * Road network ([OpenStreetMap](https://openstreetmap.org));
 * Transportation hubs (collect manually);
 * Business centers ([Fliggy](https://www.fliggy.com));

If you're interested in these data, you can crawl or get the data from our provided urls.



<!--## Project Structure

The project is organized as follows: 
	
```
data/
	meta_data: multiple source data;
	road/
		train_bound: data version for feed forward neural network;
		train_bound_cnn: data version for convolutional network;
metrics/
	the maximum mean discrepancy metric implementation
models/
	core models and run scripts
results/
	the spatial-temporal characteristics analysis results
road_match/
	map matching and feature extraction scripts
util/
	general common functions
```
-->

## Run Command
	
1. Run script:
	
	```
	>> python main.py --train_cities bj --test_cities nb --model_choice 0 --y_scale --epochs 200
	```
	
2. Parameter description:
	* **model_choice**: model choice for the problem;
	* **train_cities**: source cities we would transfer knowledge from
	* **target_cities**: target citeis that will be applied the model
	* **y_scale**: whether to scale the target label;
	* **epochs**: maximum training epochs;

<!--## Citation

If you use this code or the data for your research, please cite our paper:

```
@inproceedings{Liu2018Where,
  title={Where Will Dockless Shared Bikes be Stacked? — Parking Hotspots Detection in a New City},
  author={Liu, Zhaoyang and Shen, Yanyan and Zhu, Yanmin},
  booktitle={The 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
  pages={378-386},
  year={2018},
}
```-->

## Reference

1. Bike lane planning work in KDD 2017, [paper link](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/main-1.pdf):

```
@inproceedings{Bao2017Planning,
  title={Planning Bike Lanes based on Sharing-Bikes' Trajectories},
  author={Bao, Jie and He, Tianfu and Ruan, Sijie and Li, Yanhua and Zheng, Yu},
  booktitle={ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={1377-1386},
  year={2017},
}
```


## License

1. For academic and non-commercial use only.
2. For commercial use, please contact [Mobike Company](https://mobike.com/cn/)
    
