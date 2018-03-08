# gxgboost
GAUSS wrapper for the popular XGBoost library. 

> XGBoost is an optimized distributed gradient boosting library designed to be highly ***efficient***, ***flexible*** and ***portable***.
> It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework.

## Installation

```bash
$ git clone https://github.com/aptech/gxgboost
$ cd gxgboost
$ git submodule init
$ git submodule update
$ git foreach --recursive submodule init
$ git foreach --recursive submodule update
$ mkdir build
$ cd build
```

### Windows
```bash
$ cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
$ nmake install OR jom install
```

### Linux
```bash
$ cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
$ make -jN install
```

gxgboost.zip is now installable by the GAUSS by either:
- Using either the application installer via Tools -> Install Application
- Using the GAUSS package installer CLI (gpkg(.exe)) in the GAUSS installation directory:
  - gpkg install gxgboost.zip

## Usage
Refer to data/gxgboost.e for a full example and data/gxgboost.sdf for full description of parameters.

### Setup
Initialize a control structure for the boosting method of your choice.

#### Tree Booster
```
struct XGBTree ctl;
ctl = xgbcreatectl("tree");
```

#### Dart Boooster
```
struct XGBDart ctl;
ctl = xgbcreatectl("dart");
```

#### Linear Booster
```
struct XGBLinear ctl;
ctl = xgbcreatectl("linear");
```

### Train
```
struct XGBModel model;
model = xgbFit(labels, train_data, ctl);
```

### Predict
```
pred = xgbPredict(model, test_data);
```

## License
-------
© Contributors, 2016. Licensed under an [Apache-2](https://github.com/dmlc/xgboost/blob/master/LICENSE) license.
