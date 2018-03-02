#ifndef GXGBOOST_H
#define GXGBOOST_H

#ifdef __cplusplus
extern "C" {
#endif

extern int gxgboost_train_tree(struct TreeBoosterTrainInput *input);
extern int gxgboost_predict(struct TreeBoosterPredictInput *input);

#ifdef __cplusplus
}
#endif

#endif // GXGBOOST_H
