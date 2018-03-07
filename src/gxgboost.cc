/*!
 * Copyright 2014 by Contributors
 * \file cli_main.cc
 * \brief The command line interface program of xgboost.
 *  This file is not included in dynamic library.
 */
// Copyright 2014 by Contributors
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX

#include <xgboost/learner.h>
#include <xgboost/data.h>
#include <xgboost/logging.h>
#include <xgboost/c_api.h>
#include <dmlc/timer.h>
#include <iomanip>
#include <ctime>
#include <string>
#include <cstdio>
#include <cstring>
#include <vector>
#include "./common/sync.h"
#include "./common/config.h"
#include "./common/math.h"
#include "./common/io.h"
#include "./data/simple_csr_source.h"
#include "gxgboost_export.h"

#ifndef GXGBOOST_EXPORT
#define GXGBOOST_EXPORT
#endif

#define getsize(R,C,S) ((R)*(C)*(S)/(size_t)8 + ( ((R)*(C)*(S))%(size_t)8 != (size_t)0 ) )

namespace xgboost {
namespace gauss {
typedef dmlc::SeekStream SeekStream;
/*! \brief a in memory buffer that can be read and write as stream interface */
struct MemoryBufferStream : public SeekStream {
public:
    explicit MemoryBufferStream(char *&p_buffer)
        : p_buffer_(p_buffer) {
        curr_ptr_ = 0;
        len_ = 0;
    }
    virtual ~MemoryBufferStream(void) {}
    virtual size_t Read(void *ptr, size_t size) {
        size_t nread = std::min(len_ - curr_ptr_, size);
        if (nread != 0) std::memcpy(ptr, p_buffer_ + curr_ptr_, nread);
        curr_ptr_ += nread;
        return nread;
    }
    virtual void Write(const void *ptr, size_t size) {
        if (size == 0) return;
        if (curr_ptr_ + size > len_) {
            len_ = curr_ptr_ + size;
            p_buffer_ = (char*)realloc(p_buffer_, len_);
        }
        std::memcpy(p_buffer_ + curr_ptr_, ptr, size);
        curr_ptr_ += size;
    }
    virtual void Seek(size_t pos) {
        curr_ptr_ = static_cast<size_t>(pos);
    }
    virtual size_t Tell(void) {
        return curr_ptr_;
    }
    virtual bool AtEnd(void) const {
        return curr_ptr_ == len_;
    }
    size_t Length(void) const {
        return len_;
    }

private:
    /*! \brief in memory buffer */
    char *&p_buffer_;
    /*! \brief current pointer */
    size_t curr_ptr_;
    size_t len_;
};  // class MemoryBufferStream
}

struct InputData
{
    double *data;
    double *rows;
    double *cols;
};

struct VectorData
{
    double *data;
    double *size;
};

struct LearningTaskParams
{
    /**
     * @brief num_round
     * number of boosting iterations
     * default: 10
     */
    double *num_round;

    /**
     * @brief objective
     * Specify the learning task and the corresponding learning objective. The objective options are below:
     *   "reg:linear" --linear regression
     *   "reg:logistic" --logistic regression
     *   "binary:logistic" --logistic regression for binary classification, output probability
     *   "binary:logitraw" --logistic regression for binary classification, output score before logistic transformation
     *   "gpu:reg:linear", "gpu:reg:logistic", "gpu:binary:logistic", gpu:binary:logitraw"
     *      --versions of the corresponding objective functions evaluated on the GPU; note that like the GPU histogram
     *        algorithm, they can only be used when the entire training session uses the same dataset
     *   "count:poisson" --poisson regression for count data, output mean of poisson distribution
     *     max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
     *   "survival:cox" --Cox regression for right censored survival time data (negative values are considered right censored).
     *     Note that predictions are returned on the hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the
     *     proportional hazard function h(t) = h0(t) * HR).
     *   "multi:softmax" --set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
     *   "multi:softprob" --same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata,
     *     nclass matrix. The result contains predicted probability of each data point belonging to each class.
     *   "rank:pairwise" --set XGBoost to do ranking task by minimizing the pairwise loss
     *   "reg:gamma" --gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling
     *     insurance claims severity, or for any outcome that might be gamma-distributed
     *   "reg:tweedie" --Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or
     *     for any outcome that might be Tweedie-distributed.
     * default: 'reg:linear'
     */
    char *objective;

    /**
     * @brief base_score
     * the initial prediction score of all instances, global bias
     * for sufficient number of iterations, changing this value will not have too much effect.
     * deault: 0.5
     */
    double *base_score;

    /**
     * @brief eval_metric
     * evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking )
     * User can add multiple evaluation metrics, for python user, remember to pass the metrics in as list of parameters pairs instead of map, so that latter 'eval_metric' won't override previous one
     * The choices are listed below:
     *   "rmse": root mean square error
     *   "mae": mean absolute error
     *   "logloss": negative log-likelihood
     *   "error": Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation
     *     will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
     *   "error@t": a different than 0.5 binary classification threshold value could be specified by providing a numerical value through 't'.
     *   "merror": Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
     *   "mlogloss": Multiclass logloss
     *   "auc": Area under the curve for ranking evaluation.
     *   "ndcg":Normalized Discounted Cumulative Gain
     *   "map":Mean average precision
     *   "ndcg@n","map@n": n can be assigned as an integer to cut off the top positions in the lists for evaluation.
     *   "ndcg-","map-","ndcg@n-","map@n-": In XGBoost, NDCG and MAP will evaluate the score of a list without any positive samples as 1.
     *     By adding "-" in the evaluation metric XGBoost will evaluate these score as 0 to be consistent under some conditions. training repeatedly
     *   "poisson-nloglik": negative log-likelihood for Poisson regression
     *   "gamma-nloglik": negative log-likelihood for gamma regression
     *   "cox-nloglik": negative partial log-likelihood for Cox proportional hazards regression
     *   "gamma-deviance": residual deviance for gamma regression
     *   "tweedie-nloglik": negative log-likelihood for Tweedie regression (at a specified value of the tweedie_variance_power parameter)
     * default: according to objective
     */
    char *eval_metric;

    /**
     * @brief seed
     * random number seed.
     * default: 0
     */
    double *seed;

    /**
     * @brief tweedie_variance_power
     * parameter that controls the variance of the Tweedie distribution
     *   var(y) ~ E(y)^tweedie_variance_power
     * range: (1,2)
     *   set closer to 2 to shift towards a gamma distribution
     *   set closer to 1 to shift towards a Poisson distribution.
     * default: 1.5
     */
    double *tweedie_variance_power;
};

struct TreeBoosterTrainParams
{
    /**
     * @brief eta
     * step size shrinkage used in update to prevents overfitting.
     * After each boosting step, we can directly get the weights of
     * new features. and eta actually shrinks the feature weights to
     * make the boosting process more conservative.
     * range: [0,1]
     * default: 0.3
    */
    double *eta;

    /**
     * @brief gamma
     * minimum loss reduction required to make a further partition
     * on a leaf node of the tree. The larger, the more conservative
     * the algorithm will be.
     * range: [0,∞]
     * default: 0
    */
    double *gamma;

    /**
     * @brief max_depth
     * maximum depth of a tree, increase this value will make the model
     * more complex / likely to be overfitting. 0 indicates no limit,
     * limit is required for depth-wise grow policy.
     * range: [0,∞]
     * default: 6
     */
    double *max_depth;

    /**
     * @brief min_child_weight
     * minimum sum of instance weight (hessian) needed in a child.
     * If the tree partition step results in a leaf node with the
     * sum of instance weight less than min_child_weight, then the
     * building process will give up further partitioning. In linear
     * regression mode, this simply corresponds to minimum number
     * of instances needed to be in each node. The larger, the
     * more conservative the algorithm will be.
     * range: [0,∞]
     * default: 1
     */
    double *min_child_weight;

    /**
     * @brief max_delta_step
     * Maximum delta step we allow each tree's weight estimation to
     * be. If the value is set to 0, it means there is no constraint.
     * If it is set to a positive value, it can help making the update
     * step more conservative. Usually this parameter is not needed,
     * but it might help in logistic regression when class is extremely
     * imbalanced. Set it to value of 1-10 might help control the update
     * range: [0,∞]
     * default: 0
    */
    double *max_delta_step;

    /**
     * @brief subsample
     * ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
     * range: [0,1]
     * default: 1
     */
    double *subsample;

    /**
     * @brief colsample_bytree
     * subsample ratio of columns when constructing each tree.
     * range: (0,1]
     * default: 1
     */
    double *colsample_bytree;

    /**
     * @brief colsample_bylevel
     * subsample ratio of columns for each split, in each level.
     * range: (0,1]
     * default: 1
     */
    double *colsample_bylevel;

    /**
     * @brief lambda
     * L2 regularization term on weights, increase this value will make model more conservative.
     * default: 1
     */
    double *lambda;

    /**
     * @brief alpha
     * L1 regularization term on weights, increase this value will make model more conservative.
     * default: 0
     */
    double *alpha;


    /**
     * @brief tree_method
     * The tree construction algorithm used in XGBoost(see description in the reference paper)
     * Distributed and external memory version only support approximate algorithm.
     * Choices: {'auto', 'exact', 'approx', 'hist', 'gpu_exact', 'gpu_hist'}
     *   'auto': Use heuristic to choose faster one.
     *     For small to medium dataset, exact greedy will be used.
     *     For very large-dataset, approximate algorithm will be chosen.
     *     Because old behavior is always use exact greedy in single machine, user will get a
     *     message when approximate algorithm is chosen to notify this choice.
     *   'exact': Exact greedy algorithm.
     *   'approx': Approximate greedy algorithm using sketching and histogram.
     *   'hist': Fast histogram optimized approximate greedy algorithm. It uses some performance
     *           improvements such as bins caching.
     *   'gpu_exact': GPU implementation of exact algorithm.
     *   'gpu_hist': GPU implementation of hist algorithm.
     * default: 'auto'
     */
    char *tree_method;

    /**
     * @brief sketch_eps
     * This is only used for approximate greedy algorithm.
     * This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select
     * number of bins, this comes with theoretical guarantee with sketch accuracy.
     * Usually user does not have to tune this. but consider setting to a lower number for more
     * accurate enumeration.
     * range: (0, 1)
     * default: 0.03
     */
    double *sketch_eps;

    /**
     * @brief scale_pos_weight
     * Control the balance of positive and negative weights, useful for unbalanced classes.
     * A typical value to consider: sum(negative cases) / sum(positive cases) See Parameters
     * Tuning for more discussion. Also see Higgs Kaggle competition demo for examples: R, py1, py2, py3
     * default: 1
     */
    double *scale_pos_weight;

    /**
     * @brief updater
     * A comma separated string defining the sequence of tree updaters to run, providing a modular way to
     * construct and to modify the trees. This is an advanced parameter that is usually set automatically,
     * depending on some other parameters. However, it could be also set explicitly by a user. The following
     * updater plugins exist:
     *   'grow_colmaker': non-distributed column-based construction of trees.
     *   'distcol': distributed tree construction with column-based data splitting mode.
     *   'grow_histmaker': distributed tree construction with row-based data splitting based on global proposal of histogram counting.
     *   'grow_local_histmaker': based on local histogram counting.
     *   'grow_skmaker': uses the approximate sketching algorithm.
     *   'sync': synchronizes trees in all distributed nodes.
     *   'refresh': refreshes tree's statistics and/or leaf values based on the current data. Note that no random subsampling of data rows is performed.
     *   'prune': prunes the splits where loss < min_split_loss (or gamma).
     * In a distributed setting, the implicit updater sequence value would be adjusted as follows:
     *   'grow_histmaker,prune' when dsplit='row' (or default) and prob_buffer_row == 1 (or default); or when data has multiple sparse pages
     *   'grow_histmaker,refresh,prune' when dsplit='row' and prob_buffer_row < 1
     *   'distcol' when dsplit='col'
     * default: 'grow_colmaker,prune'
     */
    char *updater;

    /**
     * @brief refresh_leaf
     * This is a parameter of the 'refresh' updater plugin. When this flag is true, tree leafs as well as
     * tree nodes' stats are updated. When it is false, only node stats are updated.
     * default: 1
     */
    double *refresh_leaf;

    /**
     * @brief process_type
     * A type of boosting process to run.
     * Choices: {'default', 'update'}
     *   'default': the normal boosting process which creates new trees.
     *   'update': starts from an existing model and only updates its trees. In each boosting iteration,
     *             a tree from the initial model is taken, a specified sequence of updater plugins is
     *             run for that tree, and a modified tree is added to the new model. The new model would
     *             have either the same or smaller number of trees, depending on the number of boosting
     *             iteratons performed. Currently, the following built-in updater plugins could be meaningfully
     *             used with this process type: 'refresh', 'prune'. With 'update', one cannot use updater plugins
     *             that create new trees.
     * default: 'default'
     */
    char *process_type;

    /**
     * @brief grow_policy
     * Controls a way new nodes are added to the tree.
     * Currently supported only if tree_method is set to 'hist'.
     * Choices: {'depthwise', 'lossguide'}
     *   'depthwise': split at nodes closest to the root.
     *   'lossguide': split at nodes with highest loss change.
     * default: 'depthwise'
     */
    char *grow_policy;


    /**
     * @brief max_leaves
     * Maximum number of nodes to be added. Only relevant for the 'lossguide' grow policy.
     * default: 0
     */
    double *max_leaves;


    /**
     * @brief max_bin
     * This is only used if 'hist' is specified as tree_method.
     * Maximum number of discrete bins to bucket continuous features.
     * Increasing this number improves the optimality of splits at the cost of higher computation time.
     * default: 256
     */
    double *max_bin;

    /**
     * @brief predictor
     * The type of predictor algorithm to use. Provides the same results but allows the use of GPU or CPU.
     *   'cpu_predictor': Multicore CPU prediction algorithm.
     *   'gpu_predictor': Prediction using GPU. Default for 'gpu_exact' and 'gpu_hist' tree method.
     *
     * default: 'cpu_predictor'
    */
    char *predictor;
};

struct DartBoosterTrainParams
{
    struct TreeBoosterTrainParams tree;

    /**
     * @brief sample_type
     * type of sampling algorithm.
     *   "uniform": dropped trees are selected uniformly.
     *   "weighted": dropped trees are selected in proportion to weight.
     * default: 'uniform'
     */
    char *sample_type;

    /**
     * @brief normalize_type
     * type of normalization algorithm.
     *   "tree": new trees have the same weight of each of dropped trees.
     *           weight of new trees are 1 / (k + learning_rate)
     *           dropped trees are scaled by a factor of k / (k + learning_rate)
     *   "forest": new trees have the same weight of sum of dropped trees (forest).
     *           weight of new trees are 1 / (1 + learning_rate)
     *           dropped trees are scaled by a factor of 1 / (1 + learning_rate)
     * default: 'tree'
     */
    char *normalize_type;

    /**
     * @brief rate_drop
     * dropout rate (a fraction of previous trees to drop during the dropout).
     * range: [0.0, 1.0]
     * default: 0.0
     */
    double *rate_drop;

    /**
     * @brief one_drop
     * when this flag is enabled, at least one tree is always dropped during the dropout
     * (allows Binomial-plus-one or epsilon-dropout from the original DART paper).
     * default: 0
     */
    double *one_drop;

    /**
     * @brief skip_drop
     * Probability of skipping the dropout procedure during a boosting iteration.
     * If a dropout is skipped, new trees are added in the same manner as gbtree.
     * Note that non-zero skip_drop has higher priority than rate_drop or one_drop.
     * range: [0.0, 1.0]
     * default: 0.0
     */
    double *skip_drop;
};

struct LinearBoosterTrainParams
{
    /**
     * @brief lambda
     * L2 regularization term on weights, increase this value will make model more conservative.
     * Normalised to number of training examples.
     * default: 0
     */
    double *lambda;

    /**
     * @brief alpha
     * L1 regularization term on weights, increase this value will make model more conservative.
     * Normalised to number of training examples.
     * default: 0
     */
    double *alpha;

    /**
     * @brief updater
     * Linear model algorithm
     *   'shotgun': Parallel coordinate descent algorithm based on shotgun algorithm. Uses 'hogwild' parallelism and
     *              therefore produces a nondeterministic solution on each run.
     *   'coord_descent': Ordinary coordinate descent algorithm. Also multithreaded but still produces a deterministic solution.
     * default: 'shotgun'
     */
    char *updater;
};

struct BoosterTrainInput
{
    struct InputData input;
    double *labels;
    struct VectorData model;
    struct LearningTaskParams learning;
};

struct TreeBoosterTrainInput
{
    struct BoosterTrainInput train;
    struct TreeBoosterTrainParams params;
};

struct DartBoosterTrainInput
{
    struct BoosterTrainInput train;
    struct DartBoosterTrainParams params;
};

struct LinearBoosterTrainInput
{
    struct BoosterTrainInput train;
    struct LinearBoosterTrainParams params;
};

struct BoosterPredictParams
{
    /**
     * @brief output_margin
     * whether to only predict margin value instead of transformed prediction
     * default: 0
     */
    double *output_margin;

    /**
     * @brief ntree_limit
     * limit number of trees used for prediction, this is only valid for boosted trees
     *  when the parameter is set to 0, we will use all the trees
     * default: 0
     */
    double *ntree_limit;

    /**
     * @brief pred_leaf
     * whether to only predict the leaf index of each tree in a boosted tree predictor
     * default: 0/false
     */
    double *pred_leaf;

    /**
     * @brief pred_contribs
     * whether to only predict the feature contributions
     * default: 0/false
     */
    double *pred_contribs;

    /**
     * @brief approx_contribs
     * whether to approximate the feature contributions for speed
     * default: 0/false
     */
    double *approx_contribs;

    /**
     * @brief pred_interactions
     * whether to compute the feature pair contributions
     * default: 0/false
     */
    double *pred_interactions;
};

struct BoosterPredict
{
    struct InputData test;
    struct VectorData model;
    struct VectorData results;
    struct BoosterPredictParams params;
};

#define GSCALERR(A,B)      *(((unsigned int *)(A))+1) = 0x7fffffff, *((unsigned int *)(A)) = (unsigned int)(B)

template<typename T = double>
static std::string convert_arg(double *input)
{
    return std::to_string(static_cast<T>(*input));
}

static int gxgboost_load(Learner *handle,
                         const void* buf,
                         size_t len)
{
    common::MemoryFixSizeBuffer fs((void*)buf, len);  // NOLINT(*)
    handle->Load(&fs);
    std::vector<std::pair<std::string, std::string> > cfg;
    static_cast<dmlc::Stream*>(&fs)->Read(&cfg);
    handle->Configure(cfg);
    return 0;
}

static int gxgboost_create(const double* data,
                           xgboost::bst_ulong nrow,
                           xgboost::bst_ulong ncol,
                           double missing,
                           DMatrix** out)
{
    std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());

    data::SimpleCSRSource& mat = *source;
    mat.row_ptr_.resize(1+nrow);
    bool nan_missing = common::CheckNAN<double>(missing);
    mat.info.num_row = nrow;
    mat.info.num_col = ncol;
    const double* data0 = data;

    // count elements for sizing data
    data = data0;
    for (xgboost::bst_ulong i = 0; i < nrow; ++i, data += ncol) {
        xgboost::bst_ulong nelem = 0;
        for (xgboost::bst_ulong j = 0; j < ncol; ++j) {
            if (common::CheckNAN<double>(data[j])) {
                CHECK(nan_missing)
                        << "There are NAN in the matrix, however, you did not set missing=NAN";
            } else {
                if (nan_missing || data[j] != missing) {
                    ++nelem;
                }
            }
        }
        mat.row_ptr_[i+1] = mat.row_ptr_[i] + nelem;
    }
    mat.row_data_.resize(mat.row_data_.size() + mat.row_ptr_.back());

    data = data0;
    for (xgboost::bst_ulong i = 0; i < nrow; ++i, data += ncol) {
        xgboost::bst_ulong matj = 0;
        for (xgboost::bst_ulong j = 0; j < ncol; ++j) {
            if (common::CheckNAN<double>(data[j])) {
            } else {
                if (nan_missing || data[j] != missing) {
                    mat.row_data_[mat.row_ptr_[i] + matj] = RowBatch::Entry(j, data[j]);
                    ++matj;
                }
            }
        }
    }

    mat.info.num_nonzero = mat.row_data_.size();
    *out = DMatrix::Create(std::move(source));

    return 0;
}

static int gxgboost_save(Learner *handle,
                         double* out_len,
                         double* out_dptr,
                         const std::vector<std::pair<std::string, std::string> > &cfg)
{
    char *buf = nullptr;

    gauss::MemoryBufferStream fo(buf);
    handle->Save(&fo);
    static_cast<dmlc::Stream*>(&fo)->Write(cfg);

    size_t len = getsize(fo.Length(), 1, 1);
    buf = reinterpret_cast<char*>(realloc(buf, len * sizeof(double)));

    *((double**)out_dptr) = reinterpret_cast<double*>(buf);
    *out_len = static_cast<double>(len);

    return 0;
}

static DMatrix* gxgboost_create_data(const struct InputData *input)
{
    double missing;
    GSCALERR(&missing, 0);

    DMatrix *dtrain_p = nullptr;

    // Copy GAUSS data
    gxgboost_create(input->data,
                    static_cast<bst_ulong>(*input->rows),
                    static_cast<bst_ulong>(*input->cols),
                    missing,
                    &dtrain_p);

    return dtrain_p;
}

static void gxgboost_configure_learning(std::vector<std::pair<std::string, std::string> > &cfg, struct LearningTaskParams *params)
{
    cfg.push_back(std::make_pair("num_round", convert_arg<int>(params->num_round)));
    cfg.push_back(std::make_pair("objective", std::string(params->objective)));
    cfg.push_back(std::make_pair("base_score", convert_arg<>(params->base_score)));
    cfg.push_back(std::make_pair("seed", convert_arg<int>(params->seed)));

    if (strlen(params->eval_metric)) {
        cfg.push_back(std::make_pair("eval_metric", std::string(params->eval_metric)));

        const char *tweedie_nloglik = "tweedie-nloglik";
        if (!strncmp(params->eval_metric, tweedie_nloglik, strlen(tweedie_nloglik)))
            cfg.push_back(std::make_pair("tweedie_variance_power", convert_arg<>(params->tweedie_variance_power)));
    }
}

static void gxgboost_configure_tree(std::vector<std::pair<std::string, std::string> > &cfg, struct TreeBoosterTrainParams *params)
{
    cfg.push_back(std::make_pair("eta", convert_arg<>(params->eta)));
    cfg.push_back(std::make_pair("gamma", convert_arg<int>(params->gamma)));
    cfg.push_back(std::make_pair("max_depth", convert_arg<int>(params->max_depth)));
    cfg.push_back(std::make_pair("min_child_weight", convert_arg<int>(params->min_child_weight)));
    cfg.push_back(std::make_pair("max_delta_step", convert_arg<int>(params->max_delta_step)));
    cfg.push_back(std::make_pair("subsample", convert_arg<>(params->subsample)));
    cfg.push_back(std::make_pair("colsample_bytree", convert_arg<>(params->colsample_bytree)));
    cfg.push_back(std::make_pair("colsample_bylevel", convert_arg<>(params->colsample_bylevel)));
    cfg.push_back(std::make_pair("lambda", convert_arg<int>(params->lambda)));
    cfg.push_back(std::make_pair("alpha", convert_arg<int>(params->alpha)));
    cfg.push_back(std::make_pair("tree_method", std::string(params->tree_method)));
    cfg.push_back(std::make_pair("sketch_eps", convert_arg<>(params->sketch_eps)));
    cfg.push_back(std::make_pair("scale_pos_weight", convert_arg<>(params->scale_pos_weight)));
    cfg.push_back(std::make_pair("updater", std::string(params->updater)));
    cfg.push_back(std::make_pair("refresh_leaf", convert_arg<int>(params->refresh_leaf)));
    cfg.push_back(std::make_pair("process_type", std::string(params->process_type)));
    cfg.push_back(std::make_pair("grow_policy", std::string(params->grow_policy)));
    cfg.push_back(std::make_pair("max_leaves", convert_arg<int>(params->max_leaves)));
    cfg.push_back(std::make_pair("max_bin", convert_arg<int>(params->max_bin)));
    cfg.push_back(std::make_pair("predictor", std::string(params->predictor)));
}

static void gxgboost_configure_dart(std::vector<std::pair<std::string, std::string> > &cfg, struct DartBoosterTrainParams *params)
{
    gxgboost_configure_tree(cfg, &params->tree);
    cfg.push_back(std::make_pair("sample_type", std::string(params->sample_type)));
    cfg.push_back(std::make_pair("normalize_type", std::string(params->normalize_type)));
    cfg.push_back(std::make_pair("rate_drop", convert_arg<>(params->rate_drop)));
    cfg.push_back(std::make_pair("one_drop", convert_arg<int>(params->one_drop)));
    cfg.push_back(std::make_pair("skip_drop", convert_arg<>(params->skip_drop)));
}

static void gxgboost_configure_linear(std::vector<std::pair<std::string, std::string> > &cfg, struct LinearBoosterTrainParams *params)
{
    cfg.push_back(std::make_pair("lambda", convert_arg<int>(params->lambda)));
    cfg.push_back(std::make_pair("alpha", convert_arg<int>(params->alpha)));
    cfg.push_back(std::make_pair("updater", std::string(params->updater)));
}

/**
 * @brief gxgboost_train Main train function. Configures the learner parameters and performs the boosting function.
 * @param cfg
 * @param booster
 * @return
 */
static int gxgboost_train(std::vector<std::pair<std::string, std::string> > &cfg, struct BoosterTrainInput *booster)
{
    std::shared_ptr<DMatrix> dtrain(gxgboost_create_data(&booster->input));
    dtrain->info().SetInfo("label", booster->labels, kDouble, static_cast<size_t>(*booster->input.rows));

    std::vector<std::shared_ptr<DMatrix> > cache_mats;
    cache_mats.emplace_back(dtrain);
    std::unique_ptr<Learner> learner(Learner::Create(cache_mats));

    gxgboost_configure_learning(cfg, &booster->learning);

    int version = 0;

    learner->Configure(cfg);
    learner->InitModel();

    // start training.
    const double start = dmlc::GetTime();
    for (int i = version / 2; i < static_cast<int>(*booster->learning.num_round); ++i) {
        if (version % 2 == 0) {
            LOG(CONSOLE) << "boosting round " << i << ", " << (dmlc::GetTime() - start) << " sec elapsed";
            learner->UpdateOneIter(i, dtrain.get());
            version += 1;
        }
//        std::string res = learner->EvalOneIter(i, eval_datasets, eval_data_names);
        version += 1;
    }

    gxgboost_save(learner.get(), booster->model.size, booster->model.data, cfg);

    double elapsed = dmlc::GetTime() - start;
    LOG(CONSOLE) << "update end, " << elapsed << " sec in all";

    return 0;
}

extern "C" GXGBOOST_EXPORT int gxgboost_train_dart(struct DartBoosterTrainInput *input)
{
    // Assign GAUSS input to parameter class
    std::vector<std::pair<std::string, std::string> > cfg;
    cfg.push_back(std::make_pair("booster", "dart"));
    gxgboost_configure_dart(cfg, &input->params);
    return gxgboost_train(cfg, &input->train);
}

extern "C" GXGBOOST_EXPORT int gxgboost_train_linear(struct LinearBoosterTrainInput *input)
{
    // Assign GAUSS input to parameter class
    std::vector<std::pair<std::string, std::string> > cfg;
    cfg.push_back(std::make_pair("booster", "linear"));
    gxgboost_configure_linear(cfg, &input->params);
    return gxgboost_train(cfg, &input->train);
}

extern "C" GXGBOOST_EXPORT int gxgboost_train_tree(struct TreeBoosterTrainInput *input)
{
    // Assign GAUSS input to parameter class
    std::vector<std::pair<std::string, std::string> > cfg;
    cfg.push_back(std::make_pair("booster", "gbtree"));
    gxgboost_configure_tree(cfg, &input->params);
    return gxgboost_train(cfg, &input->train);
}

extern "C" GXGBOOST_EXPORT int gxgboost_predict(struct BoosterPredict *input) {
    // load data
    std::unique_ptr<DMatrix> dtest(gxgboost_create_data(&input->test));

    // load model
    std::unique_ptr<Learner> learner(Learner::Create({}));
    gxgboost_load(learner.get(),
                  input->model.data,
                  static_cast<size_t>(*input->model.size) * sizeof(double));

    HostDeviceVector<bst_float> preds;
    learner->Predict(dtest.get(),
                     static_cast<bool>(*input->params.output_margin != 0.0),
                     &preds,
                     static_cast<unsigned>(*input->params.ntree_limit),
                     static_cast<bool>(*input->params.pred_leaf != 0.0),
                     static_cast<bool>(*input->params.pred_contribs != 0.0),
                     static_cast<bool>(*input->params.approx_contribs != 0.0),
                     static_cast<bool>(*input->params.pred_interactions != 0.0));

    std::vector<bst_float> &preds_data = preds.data_h();
    if (!preds_data.empty()) {
        double *dp = reinterpret_cast<double*>(malloc(preds_data.size() * sizeof(double)));

        std::copy(preds_data.begin(), preds_data.end(), dp);

        *((double**)(input->results.data)) = dp;
        *input->results.size = static_cast<double>(preds_data.size());
    }

    return 0;
}

}  // namespace xgboost
