from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import cross_val_score

# 该函数得到想要观察的分数
def hyperopt_train_test(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']
    
    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']
    
    clf = DecisionTreeClassifier(**params)
    return cross_val_score(clf, X_, y).mean()


# 定义自己要观察的参数空间
space4dt = {
    'max_depth': hp.choice('max_depth', range(1, 20)), 
    'max_features': hp.choice('max_features', range(1, 5)),
    'criterion': hp.choice('criterion', ['gini', 'entropy']), 
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}

# 定义给搜索参数的函数的返回值
## 需要包含一个loss字段和status字段
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

# 定义参数搜索的过程
trials = Trials()
best = fmin(f, space4dt, algo=tpe.suggest, max_evals=300, trials=trials)

# print(best) 

# 打印搜索过程中所有的观察结果
for t in trials.trials:
    print(t)