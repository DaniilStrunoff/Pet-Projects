from sklearn.model_selection import KFold
from distances import euclidean_distance, cosine_distance
import nearest_neighbors


def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    scores_dict = {}
    
    if cv == None:
        folds = KFold(n_splits=3)
        for k in k_list:
            a = KNNClassifier(k, **kwargs)
            scores_list = []
            for train_index, test_index in folds.split(X):
                a.fit(X[train_index],y[train_index])
                Y = a.predict(X[test_index])
                score = accuracy(Y,y[test_index])
                scores_list.append(score)
            scores_dict[f"{k}"] = scores_list
        return scores_dict
              
    
    for k in k_list:
        a = KNNClassifier(k, **kwargs)
        scores_list = []
        for i in cv:
            a.fit(X[i[0]],y[i[0]])
            Y = a.predict(X[i[1]])
            score = accuracy(Y,y[i[1]])
            scores_list.append(score)
        scores_dict[f"{k}"] = scores_list
    return scores_dict