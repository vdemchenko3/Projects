import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


class BiasedRandomForest():
    # This class is a modified RandomForest class that uses trees from both the base and critical datasets for predictions
    def __init__(self, x, y, x_braf, y_braf, num_trees, num_trees_braf, num_features, sample_size=None, sample_size_braf=None, depth=10, min_leaf=5):
        np.random.seed(42) # keep same randoms for testing and reproducibility
        if sample_size is None:
            self.sample_size = len(y)
        if sample_size_braf is None:
            self.sample_size_braf = len(y_braf)
        self.x = x
        self.y = y
        self.num_trees = num_trees
        self.x_braf = x_braf
        self.y_braf = y_braf
        self.num_trees_braf = num_trees_braf
        self.num_features = num_features
        # number of decision splits
        self.depth = depth
        # minimum number of rows required in a node to cause further split. 
        self.min_leaf = min_leaf 
        
        # deciding how many features to include
        if self.num_features == 'sqrt':
            self.num_features = int(np.sqrt(x.shape[1]))
        elif self.num_features == 'log2':
            self.num_features = int(np.log2(x.shape[1]))
        else:
            self.num_features = num_features
        
        # creating tree instances
        trees = [self.create_tree() for i in range(num_trees)]
        trees_braf = [self.create_tree_braf() for i in range(num_trees_braf)]
        self.trees = trees+trees_braf
        
    def create_tree(self):
        # This is effectively bootstrapping sample_size number of samples in place
        rand_idxs = np.random.choice(len(self.y), replace = True, size = self.sample_size)

        # This gets us the features we've decided to use
        feature_idxs = np.random.choice(self.x.shape[1], size = self.num_features)
        
        # returns a DecisionTree using the randomized indexes selected and min_leaf
        return DecisionTree(self.x.iloc[rand_idxs], 
                            self.y[rand_idxs], 
                            self.num_features,
                            idxs=np.array(range(self.sample_size)),
                            feature_idxs = feature_idxs,
                            depth = self.depth,
                            min_leaf =  self.min_leaf)
    
    def create_tree_braf(self):
        # This is effectively bootstrapping sample_size number of samples in place
        rand_idxs = np.random.choice(len(self.y_braf), replace = True, size = self.sample_size_braf)

        # This gets us the features we've decided to use
        feature_idxs = np.random.choice(self.x_braf.shape[1], size = self.num_features)
        
        # returns a DecisionTree using the randomized indexes selected and min_leaf
        return DecisionTree(self.x_braf.iloc[rand_idxs], 
                            self.y_braf[rand_idxs], 
                            self.num_features,
                            idxs=np.array(range(self.sample_size_braf)),
                            feature_idxs = feature_idxs,
                            depth = self.depth,
                            min_leaf =  self.min_leaf)

    def predict(self, x):
        # gets the mean of all predictions across the RF
        pred = np.mean([t.predict(x) for t in self.trees], axis = 0)
        
        # returns a 1 or 0 given the predictions
        return [1 if p>0.5 else 0 for p in pred]
    
    def predict_proba(self, x):
        # gets the mean of all predictions across the RF
        pred = np.mean([t.predict(x) for t in self.trees], axis = 0)
        
        # returns 'probability' that a 1 or 0 will be selected
        return [[1.-p,p] for p in pred] 

class RandomForest():
    def __init__(self, x ,y, num_trees, num_features, sample_size=None, depth=10, min_leaf=5):
        np.random.seed(42) # keep same randoms for testing and reproducibility
        if sample_size is None:
            self.sample_size = len(y)
        self.x = x
        self.y = y
        self.num_trees = num_trees
        self.num_features = num_features
        # number of decision splits
        self.depth = depth
        # minimum number of rows required in a node to cause further split. 
        self.min_leaf = min_leaf 
        
        # deciding how many features to include
        if self.num_features == 'sqrt':
            self.num_features = int(np.sqrt(x.shape[1]))
        elif self.num_features == 'log2':
            self.num_features = int(np.log2(x.shape[1]))
        else:
            self.num_features = num_features
        
        # creating tree instances
        self.trees = [self.create_tree() for i in range(num_trees)]
        
    def create_tree(self):
        # This is effectively bootstrapping sample_size number of samples in place
        rand_idxs = np.random.choice(len(self.y), replace = True, size = self.sample_size)

        # This gets us the features we've decided to use
        feature_idxs = np.random.choice(self.x.shape[1], size = self.num_features)
        
        # returns a DecisionTree using the randomized indexes selected and min_leaf
        return DecisionTree(self.x.iloc[rand_idxs], 
                            self.y[rand_idxs], 
                            self.num_features,
                            idxs=np.array(range(self.sample_size)),
                            feature_idxs = feature_idxs,
                            depth = self.depth,
                            min_leaf =  self.min_leaf)
    
    def predict(self, x):
        # gets the mean of all predictions across the RF
        pred = np.mean([t.predict(x) for t in self.trees], axis = 0)
        
        # returns a 1 or 0 given the predictions
        return [1 if p>0.5 else 0 for p in pred]
    
    def predict_proba(self, x):
        # gets the mean of all predictions across the RF
        pred = np.mean([t.predict(x) for t in self.trees], axis = 0)
        
        # returns 'probability' that a 1 or 0 will be selected
        return [[1.-p,p] for p in pred]    

class DecisionTree():
    def __init__(self, x, y, num_features, idxs, feature_idxs, depth, min_leaf):
        self.x = x
        self.y = y
        self.num_features = num_features
        self.depth = depth
        self.idxs = idxs
        self.feature_idxs = feature_idxs
        self.min_leaf = min_leaf
        self.n_rows = len(idxs)
        self.n_cols = x.shape[1]
        self.val = np.mean(y[idxs])
        # checks how effective a split of a tree node is
        self.score = float('inf') 
        # finds which variable to split on
        self.find_varsplit() 
      
    def find_varsplit(self):
        # find best split in current tree
        for i in self.feature_idxs:
            self.find_best_split(i)
        
        # check for leaf node so no split to be made
        if self.is_leaf:
            return
            
        # if it is note a leaf node, we need to create an lhs and rhs
        x = self.split_col
        # np.nonzero gets a boolean array, but turns it into indexes of the Truths
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        
        # get random features
        lhs_feat = np.random.choice(self.x.shape[1], size = self.num_features)
        rhs_feat = np.random.choice(self.x.shape[1], size = self.num_features)
        
        self.lhs_tree = DecisionTree(self.x, 
                                self.y, 
                                self.num_features, 
                                self.idxs[lhs], 
                                lhs_feat, 
                                depth = self.depth-1,
                                min_leaf = self.min_leaf)
        self.rhs_tree = DecisionTree(self.x,
                                self.y,
                                self.num_features,
                                self.idxs[rhs],
                                rhs_feat,
                                depth = self.depth-1,
                                min_leaf = self.min_leaf)
       
    def find_best_split(self, var_idx):
        # takes variable index (var_idx) and finds if it's a better split than we have so far
        # since the initial score is set to infinity, it will always be more beneficial to create a split
        # we'll be doing this in a greedy way with O(n) runtime
        
        # get all the rows that we're considering for this tree, but only the particular variable we're looking at
        x = self.x.values[self.idxs, var_idx]
        y = self.y[self.idxs]
        
        # get the indices of the sorted data
        sort_idx = np.argsort(x)
        
        sort_x = x[sort_idx]
        sort_y = y[sort_idx]
        
        # loop over all data entries possible for this tree
        for i in range(0, self.n_rows - self.min_leaf-1):
            # making sure we're not on a leaf node and skipping over same, bootstrapped values in the data
            if i < self.min_leaf or sort_x[i] == sort_x[i+1]:
                continue
            lhs = np.nonzero(sort_x <= sort_x[i])[0]
            rhs = np.nonzero(sort_x > sort_x[i])[0]
            
            # if we're looking at a split with no rhs, skip it because it's not a split
            if rhs.sum()==0: 
                continue
            
            gini = calc_gini(lhs, rhs, sort_y)
            
            # check if this split's gini is better than another splith
            if gini < self.score:
                self.var_idx = var_idx
                self.score = gini
                self.split = sort_x[i]
    
    # function that's calculate 'on the fly' and can be used without parentheses
    # @ is a 'decorator'
    @property
    def split_name(self):
        return self.x.columns[self.var_idx]
    
    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]
    
    @property
    def is_leaf(self):
        return self.score == float('inf')
    
    def __repr__(self):
        # This function helps us to get a better representation of the objects when we print them out
        s = f'n: {self.n_rows}; val: {self.val}'
        if not self.is_leaf:
            # if this node isn't a leaf, print out the score, split, and name on which it split
            s += f'; score: {self.score}; split: {self.split}; var: {self.split_name}'
        return s
       
    def predict(self, x):
        # get predictions of the tree are the predictions for each row in an array
        # loops through rows because x is matrix and the leading axis is 0 meaning row
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self, xi):
        # predictions for each row
        
        # check if leaf node
        if self.is_leaf:
            return self.val
        
        # if variable in row xi is <= split value, go down left tree, otherwise right tree
        best = self.lhs_tree if xi[self.var_idx] <= self.split else self.rhs_tree
        return best.predict_row(xi)

class Metrics():
    # This class holds all the metrics that we use
    def __init__(self, y_true, pred, prob):
        self.y_true = y_true
        self.pred = pred
        self.prob = prob

    def compute_confusion_matrix(self):
        # Computes a confusion matrix using numpy for two np.arrays true and pred
        
        # Number of classes
        c = len(np.unique(self.y_true)) 
        
        # makes sure confusion matrix is at least a 2x2
        if c <=1 :
            c = 2
        
        cm = np.zeros((c, c))
        
        for a, p in zip(self.y_true, self.pred):
            cm[int(a)][int(p)] += 1
        
        return cm


    def calc_precision_recall(self, conf_mat):
        # calculate the precision, recall, and specificity given a confusion matrix
        # This formulation can be applied to multiple classes
    
        TP = conf_mat.diagonal()
        FP = np.sum(conf_mat, axis=0) - TP
        FN = np.sum(conf_mat, axis=1) - TP
        TN = conf_mat.sum() - (FP + FN + TP)
        
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        # we don't use this for our purpose, but good to have it just in case
        specificity = TN / (TN + FP)

        # returns metrics for 1 class since we have a binary classification problem
        return precision[1], recall[1], specificity[1]


    def calc_roc_prc(self):
        # calculates the values for the ROC and PRC curves

        # false positive rate
        fpr = []
        # true positive rate
        tpr = []
        # precision rate
        prec = []
        # Iterate thresholds from 0.0, 0.01, ... 1.0
        thresholds = np.arange(0.0, 1.01, .01)

        # get number of positive and negative examples in the dataset
        P = sum(self.y_true)
        N = len(self.y_true) - P

        # iterate through all thresholds and determine fraction of true positives
        # and false positives found at this threshold
        for thresh in thresholds:
            FP=0
            TP=0
            for i in range(len(self.prob)):
                if (self.prob[i] > thresh):
                    if self.y_true[i] == 1:
                        TP = TP + 1
                    if self.y_true[i] == 0:
                        FP = FP + 1
            
            # checks to make sure there are no undefined values and sets them to 1
            if N == 0:
                fpr.append(1)
            else:
                fpr.append(FP/float(N))
            
            if P == 0:
                tpr.append(1)
            else:
                tpr.append(TP/float(P))
            
            if (TP+FP) == 0:
                prec.append(1)
            else:
                prec.append(TP/(TP+FP))
        
        # return values in ascending order
        return np.array(tpr)[::-1], np.array(fpr)[::-1], np.array(prec)[::-1]


    def AUC(self,x,y):
        # return Area Under Curve using the trapezoidal rule
        return np.trapz(y,x)


def calc_gini(left, right, y):
    # this function calculates the gini score and will be the decision maker for splitting trees
    # this is the default in sklearn, but we can use other scoring functions if we wish
    
    classes = np.unique(y)
    
    n = len(left) + len(right)
    s1 = 0
    s2 = 0
    
    for cls in classes:
        # find probability of each class and add the square to s1/s2
        p1 = len(np.nonzero(y[left] == cls)[0]) / len(left)
        s1 += p1*p1
        
        p2 = len(np.nonzero(y[right] == cls)[0]) / len(right)
        s2 += p2*p2
        
    # weighted average of (1-sum_left) and (1-sum_right)
    gini = (1-s1)*(len(left)/n) + (1-s2)*(len(right)/n)
    
    return gini


def euclidean_distance(row1,row2,length):
    # calculates Euclidean distance between two data entries with i columns
    
    distance = 0.0
    
    for i in range(length):
        distance += (row1[i] - row2[i])**2.
    
    return np.sqrt(distance)


def get_KNN(dataset, test_row, K):
	# function to get K nearest neighbors from pandas df

    distances = []
    
    length = dataset.shape[1]
    # loop over all rows in dataset
    for i in range(dataset.shape[0]):
        # calculate distance to each row
        dist = euclidean_distance(test_row, dataset.iloc[i], length)
        distances.append((dataset.iloc[i], dist))
    
    # sort the distances
    distances.sort(key=lambda x: x[1])
    
    # get the K closest distances
    # since for our purposes we're using a separate dataset, 
    # we're not worried about the same row being it's own neighbor
    neighbors = distances[:K]
    
    # returns the row and the distance
    return neighbors


def get_folds(X, K):
    # This function is a modification from sklearn to get the indices of K folds
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    
    n_folds = K
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=np.int)
    fold_sizes[:n_samples % n_folds] += 1
    
    folds = []
    
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
        
    return folds


def get_folds_braf(X, X_crit, K):
    # This function is a modification from sklearn to get the indices of K folds
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    n_samples_crit = len(X_crit)
    indices_crit = np.arange(n_samples_crit)
    
    
    n_folds = K
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=np.int)
    fold_sizes[:n_samples % n_folds] += 1
    
    fold_sizes_crit = np.full(n_folds, n_samples_crit // n_folds, dtype=np.int)
    fold_sizes_crit[:n_samples_crit % n_folds] += 1
    
    
    folds = []
    folds_crit = []
    
    current = 0
    for i in range(len(fold_sizes)):
        start, stop = current, current + fold_sizes[i]
        folds.append(indices[start:stop])
        current = stop
        
        crit_idx = []
        j = 0
        
        # collect indices from the critical dataset until you find enough for each critical fold
        # that satisfy the below conditions
        while len(crit_idx) < fold_sizes_crit[i]:
           
            # check if the current row from the critical data set is not in the current fold
            # for the regular dataset and not in a previous critical fold
            if j < X_crit.shape[0] and (X.iloc[indices[start:stop]] == X_crit.loc[j]).any().all() == False \
            and all(j not in lst for lst in folds_crit):
                # if not, append its index to the list that contains the indices for this critical fold
                crit_idx.append(j)
            
            # if we've used up all the entries, choose some at random
            # this is not ideal and there might be some overlap between the 
            # regular and critical folds.  This will likely only happen on the 
            # last fold and shouldn't occur for many values
            elif j >= X_crit.shape[0]:
                j_rand = random.randrange(0, X_crit.shape[0], 1)
                if (X.iloc[indices[start:stop]] == X_crit.loc[j_rand]).any().all() == False \
                and j_rand not in crit_idx:
                    crit_idx.append(j_rand)
                    
            j+=1

        folds_crit.append(np.array(crit_idx))
        
    return folds, folds_crit


def get_BRAF_df(dataset, col, mincls, majcls, knn):
    # function for getting the critical dataset for BRAF
    
    # define initial critical dataset as the minority class dataset
    df_crit = dataset.loc[dataset[col] == mincls]
    # define majority dataset
    df_maj = dataset.loc[dataset[col] == majcls]
    # define minority dataset
    df_min = dataset.loc[dataset[col] == mincls]
    
    # loop over the minority dataset to get KNN and add them to the critical dataset
    for i in range(df_min.shape[0]):
    	# Get k nearest neighbors
        Tnn = get_KNN(df_maj, df_min.iloc[i], knn)

        for j in range(len(Tnn)-1):
        	# Check if this particular neighbor is in the critical dataset
            if (df_crit == Tnn[j][0]).all(1).any() == False:
                df_crit = df_crit.append(Tnn[j][0],ignore_index=True)

    df_crit = df_crit.sample(frac=1).reset_index(drop=True)
    return df_crit


def get_cv_preds_braf(X, Y, X_braf, Y_braf, k_folds, n_trees, n_trees_braf, n_feat, dep, m_leaf):
    # This is a function for evaluating k-folds CV and returning the true y, predictions, and probabilities
    # it works for a BiasedRandomForest class


    folds, folds_braf = get_folds_braf(X, X_braf, k_folds)
    
    # scores are ordered in precision, recall, AUROC, AUPRC, TPR, FPR, prec_rate
    y_pred_prob = []
    for i in range(len(folds)):
        # indices for the validation subset
        val = X.index.isin(folds[i])
        val_braf = X_braf.index.isin(folds_braf[i])
        
        # separate the dataset into train and test for each CV
        X_train = X.iloc[~val]
        X_test = X.iloc[val]
        
        Y_train = Y.iloc[~val]
        Y_test = Y.iloc[val] 

        # braf
        X_train_braf = X_braf.iloc[~val_braf]
        
        Y_train_braf = Y_braf.iloc[~val_braf]
        

        # call BiasedRandomForest Class
        ens = BiasedRandomForest(X_train, 
                                np.array(Y_train), 
                                X_train_braf, 
                                np.array(Y_train_braf), 
                                num_trees=n_trees, 
                                num_trees_braf=n_trees_braf, 
                                num_features=n_feat, 
                                depth=dep, 
                                min_leaf=m_leaf)


        # the predictions and their probabilities
        # we only use the X_test values for the CV (rather than X_test+X_test_braf) 
        # since they most closely resemble the test dataset that we'll use when evaluating the model
        pred_RF = ens.predict(X_test.values)
        prob_RF = ens.predict_proba(X_test.values)
    
        
        y_pred_prob.append([np.squeeze(np.array(Y_test)),np.array(pred_RF),np.array(prob_RF)[:,1]])
        
    return y_pred_prob


def get_cv_preds(X, Y, k_folds, n_trees, n_feat, dep, m_leaf):
    # This is a function for evaluating k-folds CV and returning the true y, predictions, and probabilities
    # it works for the RandomForest class
    
    folds = get_folds(X, k_folds)
    
    # scores are ordered in precision, recall, AUROC, AUPRC, TPR, FPR, prec_rate
    y_pred_prob = []
    for i in range(len(folds)):
        # indices for the validation subset
        val = X.index.isin(folds[i])
        
        # separate the dataset into train and test for each CV
        X_train = X.iloc[~val]
        X_test = X.iloc[val]
        
        Y_train = Y.iloc[~val]
        Y_test = Y.iloc[val] 
        
        # call RandomForest Class
        ens = RandomForest(X_train, 
                           np.array(Y_train), 
                           num_trees=n_trees, 
                           num_features=n_feat, 
                           depth=dep, 
                           min_leaf=m_leaf)
        
        # the predictions and their probabilities
        pred_RF = ens.predict(X_test.values)
        prob_RF = ens.predict_proba(X_test.values)
    
        
        y_pred_prob.append([np.squeeze(np.array(Y_test)),np.array(pred_RF),np.array(prob_RF)[:,1]])
        
    return y_pred_prob


def get_cv_scores(y_pred_prob):
    # calculate the scores given the true y values, predictions, and probabilites

    # scores are ordered in precision, recall, AUROC, AUPRC, TPR, FPR, prec_rate
    scores = []
    for i in range(len(y_pred_prob)):
        met = Metrics(y_pred_prob[i][0], y_pred_prob[i][1], y_pred_prob[i][2])

        # confusion matrix
        cm = met.compute_confusion_matrix()
        # precision recall
        precision, recall, _ = met.calc_precision_recall(cm)
        # TPR, FPR, 'precision rate'
        tpr, fpr, prec = met.calc_roc_prc()
        # AUROC and AUPRC
        auprc = met.AUC(tpr,prec)
        auroc = met.AUC(fpr,tpr)
        
        scores.append([precision,recall,auroc,auprc,tpr,fpr,prec])
    
    return scores


def plot_PRC(rec,prec,label,AUC):
	# plots and saves PRC plot 
	# label is to differentiate CV and test sets

    fig, ax = plt.subplots(figsize= [12,10])
    ax.set_xlabel('Recall', fontsize=23)
    ax.set_ylabel('Precision', fontsize=23)
    ax.set_title(f'{label} PRC Curve with AUPRC: {AUC:.4f}', fontsize=27)
    ax.plot(rec,prec,linewidth=3,color='b')
    ax.spines['top'].set_linewidth(2.3)
    ax.spines['left'].set_linewidth(2.3)
    ax.spines['right'].set_linewidth(2.3)
    ax.spines['bottom'].set_linewidth(2.3)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        
    # plt.savefig(f'{label}_PRC_Curve',fmt='png')


def plot_ROC(fpr,tpr,label,AUC):
	# plots and saves ROC plot 
	# label is to differentiate CV and test sets

    fig, ax = plt.subplots(figsize= [12,10])
    ax.set_xlabel('False Positive Rate', fontsize=23)
    ax.set_ylabel('True Positive Rate', fontsize=23)
    ax.set_title(f'{label} ROC Cure with AUROC: {AUC:.4f}', fontsize=27)
    ax.plot(fpr,tpr,linewidth=3,color='r')
    ax.spines['top'].set_linewidth(2.3)
    ax.spines['left'].set_linewidth(2.3)
    ax.spines['right'].set_linewidth(2.3)
    ax.spines['bottom'].set_linewidth(2.3)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    
    # plt.savefig(f'{label}_ROC_Curve',fmt='png')




