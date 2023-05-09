import gurobipy as gp
from gurobipy import GRB

import numpy as np 
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, List, Union, Dict

from mip import  maximize, minimize, xsum
from mip import Model, BINARY, OptimizationStatus

class FairnessEvaluator:
    METRIC_FORMULAS = {'fnr': 'fn / (fn + tp)',
                       'fpr': 'fp / (fp + tn)',
                       'es': 'tp+fp',
                       'di': '(tp + fp) / (tp + fp + tn + fn)',
                       'prec': 'tp / (tp + fp)',
                       'tpr': 'tp / (tp + fn)',
                       'recall': 'tp / (tp + fn)'}
                       
    DEFAULT_METRICS = ['fnr', 'fpr','es']

    def __init__(self,
                 X: pd.DataFrame,
                 label: str,
                 protected_attrs: List[str],
                 reference_groups: Dict = None,
                 metrics: List[str]=DEFAULT_METRICS,
                 metrics_to_plot: List[str] = None,
                 fairness_bounds: List[float] = [0.8,1.2],
                 precision_ub: float = 1.0,
                 fully_constrained: bool = True
                 ):
        """
        FairnessEvaluator is the class that will evaluate and plot your results.

        :param X: pandas DataFrame, dataset to be processed by FairnessEvaluator. Must have contain a column with the protected attribute and a column called label_value containing the label
        :param label: str, name of label column
        :param protected_attrs: List of strs, names of protected attribute columns in pandas DataFrame
        :param reference_groups: Dict mapping values from the respective protected attributes that fairness values will be referenced to, default is first value encountered in protected attribute column, ex. {sex: ['female']}
        :param metrics: n-element array with fairness metrics to be included, currently supports: es - equal selection, di - disparate impact, fnr - false negative rate, fpr - false positive rate, default = ['es','fpr','fnr']
        :param metrics_to_plot: fairness metrics to be plotted, supports same as metrics, default = metrics
        :param metric_fairness_bounds: dictionary with specific bounds for metrics, ex. {'es'=[0.5,1.5], 'fpr'=[0.9,1.1]}
        :param precision_ub: float in (0,1.0], upper limit of precision score, default = 1.0
        :param fully_constrained: bool, if True, create fairness constraints using each group as a reference, if false only use the defined reference group, default = True
        :return: A FairnessOptimzer instance
        """ 

        # encode if value is string
        self.encode_attr = {}
        for attr in protected_attrs+[label]:
            if X.dtypes[attr]==object:
                attr_map = {value:encode_value for encode_value,value in enumerate(X[attr].unique())}
                X[attr] = X[attr].map(lambda x: attr_map[x])
                self.encode_attr[attr]=attr_map
                
        self.X = X
        self.label = label
        self.protected_attrs = protected_attrs

        if len(protected_attrs) > 1:
            self.__createIntersectionalFeature('only')

        self.protected_attr_values = {}
        for attr in self.protected_attrs:
            self.protected_attr_values[attr] = list(set(self.X[attr]))
        
        self.protected_attr_binary = {}
        for attr in self.protected_attrs:
            if len(self.protected_attr_values[attr]) > 2:
                self.protected_attr_binary[attr] = False
            else:
                self.protected_attr_binary[attr] = True
        
        self.metrics = metrics
        
        if metrics_to_plot == None:
            self.metrics_to_plot = metrics
        else:
            self.metrics_to_plot = metrics_to_plot
            
        self.fairness_bounds = fairness_bounds
        
        self.lb = fairness_bounds[0]
        self.ub = fairness_bounds[1]

        self.precision_ub = precision_ub
        self.fully_constrained = fully_constrained

        if reference_groups == None:
            self.reference_groups = {}
            for attr in self.protected_attrs:
                self.reference_groups[attr] = self.protected_attr_values[attr][0]
        
        self.entities = self.X.index.tolist()
        self.label_values = self.X[label].tolist()
        self.results_df = None

        self.results_columns = [ "k","k_perc","optimal","precision_unconstrained",
                                 "recall_unconstrained","precision","recall","group_confusion_matrix"]
    
    def evaluate(self,
                 k_range: List[int]=[5,100],
                 k_step: int=5) -> pd.DataFrame:
        """
        Run the FairnessEvaluator over a range of k (in percentage)

        :param k_range: 2-element array containing the k-range to be evaulated, in percentage, default = [5,100]
        :param k_step: int, the step for the k range, default = 5
        :return: A pandas DataFrame containing the results of the FairnessEvaluator, also stored in self.results_df
        """ 
        _ = self.__createResultsDf()

        self.xlim = k_range[1]

        pred_final_list_func = lambda row: 1 if row.name in final_list else 0

        for k_perc in range(k_range[0],k_range[1],k_step):
            r = {}
            r['k_perc'] = k_perc
            k = np.floor(self.X.shape[0]*(k_perc/100))
            r['k'] = k

            # unconstrained_model
            num_of_positives = sum(self.label_values)
            if k*self.precision_ub <= num_of_positives:
                r['precision_unconstrained'] = self.precision_ub
                r['recall_unconstrained'] = k*self.precision_ub / num_of_positives
            else:
                r['precision_unconstrained'] = num_of_positives / k
                r['recall_unconstrained'] = 1              

            # constrained_model
            final_list, model, optimal = self.__runNewModel(self.protected_attrs[0], k)

            if optimal:
                r['optimal'] = 1    
                df_t = self.X.copy()
                df_t['pred'] = df_t.apply(pred_final_list_func, axis = 1)
                r['precision'] = precision_score(df_t[self.label],df_t['pred'])
                r['recall'] = recall_score(df_t[self.label], df_t['pred'])

                # Get confusion matrix for each group
                group_confusion_mat = {}
                for v in self.protected_attr_values[self.protected_attrs[0]]:
                    df_temp = df_t.copy()
                    df_temp = df_temp[df_temp[self.protected_attrs[0]] == v]
                    tn, fp, fn, tp = confusion_matrix(df_temp[self.label],df_temp['pred']).ravel()
                    group_confusion_mat[v] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
                r['group_confusion_matrix'] = group_confusion_mat
                
                for m in self.metrics_to_plot:
                    for attr in self.protected_attrs:
                        if self.protected_attr_binary[attr]:
                            key = m + '_' + attr
                            key_values = key + '_values'
                            r[key] = self.__getMetricDisparity(df_t, attr, m)[1]

                        else:
                            key_min = m + '_' + attr + '_min'
                            key_max = m + '_' + attr + '_max'
                            r[key_min] = min(self.__getMetricDisparity(df_t, attr, m).values())
                            r[key_max] = max(self.__getMetricDisparity(df_t, attr, m).values())
            else:
                r['optimal'] = 0

            self.results_df = self.results_df.append(r, ignore_index=True)
        
        return self.results_df

    def plot(self,
             x_scale_perc: bool=True):
        """
        Prints the plot the results the FairnessEvaluator over a range of k (in percentage)

        :param x_scale_perc: bool, if True then x-axis will labeled with k%, otherwise labeled with k, default = True
        :return: None
        """ 
        if self.results_df is None:
            raise Exception("FairnessEvaluator.plot() must be preceded by a succesful execution of FairnessEvaluator.evaluate()")

        if x_scale_perc:
            x_scale = 'k_perc'
        else:
            x_scale = 'k'

        fig, axes = plt.subplots(2,1)
        ax0 = sns.lineplot(ax=axes[0], x=x_scale, y='precision', color='red', zorder=10, data=self.results_df, label = "Precision")
        ax0 = sns.lineplot(ax=axes[0], x=x_scale, y='precision_unconstrained', zorder=1, color='red', alpha=0.5, linestyle = 'dashed', label= "Precision (Unc.)", data=self.results_df)
        ax0 = sns.lineplot(ax=axes[0], x=x_scale, y='recall', color='blue', zorder=10, data=self.results_df, label = "Recall")
        ax0 = sns.lineplot(ax=axes[0], x=x_scale, y='recall_unconstrained', zorder=1, color='blue', alpha=0.5, linestyle = 'dashed', label = "Recall (Unc.)", data=self.results_df)
        ax0.set_ylabel('%')
        ax0.set_ylim(0,1.01)
        if x_scale_perc:
            ax0.set_xlim(0,self.xlim)
        else:
            ax0.set_xlim(0,(self.X.shape[0])*(self.xlim/100))
        ax0.legend(bbox_to_anchor=(1.01, 1))

        for m in self.metrics_to_plot:
            for attr in self.protected_attrs:
                if self.protected_attr_binary[attr]:
                    key = m + '_' + attr
                    ax1 = sns.lineplot(ax=axes[1], x=x_scale, y=key, data=self.results_df, label=key)
                else:
                    key_min = m + '_' + attr + '_min'
                    key_max = m + '_' + attr + '_max'
                    ax1 = sns.lineplot(ax=axes[1], x=x_scale, y=key_min, data=self.results_df, label=key_min)
                    ax1 = sns.lineplot(ax=axes[1], x=x_scale, y=key_max, data=self.results_df, label=key_max)

        ax1.axhline(y = 1.2, color = 'grey', linestyle = 'dashed')
        ax1.axhline(y = 0.8, color = 'grey', linestyle = 'dashed')

        ax1.set_ylabel('Disparity')     
        if x_scale_perc:
            ax1.set_xlim(0,self.xlim)
            ax1.set_xlabel('k% (percentage of entities)')

        else:
            ax1.set_xlim(0,(self.X.shape[0])*(self.xlim/100))
            ax1.set_xlabel('k (number of entities)')

        ax1.legend(bbox_to_anchor=(1.01, 1))
    
    def __createIntersectionalFeature(self, intersectionality_option):
        '''
        Creates the intersectional feature for the dataset, saves it to self.X
          if the intersectionality_option is 'full' it adds the new feature to self.protected_attrs
          if the option is 'only', it replaces the self.protected_attrs list with a only the new feature

        :return: None
        '''
        inter_col_name = 'inter'
        for attr in self.protected_attrs:
            inter_col_name = inter_col_name + '_' + str(attr)

        #def __createIntersection(row, attr_list):
        #    value = ''
        #    for attr in attr_list:
        #        value = value + str(row[attr])
        #    return value

        #self.X[inter_col_name] = self.X.apply(__createIntersection, attr_list=self.protected_attrs, axis=1)
        
        self.X[inter_col_name] = self.X.apply(lambda row, attr_list: ''.join([str(row[attr]) for attr in attr_list]), attr_list=self.protected_attrs, axis = 1)
        
        if intersectionality_option == 'full':
            self.protected_attrs.append(inter_col_name)
        elif intersectionality_option == 'only':
            self.protected_attrs = [inter_col_name]

    def __runModel(self,
                   k: int,
                   with_constraints: bool = False) -> Union[List[int],object]:
        """"
        Run fairness-constrained MIP

        :param k: int, how many elements to select
        :return: selected - the entities selected by the MIP
                 model - the MIP object
                 optimal - bool, True if the MIP found an optimal solution
        """ 

        x, model = self.__initModel(entities =self.entities,
                                    label_values=self.label_values,
                                    precision_ub=self.precision_ub,
                                    k=k)

        if with_constraints:
            model = self.__ModelConstraints(x, model)

        status = model.optimize()
        optimal = (status == OptimizationStatus.OPTIMAL)
        
        try:
            selected = [i for i in self.entities if x[i].x >= 0.99]
        except:
            selected = []
            
        return selected, model, optimal

    def __getMetricDisparity(self,
                             X_in: pd.DataFrame,
                             attr: str,
                             formula: str) -> Dict[str,Any]:
        """"
        Evaluate the disparity based on a fairness metric, referenced to self.reference_group

        :param X_in: pandas DataFrame, the dataframe to be evaluated
        :param attr: attribute name
        :param formula: str, one of the supported formulas for calcuating metric disparities, see FairnessEvaluator.metric_formulas
        :return: a dictionary containing the disparities
        """
        metric = {}
        metric_disparity = {}
        for v in self.protected_attr_values[attr]:
            X_temp = X_in[X_in[attr] == v]
            tn, fp, fn, tp = confusion_matrix(X_temp[self.label],X_temp['pred'] ).ravel()
            metric[v] = eval(self.METRIC_FORMULAS[formula])
        for v in self.protected_attr_values[attr]:
            if metric[v] == metric[self.reference_groups[attr]]:
                metric_disparity[v] = 1.0
            else:
                metric_disparity[v] = metric[v] / metric[self.reference_groups[attr]]
        return metric_disparity 

    def __createResultsDf(self):
        """"
        Create the results dataframe for the FairnessEvaluator

        :return: an empty results dataframe, stored in self.results_df
        """
        for attr in self.protected_attrs:
            if self.protected_attr_binary[attr]:
                for m in self.metrics_to_plot:
                    self.results_columns.append(m + '_' + attr)
            else:
                for m in self.metrics_to_plot:
                    self.results_columns.append(m + '_' + attr + '_min')
                    self.results_columns.append(m + '_' + attr + '_max')

        self.results_df = pd.DataFrame(
                    columns=(self.results_columns)
        )

        return self.results_df

    def __createGroupVariables(self, attr):
        """"
        Create helper variables used for defining in constraints in FairnessEvaluator.__runConstrainedModel()

        :return: groups - a dictionary labeled by group values that contains a binary array with lenght equal to the number of entites, where an entry of 1 indicates entity membership of that group
                 groups_n - a dictionary containing the number in each group
                 groups_positives_n - a dictionary containing the number of positives in each group
                 groups_negatives_n - a dictionary containing the number of negatives in each group
        """
        groups = {}
        groups_n = {}
        groups_positives_n = {}
        groups_negatives_n = {}
        for v in self.protected_attr_values[attr]:
            groups[v] = np.where(self.X[attr] == v, 1, 0)
            groups_n[v] = sum(groups[v])
            groups_positives_n[v] = self.X[self.X[attr] == v][self.label].sum()
            groups_negatives_n[v] = groups_n[v] - groups_positives_n[v]
        
        return groups, groups_n, groups_positives_n, groups_negatives_n 

    def __runNewModel(self,attr,k):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        env.start()

        group_size_bounds = {x:{} for x in self.protected_attr_values[attr]}
        for type_obj in [GRB.MAXIMIZE,GRB.MINIMIZE]:
            for g_x in self.protected_attr_values[attr]:

                model = gp.Model("Size_bounds_model",env=env)
                x = [model.addVar(name=f"x_{i}",vtype=GRB.BINARY) for i in range(len(self.entities))]
                groups, groups_n, groups_positives_n, groups_negatives_n = self.__createGroupVariables(attr)

                obj = sum(groups[g_x][i] * x[i] for i in self.entities)
                model.setObjective(obj,type_obj)
                model.addConstr(sum(x[i] for i in self.entities) == k,'k-list')
                model.addConstr(sum(self.label_values[i]*x[i] for i in self.entities) <= self.precision_ub * k,'precision_ub')
                
                groups, groups_n, groups_positives_n, groups_negatives_n = self.__createGroupVariables(attr)

                if self.fully_constrained:
                    g_x2_list = self.protected_attr_values[attr]
                else:
                    g_x2_list = [self.reference_groups[attr]]

                for g_x1 in self.protected_attr_values[attr]:
                    for g_x2 in g_x2_list:
                        if g_x1 != g_x2:
                            model.addConstr((1 - ((1/groups_positives_n[g_x1]) * sum( [ x[i]*self.label_values[i]*groups[g_x1][i] for i in self.entities ] ))) <= self.ub * ( 1 - ((1/groups_positives_n[g_x2]) * sum( [ x[i]*self.label_values[i]*groups[g_x2][i] for i in self.entities ] ))), f'fnr_ub_{attr}' )
                            model.addConstr((1 - ((1/groups_positives_n[g_x1]) * sum( [ x[i]*self.label_values[i]*groups[g_x1][i] for i in self.entities ] ))) >= self.lb * ( 1 - ((1/groups_positives_n[g_x2]) * sum( [ x[i]*self.label_values[i]*groups[g_x2][i] for i in self.entities ] ))),f'fnr_lb_{attr}' )

                for g_x1 in self.protected_attr_values[attr]:
                    for g_x2 in g_x2_list:
                        if g_x1 != g_x2:
                            model.addConstr((1 - ((1/groups_negatives_n[g_x1]) * sum( [ (1-x[i])*(1-self.label_values[i])*groups[g_x1][i] for i in self.entities ] ))) <= self.ub * ( 1 - ((1/groups_negatives_n[g_x2]) * sum( [ (1-x[i])*(1-self.label_values[i])*groups[g_x2][i] for i in self.entities ] ))),f'fpr_ub_{attr}')
                            model.addConstr((1 - ((1/groups_negatives_n[g_x1]) * sum( [ (1-x[i])*(1-self.label_values[i])*groups[g_x1][i] for i in self.entities ] ))) >= self.lb * ( 1 - ((1/groups_negatives_n[g_x2]) * sum( [ (1-x[i])*(1-self.label_values[i])*groups[g_x2][i] for i in self.entities ] ))),f'fpr_lb_{attr}' )
                
                model.optimize()
                try:
                    if type_obj == GRB.MAXIMIZE:
                        group_size_bounds[g_x]['upper'] = sum([ groups[g_x][i] if x[i].X >= 0.99 else 0 for i in self.entities ])
                    else: 
                        group_size_bounds[g_x]['lower'] = sum([ groups[g_x][i] if x[i].X >= 0.99 else 0 for i in self.entities ])
                except:
                    if type_obj == GRB.MAXIMIZE:
                        group_size_bounds[g_x]['upper'] = -1
                    else: 
                        group_size_bounds[g_x]['lower'] = -1

        group_ppv_bounds = {x:{} for x in self.protected_attr_values[attr]}
        for type_obj in [GRB.MAXIMIZE,GRB.MINIMIZE]:
            for g_x in self.protected_attr_values[attr]:
                model = gp.Model("PPV_bounds_model", env=env)
                groups, groups_n, groups_positives_n, groups_negatives_n = self.__createGroupVariables(attr)
                y = [model.addVar(name=f"x_{i}",vtype=GRB.CONTINUOUS) for i in range(len(self.entities)+1)]
                
                obj = sum(groups[g_x][i] * self.label_values[i] * y[i+1] for i in self.entities)
                model.setObjective(obj,type_obj)

                model.addConstr(sum(y[i+1] for i in self.entities) -k*y[0] ==0,'k-list') # List size constraint
                model.addConstr(sum(groups[g_x][i]*y[i+1] for i in self.entities) == 1,'fractional-prog-constraint')
                model.addConstr(sum(self.label_values[i]*y[i+1] for i in self.entities) <= self.precision_ub * k * y[0], 'precision_ub') # Precision constraint

                if self.fully_constrained:
                    g_x2_list = self.protected_attr_values[attr]
                else:
                    g_x2_list = [self.reference_groups[attr]]
                    
                for g_x1 in self.protected_attr_values[attr]:
                    for g_x2 in g_x2_list:
                        if g_x1 != g_x2:
                            model.addConstr((y[0] - ((1/groups_positives_n[g_x1]) * sum( [ y[i+1]*self.label_values[i]*groups[g_x1][i] for i in self.entities ] ))) <= self.ub * ( y[0]  - ((1/groups_positives_n[g_x2]) * sum( [ y[i+1]*self.label_values[i]*groups[g_x2][i] for i in self.entities ] ))), 'fnr_ub')
                            model.addConstr((y[0]  - ((1/groups_positives_n[g_x1]) * sum( [ y[i+1]*self.label_values[i]*groups[g_x1][i] for i in self.entities ] ))) >= self.lb * ( y[0] - ((1/groups_positives_n[g_x2]) * sum( [ y[i+1]*self.label_values[i]*groups[g_x2][i] for i in self.entities ] ))), 'fnr_lb')

                for g_x1 in self.protected_attr_values[attr]:
                    for g_x2 in g_x2_list:
                        if g_x1 != g_x2:
                            model.addConstr((1 - ((1/groups_negatives_n[g_x1]) * sum( [ (1-y[i+1])*(1-self.label_values[i])*groups[g_x1][i] for i in self.entities ] ))) <= self.ub * ( 1 - ((1/groups_negatives_n[g_x2]) * sum( [ (1-y[i+1])*(1-self.label_values[i])*groups[g_x2][i] for i in self.entities ] ))), 'fpr_ub')
                            model.addConstr((1 - ((1/groups_negatives_n[g_x1]) * sum( [ (1-y[i+1])*(1-self.label_values[i])*groups[g_x1][i] for i in self.entities ] ))) >= self.lb * ( 1 - ((1/groups_negatives_n[g_x2]) * sum( [ (1-y[i+1])*(1-self.label_values[i])*groups[g_x2][i] for i in self.entities ] ))), 'fpr_lb')
                
                model.optimize()
                #print(model.status) # TODO: Add handling of other status than optimal
                try:
                    sum_val = sum([groups[g_x][i] * self.label_values[i] * y[i+1].X for i in self.entities ])
                    if type_obj == GRB.MAXIMIZE:

                        group_ppv_bounds[g_x]['upper'] = sum_val
                    else: 
                        group_ppv_bounds[g_x]['lower'] = sum_val
                except:
                    if type_obj == GRB.MAXIMIZE:
                        group_ppv_bounds[g_x]['upper'] = -1
                    else: 
                        group_ppv_bounds[g_x]['lower'] = -1
        
        # Tolerance
        p=6
        groups, groups_n, groups_positives_n, groups_negatives_n = self.__createGroupVariables(attr)

        model = gp.Model("NMDT",env=env)

        z = {group:[model.addVar(name=f"z_{group}_{i}",vtype=GRB.BINARY) for i in range(p)] 
            for group in self.protected_attr_values[attr]}

        lamb = {group: sum([2**(-i-1)*z[group][i] for i in range(p)])
                for group in self.protected_attr_values[attr]}

        y = {group: model.addVar(name=f"y_{group}",vtype=GRB.INTEGER, lb=group_size_bounds[group]['lower'], ub=group_size_bounds[group]['upper'])
            for group in self.protected_attr_values[attr]}

        x = { group: (group_ppv_bounds[group]['upper']-group_ppv_bounds[group]['lower'])*lamb[group] + group_ppv_bounds[group]['lower']
            for group in self.protected_attr_values[attr]}

        obj = sum(group_y*group_x for group_y,group_x in zip(y.values(),x.values()))
        model.setObjective(obj,GRB.MAXIMIZE)

        model.addConstr(sum(y.values()) == k , 'k-list')
        model.addConstr(sum(group_y*group_x for group_y,group_x in zip(y.values(),x.values())) <= k * self.precision_ub, 'precision-ub')

        if self.fully_constrained:
            g_x2_list = self.protected_attr_values[attr]
        else:
            g_x2_list = [self.reference_groups[attr]]

        for g_x1 in self.protected_attr_values[attr]:
            for g_x2 in g_x2_list:
                model.addConstr((1 - ((1/groups_positives_n[g_x1]) * x[g_x1] * y[g_x1])) <= self.ub * ( 1  - ((1/groups_positives_n[g_x2]) * x[g_x2] * y[g_x2])), 'fnr_ub')
                model.addConstr((1 - ((1/groups_positives_n[g_x1]) * x[g_x1] * y[g_x1])) >= self.lb * ( 1  - ((1/groups_positives_n[g_x2]) * x[g_x2] * y[g_x2])), 'fnr_lb')

        for g_x1 in self.protected_attr_values[attr]:
            for g_x2 in g_x2_list:
                model.addConstr((1/groups_negatives_n[g_x1]) * (y[g_x1]-x[g_x1]*y[g_x1]) <= self.ub * (1/groups_negatives_n[g_x2]) * (y[g_x2]-x[g_x2]*y[g_x2]), 'fpr_ub')
                model.addConstr((1/groups_negatives_n[g_x1]) * (y[g_x1]-x[g_x1]*y[g_x1]) >= self.lb * (1/groups_negatives_n[g_x2]) * (y[g_x2]-x[g_x2]*y[g_x2]), 'fpr_lb')

        for g_x1 in self.protected_attr_values[attr]:
            for g_x2 in g_x2_list:
                model.addConstr(x[g_x1] <= self.ub * x[g_x2], 'ppv_ub')
                model.addConstr(x[g_x1] >= self.lb * x[g_x2], 'ppv_ub')

        model.optimize()

        model_status = False
        if model.status == 2:
            model_status = True
        
        try:
            g_tps = {group: int(np.round((y[group]*x[group]).getValue()))
                    for group in self.protected_attr_values[attr]}
            g_sizes = {group: int(np.round(group_y.X)) for group,group_y in y.items()}

            final_list = self.__getFinalList(attr, g_tps, g_sizes)

            if len(final_list) != k:
                model_status = False
        except:
            final_list = []

        return final_list, model, model_status

    def __getFinalList(self,attr,g_tps,g_sizes):
        final_list = []

        for value in self.protected_attr_values[attr]:
            tp_upper_index = int(g_tps[value])
            temp_tp_list = list(self.X[(self.X[self.label] == 1) & (self.X[attr] == value)].iloc[0:tp_upper_index].index)

            fp_upper_index = int((g_sizes[value]-g_tps[value]))
            temp_fp_list = list(self.X[(self.X[self.label] == 0) & (self.X[attr] == value)].iloc[0:fp_upper_index].index)

            final_list = final_list + temp_tp_list + temp_fp_list

        return final_list

    @staticmethod
    def __initModel(entities: List[int],
                    label_values: List[int],
                    precision_ub: float,
                    k) -> Union[List[int],object]:
        """"
        Init MIP model
        :param entities: list of entities values
        :param label_values: list of label values
        :param precision_ub: float in (0,1.0], upper limit of precision score, default = 1.0
        :param k: int, how many elements to select
        :return: x - list of variables
                 model - the MIP object
        """ 
        model = Model()
        x = [model.add_var(var_type=BINARY) for i in range(len(entities))]
        model.objective = maximize(xsum(label_values[i] * x[i] for i in entities))
        model += xsum(x[i] for i in entities) == k # List size constraint
        model += xsum(label_values[i]*x[i] for i in entities) <= precision_ub * k # Precision constraint

        return x,model