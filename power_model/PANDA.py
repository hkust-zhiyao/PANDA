import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import copy
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise 
from sklearn.metrics import mean_absolute_percentage_error

class PANDA:
    
    def __init__(self):
        # there are 15 configurations and 8 workloads, thus 120 samples
        
        # feature: shape is (120, 164) 
        # configuration parameter: 0-13
        # otherlogic (87 events): 14-100
        # dcache (10 events): 101-110
        # icache (6 events): 111-116
        # bp (4 events): 117-120
        # rnu (7 events): 121-127
        # itlb (2 events): 128-129
        # dtlb (2 events): 130-131
        # regfile (5 events): 132-136
        # rob (2 events): 137-138
        # ifu (17 events): 139-155
        # lsu (3 events): 156-158
        # fu_pool (2 events): 159-160
        # isu (3 events): 161-163 
        self.panda_feature = np.load('panda_feature.npy')
        
        # label: shape is (120, 14)
        # 0 is the total power, 1-13 are for the 13 components respectively
        self.panda_label = np.load('panda_label.npy')
        
        # each pair represents the start and end points of the events related to each component 
        self.event_feature_of_components={
            "OtherLogic":[14,101],
            "DCache":[101,111],
            "ICache":[111,117],
            "BP":[117,121],
            "RNU":[121,128],
            "Itlb":[128,130],
            "Dtlb":[130,132],
            "Regfile":[132,137],
            "ROB":[137,139],
            "IFU":[139,156],
            "LSU":[156,159],
            "FU_Pool":[159,161],
            "ISU":[161,164]
        }
        
        # each list represents the configuration parameters related to each component
        self.params_feature_of_components={
            "OtherLogic":[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
            "DCache":[8,10,11,12],
            "ICache":[10,13],
            "BP":[0,7],
            "RNU":[1],
            "Itlb":[1],
            "Dtlb":[11],
            "Regfile":[1,4,5],
            "ROB":[1,3],
            "IFU":[0,1,2,13],
            "LSU":[6,8],
            "FU_Pool":[8,9],
            "ISU":[1,8,9]
        }
        
        # each list represents which configuration parameters (in the list above) should be considered in the resource function, for itlb and fu_pool, no parameter is considered
        self.encode_table={
            "BP":[0],
            "ICache":[0,1],
            "DCache":[0,1],
            "ISU":[0],
            "OtherLogic":[1],
            "IFU":[1],
            "ROB":[1],
            "Regfile":[1,2],
            "RNU":[0],
            "Dtlb":[0],
            "LSU":[0]            
        }
        
        # which model is selected for each component, 0 represents xgboost, 1 represents gradientboost
        self.component_model_selection=[1,0,0,0,0,0,0,1,1,0,1,0,1]
        
        # used for resource function of otherlogic and dtlb
        self.logic_bias = 0
        self.dtlb_bias = 0
        
        return
    
    # train and return a model 
    # input: mod: initial model, train_mod_feature: feature used to train model, train_mod_label: label used to train model
    # return: a trained model
    def train_model(self, mod, train_mod_feature, train_mod_label):
        mod.fit(train_mod_feature,train_mod_label)
        return mod
    
    # just for testing
    def function_testing_train_model(self):
        model = xgb.XGBRegressor()
        train_feat = self.panda_feature
        train_label = self.panda_label[:,0]
        print(train_feat[0])
        print(train_label[0])
        #pred_init = model.predict(train_feat)
        trained = self.train_model(model, train_feat, train_label)
        print(train_feat[0])
        print(train_label[0])
        pred_model = model.predict(train_feat)
        pred_trained = trained.predict(train_feat)
        #print(pred_init)
        print(pred_model)
        print(pred_trained)
        return
    
    def compute_reserve_station_entries(self,decodewidth_init):
        decodewidth = int(decodewidth_init+0.01)
        isu_params = [
            # IQT_MEM.numEntries IQT_MEM.dispatchWidth
            # IQT_INT.numEntries IQT_INT.dispatchWidth
            # IQT_FP.numEntries IQT_FP.dispatchWidth
            [8, decodewidth, 8, decodewidth, 8, decodewidth],
            [12, decodewidth, 20, decodewidth, 16, decodewidth],
            [16, decodewidth, 32, decodewidth, 24, decodewidth],
            [24, decodewidth, 40, decodewidth, 32, decodewidth],
            [24, decodewidth, 40, decodewidth, 32, decodewidth]
            ]
        _isu_params = isu_params[decodewidth - 1]
        return _isu_params[0]+_isu_params[2]+_isu_params[4]
    
    def estimate_bias_logic(self,feature,label):
        feature_list = [int(feature[item]+0.01) for item in range(feature.shape[0])]
        num_of_feature = len(set(feature_list))
        if num_of_feature<=2:
            self.logic_bias = 4
        else:
            reg = LinearRegression().fit(feature.reshape(feature.shape[0],1), label.reshape(label.shape[0],1))
            bias = reg.intercept_
            alpha = reg.coef_[0]
            self.logic_bias = bias / alpha     
        return
    
    def estimate_bias_dtlb(self,feature,label):
        feature_list = [int(feature[item]+0.01) for item in range(feature.shape[0])]
        num_of_feature = len(set(feature_list))
        if num_of_feature<=1:
            self.dtlb_bias = 8
        else:
            reg = LinearRegression().fit(feature.reshape(feature.shape[0],1), label.reshape(label.shape[0],1))
            bias = reg.intercept_
            alpha = reg.coef_[0]
            self.dtlb_bias = bias / alpha      
        return
    
    # transform the label for machine learning part with resource function
    # input: component_name: which component is being processed, feature: feature related to this component, label: label of this component, which is to be transformed 
    # return: transformed label
    def encode_arch_knowledge(self,component_name,feature,label):
        
        if component_name=="BP" or component_name=="ICache" or component_name=="DCache" or component_name=="RNU" or component_name=="ROB" or component_name=="IFU" or component_name=="LSU":
            scale_factor = np.ones(label.shape)
            for i in range(len(self.encode_table[component_name])):
                encode_index = self.encode_table[component_name][i]
                acc_feature = feature[:,encode_index]
                scale_factor = scale_factor * acc_feature
            encode_label = label / scale_factor
        elif component_name=="Regfile":
            scale_factor = np.zeros(label.shape)
            for i in range(len(self.encode_table[component_name])):
                encode_index = self.encode_table[component_name][i]
                acc_feature = feature[:,encode_index]
                scale_factor = scale_factor + acc_feature
            encode_label = label / scale_factor
        elif component_name=="ISU":
            encode_index = self.encode_table[component_name][0]
            decodewidth = feature[:,encode_index]
            reserve_station = np.array([self.compute_reserve_station_entries(decodewidth[i]) for i in range(decodewidth.shape[0])])
            encode_label = label / reserve_station
        elif component_name=="OtherLogic":
            encode_index = self.encode_table[component_name][0]
            self.estimate_bias_logic(feature[:,encode_index],label)
            encode_label = label / (feature[:,encode_index] + self.logic_bias)
        elif component_name=="Dtlb":
            encode_index = self.encode_table[component_name][0]
            self.estimate_bias_dtlb(feature[:,encode_index],label)
            encode_label = label / (feature[:,encode_index] + self.dtlb_bias)
        else:
            encode_label = label / 1.0
        return encode_label
    
    # just for testing
    def function_testing_encode_arch_knowledge(self):
        feature = np.array([[2,2,0.01,0.02,0.03,0.04],[4,4,0.05,0.06,0.07,0.08]])
        label = np.array([100,1000])
        transformed_label = self.encode_arch_knowledge("BP",feature,label)
        print(feature)
        print(label)
        print(transformed_label)
        
        feature1 = np.array([[2,2,0.01,0.02,0.03,0.04],[4,4,0.05,0.06,0.07,0.08]])
        label1 = np.array([100,1000])
        transformed_label1 = self.encode_arch_knowledge("FU_Pool",feature1,label1)
        print(feature1)
        print(label1)
        print(transformed_label1)
        transformed_label1[0] = -100
        print(feature1)
        print(label1)
        print(transformed_label1)
        
        return
    
    # build machine learning part for one component
    # input: component_name: which component is being processed, select_option: which model is selected, build_feat: feature for training this model, build_label: label for this component, which is to be transformed and then used to train
    # return: a trained machine learning part for this component
    def build_model_for_one_component(self, component_name, select_option, build_feat, build_label):
        build_transformed_label = self.encode_arch_knowledge(component_name, build_feat, build_label)    
        if select_option == 0:
            model = xgb.XGBRegressor()
        else:
            model = GradientBoostingRegressor()
        trained_model = self.train_model(model,build_feat,build_transformed_label)
        return trained_model
    
    # just for testing
    def function_testing_build_model_for_one_component(self):
        feature = np.array([[2,2,0.01,0.02,0.03,0.04],[4,4,0.05,0.06,0.07,0.08]])
        label = np.array([100,1000])
        trained_model = self.build_model_for_one_component("BP",0,feature,label)
        print(feature)
        print(label)
        pred = trained_model.predict(feature)
        print(pred)
        
        feature1 = np.array([[2,2,0.01,0.02,0.03,0.04],[4,4,0.05,0.06,0.07,0.08]])
        label1 = np.array([200,2000])
        trained_model1 = self.build_model_for_one_component("FU_Pool",0,feature1,label1)
        print(feature1)
        print(label1)
        pred1 = trained_model1.predict(feature1)
        print(pred1)
        
        return
    
    # build the machine learning part for PANDA
    # input: train_feature: the feature of training set, train_label: the label of training set
    # return: a list of model, each model corresponds to one component
    def build_power_model(self,train_feature,train_label):
        model_list = []
        iter = 0
        for component in self.event_feature_of_components.keys():
            # get model option
            model_select_option = self.component_model_selection[iter]
            
            # get respective feature and label
            start_event = self.event_feature_of_components[component][0]
            end_event = self.event_feature_of_components[component][1]
            feature_index = self.params_feature_of_components[component] + [item for item in range(start_event,end_event)]
            component_feature = train_feature[:,feature_index]
            label_index = iter + 1
            component_label = train_label[:,label_index]
            
            # build model
            ml_model_this_component = self.build_model_for_one_component(component,model_select_option,component_feature,component_label)
            model_list.append(ml_model_this_component)
            iter = iter + 1
        return model_list
    
    # just for testing
    def function_testing_build_power_model(self):
        training_set_index = [10,100]
        train_feature = self.panda_feature[training_set_index]
        train_label = self.panda_label[training_set_index]
        print(train_feature)
        print(train_label)
        cp_feat = copy.deepcopy(train_feature)
        cp_label = copy.deepcopy(train_label)
        model_list = self.build_power_model(train_feature,train_label)
        print(train_feature)
        print(train_label)
        error = 0
        for i in range(2):
            for j in range(cp_feat.shape[1]):
                if abs(cp_feat[i][j]-train_feature[i][j])>0.00001:
                    error = 1
        for i in range(2):
            for j in range(cp_label.shape[1]):
                if abs(cp_label[i][j]-train_label[i][j])>0.00001:
                    error = 1
        print(error)
    
    # compute the final power estimation
    # input: component_name: which component's power is being computed, feature: feature is used to compute resource function, pred: the result of machine learning part
    def decode_arch_knowledge(self,component_name,feature,pred):
        if component_name=="BP" or component_name=="ICache" or component_name=="DCache" or component_name=="RNU" or component_name=="ROB" or component_name=="IFU" or component_name=="LSU":
            scale_factor = np.ones(pred.shape)
            for i in range(len(self.encode_table[component_name])):
                decode_index = self.encode_table[component_name][i]
                acc_feature = feature[:,decode_index]
                scale_factor = scale_factor * acc_feature
            decode_pred = pred * scale_factor
        elif component_name=="Regfile":
            scale_factor = np.zeros(pred.shape)
            for i in range(len(self.encode_table[component_name])):
                decode_index = self.encode_table[component_name][i]
                acc_feature = feature[:,decode_index]
                scale_factor = scale_factor + acc_feature
            decode_pred = pred * scale_factor
        elif component_name=="ISU":
            decode_index = self.encode_table[component_name][0]
            decodewidth = feature[:,decode_index]
            reserve_station = np.array([self.compute_reserve_station_entries(decodewidth[i]) for i in range(decodewidth.shape[0])])
            decode_pred = pred * reserve_station
        elif component_name=="OtherLogic":
            decode_index = self.encode_table[component_name][0]
            decode_pred = pred * (feature[:,decode_index] + self.logic_bias)
        elif component_name=="Dtlb":
            decode_index = self.encode_table[component_name][0]
            decode_pred = pred * (feature[:,decode_index] + self.dtlb_bias)
        else:
            decode_pred = pred * 1.0
        return decode_pred
    
    
    # compute power for one component
    # input: component_name: which component is being processed, model: the machine learning model of this component, test_feat: the related feature of this component
    # return: a power prediction for this component
    def test_for_one_component(self, component_name, model, test_feat):
        pred_part = model.predict(test_feat)
        power_pred = self.decode_arch_knowledge(component_name,test_feat,pred_part)
        return power_pred
    

    # test the PANDA
    # input: test_feature: the feature of testing set, model_list: the machine learning part of PANDA
    # return: total power prediction
    def test_power_model(self,test_feature,model_list):
        iter = 0
        power_value = np.zeros(test_feature.shape[0])
        for component in self.event_feature_of_components.keys():
            # get model
            model_component = model_list[iter]
            
            # get respective feature and label
            start_event = self.event_feature_of_components[component][0]
            end_event = self.event_feature_of_components[component][1]
            feature_index = self.params_feature_of_components[component] + [item for item in range(start_event,end_event)]
            component_feature = test_feature[:,feature_index]
            
            # compute and accumulate power
            power_component = self.test_for_one_component(component,model_component,component_feature)
            power_value = power_value + power_component
            
            iter = iter + 1
            
        return power_value
    
    # just for testing
    def function_testing_power_model(self):
        training_set_index = [item for item in range(self.panda_feature.shape[0])]
        train_feature = self.panda_feature[training_set_index]
        train_label = self.panda_label[training_set_index]
        total_power_label = train_label[:,0]
        print(train_label[:,0])
        #print(train_feature)
        #print(train_label)
        cp_feat = copy.deepcopy(train_feature)
        cp_label = copy.deepcopy(train_label)
        model_list = self.build_power_model(train_feature,train_label)
        #print(train_feature)
        #print(train_label)
        error = 0
        for i in range(2):
            for j in range(cp_feat.shape[1]):
                if abs(cp_feat[i][j]-train_feature[i][j])>0.00001:
                    error = 1
        for i in range(2):
            for j in range(cp_label.shape[1]):
                if abs(cp_label[i][j]-train_label[i][j])>0.00001:
                    error = 1
        #print(error)
        power_value_pred = self.test_power_model(train_feature,model_list)
        print(power_value_pred)
        r_report = np.corrcoef(total_power_label,power_value_pred)[1][0]
        mape_report = mean_absolute_percentage_error(total_power_label,power_value_pred)
        #print("Unknown_{}_config".format(unknown))
        print("R = {}".format(r_report))
        print("MAPE = {}%".format(mape_report * 100))  
        
        
    def unknown_n_config(self,unknown):
        
        cp_feature = copy.deepcopy(self.panda_feature)
        cp_label = copy.deepcopy(self.panda_label)
        
        fold = 15
        test_size = 8 * unknown
        pred_list = []
        label_list = []
        pred_acc_vector = np.zeros((15*8))
        
        for i in range(fold):
            start_point = 8*i
            end_point = start_point + test_size
            if end_point<=self.panda_feature.shape[0]:
                testing_set = [item for item in range(start_point,end_point)]
                training_set = [item for item in range(0,start_point)] + [item for item in range(end_point,self.panda_feature.shape[0])]
            else:
                end_point = end_point - self.panda_feature.shape[0]
                testing_set = [item for item in range(start_point,self.panda_feature.shape[0])] + [item for item in range(0,end_point)]
                training_set = [item for item in range(end_point,start_point)]
                
            training_feature = (self.panda_feature[training_set]).copy()
            training_label = (self.panda_label[training_set]).copy()
            model_list = self.build_power_model(training_feature,training_label)
            
            testing_feature = (self.panda_feature[testing_set]).copy()
            testing_label = (self.panda_label[testing_set]).copy()
            power_prediction = self.test_power_model(testing_feature,model_list)
            
            pred_list = pred_list + power_prediction.tolist()
            label_list = label_list + testing_label[:,0].tolist()
            #print(pred)
            #print(label)
            pred_acc_vector[testing_set] = pred_acc_vector[testing_set] + power_prediction
        
        #print(len(label_list))
        r_report = np.corrcoef(label_list,pred_list)[1][0]
        mape_report = mean_absolute_percentage_error(label_list,pred_list)
        #print("Unknown_{}_config".format(unknown))
        #print("R = {}".format(r_report))
        #print("MAPE = {}%".format(mape_report * 100))  
        
        pred_acc_vector = pred_acc_vector / unknown
        label = self.panda_label[:,0]
        
        r_report = np.corrcoef(label,pred_acc_vector)[1][0]
        mape_report = mean_absolute_percentage_error(label,pred_acc_vector)
        #print("Unknown_{}_config".format(unknown))
        #print("R = {}".format(r_report))
        print("MAPE = {}%".format(mape_report * 100))  
        
        error = 0
        for i in range(120):
            for j in range(cp_feature.shape[1]):
                if abs(cp_feature[i][j]-self.panda_feature[i][j])>0.00001:
                    error = 1
        for i in range(120):
            for j in range(cp_label.shape[1]):
                if abs(cp_label[i][j]-self.panda_label[i][j])>0.00001:
                    error = 1
        #print(error)
        
        return r_report, mape_report
    
    
    '''# input: 
    def build_model_for_component(self, select_option, build_feat, build_label):
        num_of_component = len(select_option)
        model_list = []
        for i in range(num_of_component):
            
        return
    '''

panda = PANDA()
for i in range(1,15):
    r,mape = panda.unknown_n_config(i)
    
#panda.unknown_n_config(1)
#panda.unknown_n_config(10)
#panda.unknown_n_config(14)
#curve_mape = []
#curve_r = []
#for i in range(1,15):
#    r,mape = panda.unknown_n_config(i)
#    curve_mape.append(mape)
#    curve_r.append(r)
#print(curve_mape)
#print(curve_r)
#np.save("our_curve_mape.npy",np.array(curve_mape))
#np.save("our_curve_r.npy",np.array(curve_r))