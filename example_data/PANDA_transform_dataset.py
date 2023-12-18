import numpy as np

def transform_dataset():
    microarch_dataset = np.load('feature_set_5_9.npy')
    eda_dataset = np.load('label_set_5_7.npy')
    print(microarch_dataset.shape)
    print(eda_dataset.shape)
    feature_index = [item for item in range(38,155)] + [item for item in range(159,206)]
    power_related_microarch_dataset = microarch_dataset[:,1:,feature_index]
    label_index = [4] + [item for item in range(18,30)]
    power_related_eda_dataset = eda_dataset[:,:,label_index]
    power_related_microarch_dataset = power_related_microarch_dataset.reshape((power_related_microarch_dataset.shape[0]*power_related_microarch_dataset.shape[1],power_related_microarch_dataset.shape[2]))
    power_related_eda_dataset = power_related_eda_dataset.reshape((power_related_eda_dataset.shape[0]*power_related_eda_dataset.shape[1],power_related_eda_dataset.shape[2]))
    print(power_related_microarch_dataset.shape)
    print(power_related_eda_dataset.shape)
    #print(power_related_microarch_dataset[0][0])
    print(power_related_eda_dataset[0])
    print(power_related_eda_dataset[0][0])
    print(power_related_eda_dataset[0][1:].sum())
    other_power = power_related_eda_dataset[:,1:].sum(axis=1)
    other_power = other_power.reshape((other_power.shape[0],1))
    total_power = power_related_eda_dataset[:,0]
    total_power = total_power.reshape((total_power.shape[0],1))
    other_power = total_power - other_power
    power_related_eda_dataset = np.concatenate((total_power,other_power,power_related_eda_dataset[:,1:]),axis=1)
    print(power_related_eda_dataset.shape)
    print(power_related_eda_dataset[0])
    #print(other_power.shape)
    np.save('panda_feature.npy',power_related_microarch_dataset)
    np.save('panda_label.npy',power_related_eda_dataset)
    return

transform_dataset()