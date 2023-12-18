import os
import re
import numpy as np
import csv
#import pandas as pd
class microarch_data_process:
    
    def __init__(self):
        self.boom_core_config_table = [
            #0          1           2                3        4               5              6            7           8                     9             10               11                    12         13
            #FetchWidth DecodeWidth FetchBufferEntry RobEntry IntPhysRegister FpPhysRegister LDQ/STQEntry BranchCount MemIssue/FpIssueWidth IntIssueWidth DCache/ICacheWay DCacheTLBEntry        DCacheMSHR ICacheFetchBytes
            [4,         1,          5,               16,      36,             36,            4,           6,          1,                    1,            2,               8,                    2,         2],
            [4,         1,          8,               32,      53,             48,            8,           8,          1,                    1,            4,               8,                    2,         2],
            [4,         1,          16,              48,      68,             56,            16,          10,         1,                    2,            8,               16,                   4,         2],

            [4,         2,          8,               64,      64,             56,            12,          10,         1,                    1,            4,               8,                    2,         2],
            [4,         2,          16,              64,      80,             64,            16,          12,         1,                    2,            4,               8,                    2,         2],
            [8,         2,          24,              80,      88,             72,            20,          14,         1,                    2,            8,               16,                   4,         4],

            [8,         3,          18,              81,      88,             88,            16,          14,         1,                    2,            8,               16,                   4,         4],
            [8,         3,          24,              96,      110,            96,            24,          16,         1,                    3,            8,               16,                   4,         4],
            [8,         3,          30,              114,     112,            112,           32,          16,         2,                    3,            8,               32,                   4,         4],

            [8,         4,          24,              112,     108,            108,           24,          18,         1,                    4,            8,               32,                   4,         4],
            [8,         4,          32,              128,     128,            128,           32,          20,         2,                    4,            8,               32,                   4,         4],
            [8,         4,          40,              136,     136,            136,           36,          20,         2,                    4,            8,               32,                   8,         4],

            [8,         5,          30,              125,     108,            108,           24,          18,         2,                    5,            8,               32,                   8,         4],
            [8,         5,          35,              130,     128,            128,           32,          20,         2,                    5,            8,               32,                   8,         4],
            [8,         5,          40,              140,     140,            140,           36,          20,         2,                    5,            8,               32,                   8,         4]
        ]
        self.params_name=[
            'FetchWidth',
            'DecodeWidth',
            'FetchBufferEntry',
            'RobEntry',
            'IntPhysRegister',
            'FpPhysRegister',
            'LDQ/STQEntry',
            'BranchCount',
            'MemIssue/FpIssueWidth',
            'IntIssueWidth',
            'DCache/ICacheWay',
            'DCacheTLBEntry',
            'DCacheMSHR',
            'ICacheFetchBytes'
        ]
        self.benchmark=[
            "setup",
            "dhrystone",
            "median",
            "multiply",
            "qsort",
            "rsort",
            "spmv",
            "towers",
            "vvadd"
        ]
        self.mcpat_pattern={
            "core_area":47,
            "core_subthreshold_leakage":49,
            "core_gate_leakage":50,
            "core_dynamic":51,
            "ifu_dynamic":58,
            "ic_dynamic":65,
            "btb_dynamic":72,
            "bp_dynamic":79,
            "gp_dynamic":86,
            "l1_lp_dynamic":94,
            "l2_lp_dynamic":101,
            "chooser_dynamic":108,
            "ras_dynamic":115,
            "inst_buffer_dynamic":122,
            "inst_docoder_dynamic":129,
            "rnu_dynamic":136,
            "intRAT_dynamic":143,
            "fpRAT_dynamic":150,
            "Free_list_dynamic":157,
            "FP_list_dynamic":164,
            "LSU_dynamic":171,
            "dc_dynamic":178,
            "loadq_dynamic":185,
            "storeq_dynamic":192,
            "MMU_dynamic":198,
            "itlb_dynamic":205,
            "dtlb_dynamic":212,
            "EXU_dynamic":218,
            "RF_dynamic":225,
            "IntRF_dynamic":232,
            "FpRF_dynamic":239,
            "Inst_sche_dynamic":246,
            "Inst_wind_dynamic":253,
            "fp_wind_dynamic":260,
            "ROB_dynamic":267,
            "IntALU_dynamic":274,
            "FPU_dynamic":281,
            "CompU_dynamic":288,
            "Result_Bus_dynamic":295
        }
        self.showfeature=[
            "system.cpu.numCycles",
            "core_area",
            "core_leakage",
            "core_dynamic"
        ]
        self.components=[
            "DCache",
            "ICache",
            "BTB",
            "BP",
            "RNU",
            "Itlb",
            "Dtlb",
            "Regfile",
            "ROB"
        ]
        self.location=[
            165,
            52,
            59,
            66,
            123,
            192,
            199,
            212,
            254 
        ]
        self.event_feature_of_components={
            "DCache":["system.cpu.dcache.ReadReq.accesses::total","system.cpu.dcache.WriteReq.accesses::total","system.cpu.dcache.ReadReq.misses::total","system.cpu.dcache.WriteReq.misses::total","system.cpu.dcache.overallAccesses::total","system.cpu.dcache.overallMisses::total","system.cpu.dcache.overallMshrHits::total","system.cpu.dcache.overallMshrMisses::total","system.cpu.dcache.tags.totalRefs","system.cpu.dcache.tags.tagAccesses"],
            "ICache":["system.cpu.icache.overallAccesses::total","system.cpu.icache.overallMisses::total","system.cpu.icache.ReadReq.mshrHits::total","system.cpu.icache.ReadReq.mshrMisses::total","system.cpu.icache.tags.totalRefs","system.cpu.icache.tags.tagAccesses"],
            "BTB":["system.cpu.branchPred.BTBLookups","system.cpu.branchPred.condPredicted","system.cpu.branchPred.condIncorrect","system.cpu.commit.branches"],
            "BP":["system.cpu.branchPred.BTBLookups","system.cpu.branchPred.condPredicted","system.cpu.branchPred.condIncorrect","system.cpu.commit.branches"],
            "RNU":["system.cpu.rename.intLookups","system.cpu.rename.renamedOperands","system.cpu.rename.fpLookups","system.cpu.rename.renamedInsts","system.cpu.rename.runCycles","system.cpu.rename.blockCycles","system.cpu.rename.committedMaps"],
            "Itlb":["system.cpu.mmu.itb.accesses","system.cpu.mmu.itb.misses"],
            "Dtlb":["system.cpu.mmu.dtb.accesses","system.cpu.mmu.dtb.misses"],
            "Regfile":["system.cpu.intRegfileReads","system.cpu.fpRegfileReads","system.cpu.intRegfileWrites","system.cpu.fpRegfileWrites","system.cpu.commit.functionCalls"],
            "ROB":["system.cpu.rob.reads","system.cpu.rob.writes"],
            "IFU":["system.cpu.fetch.insts","system.cpu.fetch.branches","system.cpu.fetch.cycles","system.cpu.numRefs","system.cpu.numStoreInsts","system.cpu.numInsts","system.cpu.decode.runCycles","system.cpu.decode.blockedCycles","system.cpu.decode.decodedInsts","system.cpu.numBranches","system.cpu.statIssuedInstType_0::total","system.cpu.intInstQueueReads","system.cpu.intInstQueueWrites","system.cpu.intInstQueueWakeupAccesses","system.cpu.fpInstQueueReads","system.cpu.fpInstQueueWrites","system.cpu.fpInstQueueWakeupAccesses"],
            "LSU":["system.cpu.commit.committedInstType_0::MemRead","stats.system.cpu.commit.committedInstType_0::InstPrefetch","system.cpu.commit.committedInstType_0::MemWrite"],
            "FU_Pool":["system.cpu.intAluAccesses","system.cpu.fpAluAccesses"],
            "ISU":["system.cpu.statIssuedInstType_0::total","system.cpu.statIssuedInstType_0::MemRead","system.cpu.statIssuedInstType_0::MemWrite","system.cpu.statIssuedInstType_0::FloatMemRead","system.cpu.statIssuedInstType_0::FloatMemWrite","system.cpu.statIssuedInstType_0::IntAlu","system.cpu.statIssuedInstType_0::IntMult","system.cpu.statIssuedInstType_0::IntDiv"]
        }
        
    def get_value(self,location,prelines):
        area_value = float((prelines[location].split())[-2])
        Subthreshold_Leakage_value = float((prelines[location+2].split())[-2])
        if location==186 or location==206:
            Gate_Leakage_value = 0
            Dynamic_value = float((prelines[location+3].split())[-2])
        else:
            Gate_Leakage_value = float((prelines[location+3].split())[-2])
            Dynamic_value = float((prelines[location+4].split())[-2])
        return area_value,Subthreshold_Leakage_value,Gate_Leakage_value,Dynamic_value
        
    def extract_data(self):
        
        #name_table
        gem5_name = open("gem5_stats_name").read()
        gem5_name_table = gem5_name.splitlines()
        mcpat_name_table = [key for key in self.mcpat_pattern.keys()]
        mcpat_name_table.remove("core_subthreshold_leakage")
        mcpat_name_table[1] = "core_leakage"
        name_list = mcpat_name_table+self.params_name+gem5_name_table
        with open('feature_name','w') as f:
            for item in name_list:
                f.writelines(item+'\n')
        
        #values
        microarch_data = []
        feature_partition = []
        feature_partition_name = []
        for i in range(len(self.boom_core_config_table)):
            data_of_this_config = []
            for j in range(len(self.benchmark)):
                data_of_this_workload = []
                workload = self.benchmark[j]
                gem5_stats_file = "boom{}/{}/1GHz/stats.txt".format(i,workload)
                mcpat_file = "boom{}/{}/1GHz/mcpat_result".format(i,workload)
                
                #McPAT
                mcpat_result = open(mcpat_file).read()
                prelines = mcpat_result.splitlines()
                leakage = 0
                for key in self.mcpat_pattern.keys():
                    the_line = prelines[self.mcpat_pattern[key]-1-8]
                    value_pattern = '\s0\s|\s0\.\d+\s|\s[1-9]\d*\.\d+\s|\s0\.\d*e-\d*\s|\s[1-9]\d*\.\d*e-\d*\s'
                    values = float((re.findall(value_pattern,the_line))[0])
                    if key == "core_subthreshold_leakage":
                        leakage = leakage + values
                        continue
                    if key == "core_gate_leakage":
                        leakage = leakage + values
                        data_of_this_workload.append(leakage)
                        continue
                    if key == "core_area":
                        values = values*1000000
                    data_of_this_workload.append(values)
                    
                if i==0 and j==0:
                    feature_partition.append(len(data_of_this_workload))
                    feature_partition_name.append("McPAT_Total")
                
                area_list = []
                power_list = []
                for item in range(len(self.components)):
                    area_value = float((prelines[self.location[item]].split())[-2])
                    Subthreshold_Leakage_value = float((prelines[self.location[item]+2].split())[-2])
                    Gate_Leakage_value = float((prelines[self.location[item]+3].split())[-2])
                    Dynamic_value = float((prelines[self.location[item]+4].split())[-2])
                    area_list.append(area_value*1000000)
                    power_list.append(Dynamic_value)
                    #power_list.append(Subthreshold_Leakage_value+Gate_Leakage_value+Dynamic_value)
                    
                #IFU
                area_0,subleakage_0,gateleakage_0,dynamic_0=self.get_value(45,prelines)
                area_1,subleakage_1,gateleakage_1,dynamic_1=self.get_value(240,prelines)
                area_2,subleakage_2,gateleakage_2,dynamic_2=self.get_value(247,prelines)
                sub_area_0,sub_subleakage_0,sub_gateleakage_0,sub_dynamic_0=self.get_value(52,prelines)
                sub_area_1,sub_subleakage_1,sub_gateleakage_1,sub_dynamic_1=self.get_value(59,prelines)
                sub_area_2,sub_subleakage_2,sub_gateleakage_2,sub_dynamic_2=self.get_value(66,prelines)
                area_list.append((area_0+area_1+area_2-sub_area_0-sub_area_1-sub_area_2)*1000000)
                power_list.append(dynamic_0+dynamic_1+dynamic_2-sub_dynamic_0-sub_dynamic_1-sub_dynamic_2)
                
                #LSU
                area_0,subleakage_0,gateleakage_0,dynamic_0=self.get_value(158,prelines)
                area_1,subleakage_1,gateleakage_1,dynamic_1=self.get_value(186,prelines)
                sub_area_0,sub_subleakage_0,sub_gateleakage_0,sub_dynamic_0=self.get_value(165,prelines)
                sub_area_1,sub_subleakage_1,sub_gateleakage_1,sub_dynamic_1=self.get_value(192,prelines)
                sub_area_2,sub_subleakage_2,sub_gateleakage_2,sub_dynamic_2=self.get_value(199,prelines)
                area_list.append((area_0+area_1-sub_area_0-sub_area_1-sub_area_2)*1000000)
                power_list.append(dynamic_0+dynamic_1-sub_dynamic_0-sub_dynamic_1-sub_dynamic_2)
                
                #FU_Pool
                area_0,subleakage_0,gateleakage_0,dynamic_0=self.get_value(261,prelines)
                area_1,subleakage_1,gateleakage_1,dynamic_1=self.get_value(268,prelines)
                area_2,subleakage_2,gateleakage_2,dynamic_2=self.get_value(275,prelines)
                area_list.append((area_0+area_1+area_2)*1000000)
                power_list.append(dynamic_0+dynamic_1+dynamic_2)
                
                #params
                for param_configs in range(len(self.boom_core_config_table[i])):
                    data_of_this_workload.append(self.boom_core_config_table[i][param_configs])
                    
                if i==0 and j==0:
                    feature_partition.append(len(data_of_this_workload))
                    feature_partition_name.append("Params")
                
                #gem5                        
                gem5_result = open(gem5_stats_file).read()
                gem5_name = open("gem5_stats_name").read()
                name_table = gem5_name.splitlines()
                start_point = len(data_of_this_workload)
                numcycle = 0
                for item in name_table:
                    gem5_pattern = item+'.*\n'
                    text = re.findall(gem5_pattern,gem5_result)
                    if len(text)==0:
                        data_of_this_workload.append(0)
                    else:
                        value_pattern = '\s\d+\s|\s0\.\d+\s|\s[1-9]\d*\.\d+\s|\s0\.\d*e-\d*\s|\s[1-9]\d*\.\d*e-\d*\s'
                        value = re.findall(value_pattern,text[0])[0]
                        data_of_this_workload.append(float(value))
                        if item == "system.cpu.numCycles":
                            numcycle = float(value)
                            
                if i==0 and j==0:
                    feature_partition.append(len(data_of_this_workload))
                    feature_partition_name.append("gem5_general")
                            
                #per-components
                for keys in self.event_feature_of_components.keys():
                    event_table = self.event_feature_of_components[keys]
                    if keys!="ISU":
                        for item in event_table:
                            gem5_pattern = item+'.*\n'
                            text = re.findall(gem5_pattern,gem5_result)
                            if len(text)==0:
                                data_of_this_workload.append(0)
                            else:
                                value_pattern = '\s\d+\s|\s0\.\d+\s|\s[1-9]\d*\.\d+\s|\s0\.\d*e-\d*\s|\s[1-9]\d*\.\d*e-\d*\s'
                                value = re.findall(value_pattern,text[0])[0]
                                data_of_this_workload.append(float(value))
                    else:
                        isu_table = []
                        for item in event_table:
                            gem5_pattern = item+'.*\n'
                            text = re.findall(gem5_pattern,gem5_result)
                            if len(text)==0:
                                isu_table.append(0)
                            else:
                                value_pattern = '\s\d+\s|\s0\.\d+\s|\s[1-9]\d*\.\d+\s|\s0\.\d*e-\d*\s|\s[1-9]\d*\.\d*e-\d*\s'
                                value = re.findall(value_pattern,text[0])[0]
                                isu_table.append(float(value))
                        mem_issue = isu_table[1]+isu_table[2]+isu_table[3]+isu_table[4]
                        int_issue = isu_table[5]+isu_table[6]+isu_table[7]
                        float_issue = isu_table[0]-mem_issue-int_issue
                        data_of_this_workload = data_of_this_workload + [mem_issue,int_issue,float_issue]
                    if i==0 and j==0:
                        feature_partition.append(len(data_of_this_workload))
                        feature_partition_name.append(keys)

                #print(start_point)
                #print(len(data_of_this_workload))
                for item in range(start_point+3,len(data_of_this_workload)):
                    data_of_this_workload[item] = data_of_this_workload[item]/numcycle
            
                data_of_this_workload = data_of_this_workload + area_list + power_list
                if i==0 and j==0:
                    feature_partition.append(len(data_of_this_workload))
                    feature_partition_name.append("McPAT_components")
                data_of_this_config.append(data_of_this_workload)
            microarch_data.append(data_of_this_config)
        data_array = np.array(microarch_data)

        feature_partition = [str(item)+'\n' for item in feature_partition]
        feature_partition_name = [item+'\n' for item in feature_partition_name]
        with open('feature_partition','w') as f:
            f.writelines(feature_partition)
        with open('feature_partition_name','w') as f:
            f.writelines(feature_partition_name)    
        np.save('../example_data/feature_set.npy', data_array)
        
                               
processing = microarch_data_process()
processing.extract_data()