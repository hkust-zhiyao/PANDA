import os
import numpy as np
import re
class eda_process:
    def __init__(self):
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
        self.num_of_config = 15
        self.components_chipyard_area=[
            "dcache",#0
            "frontend/icache",#1
            "frontend/bpd/banked_predictors_0/components_2",#2
            "frontend/bpd",#3
            "core/rename_stage",#4
            "core/fp_rename_stage",#5
            "frontend/tlb",#6
            "lsu/dtlb",#7
            "core/iregfile",#8
            "core/fp_pipeline/fregfile",#9
            "core/rob",#10
            "frontend",#11
            "lsu",#12
            "core/csr_exe_unit",#13
            "core/fp_pipeline/fpiu_unit",#14
            
            "core/int_issue_unit",#15
            "core/fp_pipeline/fp_issue_unit",#16
            "core/mem_issue_unit"#17
        ]
        self.components_chipyard_power=[
            "dcache",#0
            "icache",#1
            "components_2",#2
            "bpd",#3
            "rename_stage",#4
            "fp_rename_stage",#5
            "tlb",#6
            "dtlb",#7
            "iregfile",#8
            "fregfile",#9
            "rob",#10
            "frontend",#11
            "lsu",#12
            "csr_exe_unit",#13
            "fpiu_unit",#14
            
            "int_issue_unit",#15
            "fp_issue_unit",#16
            "mem_issue_unit"#17
        ]

    def process_eda_data(self):
        eda_dataset = []
        for i in range(self.num_of_config):
            config_dataset = []
            for j in range(1, len(self.benchmark)):
                workload = self.benchmark[j]
                
                
                #num_of_cycle
                simulation_result = "path_to_chipyard/vlsi/output/chipyard.TestHarness.BoomConfig{}/{}.fsdb_log".format(i, workload)
                print(simulation_result)
                mcycle_pattern = 'mcycle = .*\n'
                simulation_text = open(simulation_result).read()
                text = re.findall(mcycle_pattern,simulation_text)
                value_pattern = '\d+'
                num_of_cycle = int(re.findall(value_pattern,text[0])[0])
                
                
                #slack
                qor_result = "path_to_design_compiler_result/boom{}tsmc/report/qor.rpt".format(i)
                slack_pattern = 'Critical Path Slack:.*\n'
                slack_text = open(qor_result).read()
                text = re.findall(slack_pattern,slack_text)
                slack = float(text[0].split()[3])
                
                
                #area
                area_result = "path_to_design_compiler_result/boom{}tsmc/report/area.rpt".format(i)
                area_pattern = 'Total cell area:.*\n'
                area_text = open(area_result).read()
                text = re.findall(area_pattern,area_text)
                area_str = text[0].split()[3]
                area = float(area_str)
                #per-components
                area_list = []
                for area_item in range(len(self.components_chipyard_area)):
                    area_pattern = "\n{}.*\n".format(self.components_chipyard_area[area_item])
                    text = re.findall(area_pattern,area_text)
                    area_sub = float(text[0].split()[1])
                    area_list.append(area_sub)

                            
                #power
                power_result = "path_to_ptpx_result/power_tsmc_boom{}tsmc_{}.rpt".format(i,workload)
                power_text = open(power_result).read()
                splited_text = power_text.splitlines()
                text = splited_text[13]
                power = text.split()
                leakage = float(power[3])
                dynamic = float(power[1]) + float(power[2])
                total = float(power[4])
                #per-components
                power_list = []
                for power_item in range(len(self.components_chipyard_power)):
                    power_pattern = " {} .*\n".format(self.components_chipyard_power[power_item])
                    text = re.findall(power_pattern,power_text)
                    splited_text = text[0].split()
                    if len(splited_text) == 2:
                        power_pattern = " {} .*\n.*\n".format(self.components_chipyard_power[power_item])
                        text = re.findall(power_pattern,power_text)
                        splited_text = text[0].split()
                    power_value = float(splited_text[-2])
                    power_list.append(power_value)
                    
                    
                    
                #FU_Pool
                FU_area = 0
                FU_area_pattern = "\n.*ExeUnit.*\n"
                text = re.findall(FU_area_pattern,area_text)
                for itr in range(len(text)):
                    FU_area = FU_area + float(text[itr].split()[1])
                FU_power = 0
                FU_power_pattern = "\n.*ExeUnit.*\n"
                text = re.findall(FU_power_pattern,power_text)
                for itr in range(len(text)):
                    FU_power = FU_power + float(text[itr].split()[-2])



                area_values = [area_list[0],area_list[1],area_list[3],area_list[4]+area_list[5],area_list[6],area_list[7],area_list[8]+area_list[9],area_list[10],area_list[11]-area_list[1]-area_list[3]-area_list[6],area_list[12]-area_list[7],FU_area,area_list[15]+area_list[16]+area_list[17]]
                power_values = [power_list[0],power_list[1],power_list[3],power_list[4]+power_list[5],power_list[6],power_list[7],power_list[8]+power_list[9],power_list[10],power_list[11]-power_list[1]-power_list[3]-power_list[6],power_list[12]-power_list[7],FU_power,power_list[15]+power_list[16]+power_list[17]]

                
                single_data = [num_of_cycle,slack,area,leakage,dynamic,total]
                single_data = single_data + area_values + power_values
                config_dataset.append(single_data)
            eda_dataset.append(config_dataset)
        data_array = np.array(eda_dataset)
        #print(data_array.shape)
        #print(data_array[0][0])
        np.save('../example_data/label_set.npy', data_array)
                
                
processing = eda_process()
processing.process_eda_data()
