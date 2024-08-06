import win32com.client as win32
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import sys
## 계산을 빨리하기위한 방법론 : 가장 루프가 긴 이터레이션에 대해서 remove_efficiency가 0.99이상이면
## 빨리 break한다
## 그렇게 해도 되는 이유는 반복을 하면서 특성이 증가하는 것이기에, NH3 제거효율은 어차피 증가할것이고
## 0.99이상이 될때 굳이 더 계산할 필요가 없긴하다.
## 계산 시간이 1/3 로 줄어들었다.
import logging
def get_next_log_filename(base_filename):
    if not os.path.exists(base_filename + ".log"):
        return base_filename+ ".log"
    i = 1
    while os.path.exists(f"{base_filename}{i}.log"):
        i+=1
    return f"{base_filename}{i}.log"
log_filename = get_next_log_filename("./logs/output")
print(log_filename)
logging.basicConfig(filename = "./%s"%log_filename ,level = logging.INFO , format = "%(asctime)s - %(message)s", encoding = 'utf-8')
def main():
    aspen = win32.dynamic.Dispatch("Apwn.Document")
    aspen.initFromArchive2(os.path.abspath("scrubber.bkp"))

    logging.info("Aspen is Launching...")

    ################### 필요한 노드 찾아두기 ####################

    ## 가스 입력
    gas_temp = aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\TEMP\\MIXED").value
    gas_pressure = aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\PRES").value
    gas_flowrate = aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\TOTFLOW").value

    gas_total_frac = aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\TOTAL").value
    gas_ammonia_frac = aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\FLOW\\MIXED\\AMMONIA").value
    gas_hydrogen_frac = aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\FLOW\\MIXED\\HYDROGEN").value
    gas_nitrogen_frac = aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\FLOW\\MIXED\\NITROGEN").value

    ## 물 입력
    water_flowrate = aspen.Tree.FindNode("\\Data\\Streams\\WATERIN\\Input\\TOTFLOW").value
    water_temp = aspen.Tree.FindNode("\\Data\\Streams\\WATERIN\\Input\\TEMP").value
    water_pressure = aspen.Tree.FindNode("\\Data\\Streams\\WATERIN\\Input\\PRES").value
    print(gas_pressure, water_pressure)
    ## 트레이 관련
    tray_numbers = aspen.Tree.FindNode("\\Data\\Blocks\\SCRUBBER").Elements("Input").Elements("NSTAGE").value
    gas_tray = aspen.Tree.FindNode("\\Data\\Blocks\\SCRUBBER\\Input\\FEED_STAGE\\GASIN").value
    column_pressure = aspen.Tree.FindNode("\\Data\\Blocks\\SCRUBBER").Elements("Input").Elements("PRES1").value

    ## 아웃풋은 돌려야 나온다
    aspen.Engine.Run2()

    ## 액체 아웃풋
    liqout_flowrate_water = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\MOLEFLOW\\MIXED\\WATER").value
    liqout_flowrate_HYDROGEN = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\MOLEFLOW\\MIXED\\HYDROGEN").value
    liqout_flowrate_NITROGEN = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\MOLEFLOW\\MIXED\\NITROGEN").value
    liqout_flowrate_AMMONIA = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\MOLEFLOW\\MIXED\\AMMONIA").value

    ######################필요 함수 정의################################
    def flowrate_to_composition(flowrate):
        ref_flowrate = 2080.57 ## kmol/hr
        predicted_reacter_conversion = (flowrate - ref_flowrate) / ref_flowrate
        nh3_flowrate = ref_flowrate * (1-predicted_reacter_conversion)
        h2_flowrate = ref_flowrate * predicted_reacter_conversion * 3/2    
        n2_flowrate = ref_flowrate * predicted_reacter_conversion * 1/2    
        total_flowrate = nh3_flowrate + h2_flowrate + n2_flowrate 
        error = total_flowrate - flowrate
        # if error <= (flowrate /100):
        #     logging.info("타당한 GASIN 조성계산")
        mol_frac = {"ammonia" : nh3_flowrate/total_flowrate , "hydrogen" : h2_flowrate/total_flowrate , "nitrogen" : n2_flowrate/total_flowrate}
        flowrate = {"ammonia" : nh3_flowrate , "hydrogen" : h2_flowrate , "nitrogen" : n2_flowrate, "total" : total_flowrate}
        return flowrate, mol_frac , predicted_reacter_conversion

    def conversion_to_flowrate_and_composition(conversion):
        ref_flowrate = 2080.57 ## kmol/hr
        flowrate = conversion * ref_flowrate + ref_flowrate
        nh3_flowrate = ref_flowrate * (1-conversion)
        h2_flowrate = ref_flowrate * conversion * 3/2    
        n2_flowrate = ref_flowrate * conversion * 1/2    
        total_flowrate = nh3_flowrate + h2_flowrate + n2_flowrate 
        mol_frac = {"ammonia" : nh3_flowrate/total_flowrate , "hydrogen" : h2_flowrate/total_flowrate , "nitrogen" : n2_flowrate/total_flowrate}
        flowrate = {"ammonia" : nh3_flowrate , "hydrogen" : h2_flowrate , "nitrogen" : n2_flowrate, "total" : total_flowrate}
        return flowrate, mol_frac , conversion

    # 데이터프레임 초기화
    data = []

    # 변수 범위 설정
    conversions = np.linspace(0.6, 0.9, 6)
    pressures = np.linspace(1.0, 4.0, 3)
    temperatures = np.linspace(25, 200, 8)
    column_pressures = np.linspace(1.0, 6.0, 15)
    water_flowrates = np.linspace(1000, 30000, 29)
    tray_numbers = np.arange(7, 15, 2)

    # 총 시뮬레이션 수 계산
    total_simulations = len(conversions) * len(pressures) * len(temperatures) * len(column_pressures) * len(water_flowrates) * len(tray_numbers)
    logging.info(f"Total simulations to run: {total_simulations}")

    # 초기화
    sim_count = 0
    time_log = []
    real_count = 0
    pseudo_count = 0
    # 계산 실행
    files = os.listdir("./datas")
    last_file_num =0
    last_file_name = ""
    for i in files:
        if ".csv" in i :
            nu = i.replace("scrubber_partial", "").replace(".csv", "")
            if int(nu) >last_file_num :
                last_file_num = int(nu)
                last_file_name = i
    logging.info(last_file_name)
    last_row = pd.read_csv("./datas/" + last_file_name).iloc[-1].to_dict()
    last_rows_conversions = np.round(last_row["conversion"], 2)
    last_rows_pressures = np.round(last_row["gas_pressure"], 2)
    last_rows_temperatures = np.round(last_row["gas_temp"], 2)
    last_rows_column_pressures = np.round(last_row["column_pressure"], 2)
    last_rows_water_flowrates = np.round(last_row["water_flowrate"], 2)
    last_rows_tray_numbers = last_row["tray_number"]
    print(last_rows_conversions,last_rows_pressures,last_rows_temperatures,last_rows_column_pressures,last_rows_water_flowrates,last_rows_tray_numbers)
    continue_flag = 0
    start_time = time.time()
    fff= 0
    first_remain_time = 1000000
    exit_flag  = 0
    for conversion in tqdm(conversions, desc="Conversions"):
        for gas_pressure in tqdm(pressures, desc="Gas Pressures", leave=False):
            for gas_temp in tqdm(temperatures, desc="Gas Temperatures", leave=False):
                for column_pressure in tqdm(column_pressures, desc="Column Pressures", leave=False):
                    for tray_number in tqdm(tray_numbers, desc="Tray Numbers", leave=False):
                        for water_flowrate in tqdm(water_flowrates, desc="Water Flowrates", leave=False):
                            # 각 조건에 맞게 입력 값 설정
                            sim_count += 1
                            pseudo_count += 1
                            if continue_flag == 1:
                                flowrate, mol_frac, _ = conversion_to_flowrate_and_composition(conversion)
                                aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\TEMP\\MIXED").Value = gas_temp
                                aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\PRES\\MIXED").Value = gas_pressure
                                aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\TOTFLOW\\MIXED").Value = flowrate['total']
                                aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\FLOW\\MIXED\\AMMONIA").Value = flowrate['ammonia']
                                aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\FLOW\\MIXED\\HYDROGEN").Value = flowrate['hydrogen']
                                aspen.Tree.FindNode("\\Data\\Streams\\GASIN\\Input\\FLOW\\MIXED\\NITROGEN").Value = flowrate['nitrogen']
                                
                                aspen.Tree.FindNode("\\Data\\Streams\\WATERIN\\Input\\TOTFLOW\\MIXED").Value = water_flowrate
                                aspen.Tree.FindNode("\\Data\\Blocks\\SCRUBBER\\Input\\NSTAGE").Value = tray_number
                                aspen.Tree.FindNode("\\Data\\Blocks\\SCRUBBER\\Input\\PRES1").Value = column_pressure
                                aspen.Tree.FindNode("\\Data\\Blocks\\SCRUBBER\\Input\\FEED_STAGE\\GASIN").Value = tray_number  # gas_tray 값을 tray_number와 같게 설정

                                # 시뮬레이션 실행
                                aspen.Engine.Run2()
                                real_count += 1
                                # 아웃풋 데이터 수집
                                liqout_flowrate_water = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\MOLEFLOW\\MIXED\\WATER").Value
                                liqout_flowrate_hydrogen = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\MOLEFLOW\\MIXED\\HYDROGEN").Value
                                liqout_flowrate_nitrogen = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\MOLEFLOW\\MIXED\\NITROGEN").Value
                                liqout_flowrate_ammonia = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\MOLEFLOW\\MIXED\\AMMONIA").Value
                                
                                remove_efficiency = (liqout_flowrate_ammonia)/ flowrate["ammonia"]
                                liqout_temp = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\STR_MAIN\\TEMP\\MIXED").Value
                                liqout_pressure = aspen.Tree.FindNode("\\Data\\Streams\\LIQOUT\\Output\\STR_MAIN\\PRES\\MIXED").Value
                                # logging.info("\n")
                                # logging.info(liqout_temp, liqout_pressure)
                                # logging.info("\n")
                                liqout_flowrate_total = liqout_flowrate_water+ liqout_flowrate_hydrogen+ liqout_flowrate_nitrogen+ liqout_flowrate_ammonia
                                
                                # 결과를 데이터프레임에 추가
                                data.append({
                                    "conversion": conversion,
                                    "gas_pressure": gas_pressure,
                                    "gas_temp": gas_temp,
                                    "column_pressure": column_pressure,
                                    "water_flowrate": water_flowrate,
                                    "tray_number": tray_number,
                                    "liqout_flowrate_water": liqout_flowrate_water,
                                    "liqout_flowrate_hydrogen": liqout_flowrate_hydrogen,
                                    "liqout_flowrate_nitrogen": liqout_flowrate_nitrogen,
                                    "liqout_flowrate_ammonia": liqout_flowrate_ammonia,
                                    "liqout_temp" : liqout_temp,
                                    "liqout_pressure" : liqout_pressure,
                                    "liqout_flowrate_total" : liqout_flowrate_total,
                                    "remove_eff": remove_efficiency,
                                })
                                if real_count % 50 == 0:
                                    fff += 1
                                    end_time = time.time()
                                    remain_time = (end_time - start_time)/real_count
                                    remain_time = (total_simulations-sim_count)*remain_time /60
                                    if fff == 1:
                                        first_remain_time = remain_time
                                    
                                    if remain_time > first_remain_time*1.5:
                                        print(111)
                                        if len(data) >= 10:
                                            df = pd.DataFrame(data)
                                            df.to_csv("./datas/scrubber_partial%s.csv"%(last_file_num+1), index=False)
                                            last_file_num +=1
                                            logging.info(f"Partial results saved at simulation {sim_count}")
                                        return 0
                                    # logging.info(end_time , start_time , total_simulations, sim_count)
                                    logging.info("남은시간 : %s 분 \n 전체 시뮬레이션 수 : %s \n 실제 시뮬레이션 된 수 : %s\n 완료 시뮬레이션 수 (건너뛴거까지): %s"%(remain_time, total_simulations, real_count, sim_count))
                                # 500번마다 데이터 저장
                                if real_count % 500 == 0:
                                    print(real_count)
                                    df = pd.DataFrame(data)
                                    df.to_csv("./datas/scrubber_partial%s.csv"%(last_file_num+1), index=False)
                                    last_file_num+=1
                                    logging.info(f"Partial results saved at simulation {sim_count}")
                                    data = []
                                if remove_efficiency > 0.99:
                                    break
                            # else: 
                                # logging.info("슈도 : %s"%pseudo_count)
                            if np.array([np.round(conversion, 2) ==last_rows_conversions ,np.round(gas_pressure, 2)  ==last_rows_pressures ,np.round(gas_temp, 2) ==last_rows_temperatures,np.round(column_pressure, 2) ==last_rows_column_pressures,np.round(water_flowrate, 2)==last_rows_water_flowrates  , tray_number == last_rows_tray_numbers]).all():
                                continue_flag =1
                            # logging.info({
                            #     "conversion": conversion,
                            #     "gas_pressure": gas_pressure,
                            #     "gas_temp": gas_temp,
                            #     "column_pressure": column_pressure,
                            #     "water_flowrate": water_flowrate,
                            #     "tray_number": tray_number,
                            #     "liqout_flowrate_water": liqout_flowrate_water,
                            #     "liqout_flowrate_hydrogen": liqout_flowrate_hydrogen,
                            #     "liqout_flowrate_nitrogen": liqout_flowrate_nitrogen,
                            #     "liqout_flowrate_ammonia": liqout_flowrate_ammonia,
                            #     "liqout_temp" : liqout_temp,
                            #     "liqout_pressure" : liqout_pressure,
                            #     "liqout_flowrate_total" : liqout_flowrate_total,
                            #     "remove_eff": remove_efficiency,
                            # })

                            

                                
    df = pd.DataFrame(data)
    df.to_csv("./datas/scrubber_partial%s.csv"%(last_file_num+1), index=False)
    last_file_num +=1
    print("saved")
    return sim_count / total_simulations
# 최종 데이터 저장
# df = pd.DataFrame(data)
# df.to_csv("simulation_results.csv", index=False)
# logging.info("Simulation completed and results saved to simulation_results.csv")
if __name__ == "__main__":
    result = main()
    sys.exit(result)