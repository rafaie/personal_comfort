import sys
import csv


TYPE_1 = 0
TYPE_2 = 1

clmn1 = ['', '']
clmn2 = ['', '']

clmn1[0] = ['roomTempreture', 'roomHumidity']
clmn2[0] = ['heartRate', 'skinTemperature', 'clothingScore', 'met',
            'resistance']

clmn1[1] = ['roomTempreture', 'roomHumidity', 'roomTempreture_15min',
            'roomHumidity_15min', 'roomTempreture_30min',
            'roomHumidity_30min']
clmn2[1] = ['heartRate', 'skinTemperature', 'clothingScore', 'met',
            'resistance', 'met_15min', 'heartRate_15min',
            'skinTemperature_15min', 'heartRate_30min',
            'skinTemperature_30min']


def gen_clm_list(clm_list_fname, t):

    with open(clm_list_fname, 'w') as fi:
        csv_out = csv.writer(fi, delimiter=',')

        clm1_opt = ["{0:b}".format(i) for i in range(2 ** len(clmn1[t]))]
        clm2_opt = ["{0:b}".format(i) for i in range(2 ** len(clmn2[t]))]

        for opt1 in clm1_opt:
            clm1 = [clmn1[t][len(opt1) - i - 1] for i in range(len(opt1))
                    if opt1[i] == '1']
            if len(clm1) > 0:
                for opt2 in clm2_opt:
                    clm2 = [clmn2[t][len(opt2) - i - 1]
                            for i in range(len(opt2)) if opt2[i] == '1']
                    if len(clm2) > 0:
                        csv_out.writerow(clm1 + clm2)
                        print(clm1 + clm2)


if __name__ == "__main__":
    clm_list_fname = 'clmn_list.csv'
    t = TYPE_1
    if len(sys.argv) > 1:
        clm_list_fname = sys.argv[1]
        if len(sys.argv) > 2:
            try:
                t = int(sys.argv[2]) - 1

                if t != TYPE_1 and t != TYPE_2:
                    data_type_in_out = TYPE_1

            except Exception:
                pass

    gen_clm_list(clm_list_fname, t)
