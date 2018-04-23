import csv
import ntpath
import sys
import os
import glob
import datetime
import pandas as pd
import numpy as np


# base_path = '../sample_data'
base_path = '../comfort_data'
final_path = 'cleaned'

# Data tyoe for the output
ALL_DATA = 0
TYPE_1 = 1
TYPE_10 = 10


def calc_avg_Value_during(base_data, l, cols, duration):
    avg = {}

    fmt = '%m/%d/%Y %H:%M:%S'
    d = datetime.datetime.strptime(base_data[l][0], fmt)
    for c in cols:
        avg[c] = [base_data[l][c]]

    for i in reversed(range(2, l)):
        d2 = datetime.datetime.strptime(base_data[i][0], fmt)
        # print(l, duration, d2, d, abs(d - d2).total_seconds() / 60, avg)
        if abs(d - d2).total_seconds() / 60 <= duration:
            for c in cols:
                avg[c].append(base_data[i][c])
        else:
            break
    out = []
    for c in cols:
        out.append(sum(avg[c])/len(avg[c]))

    # print(avg, out)
    return out


def clean_file(file, data_type_in_out=0):
    print ("-----------------------")
    print (file)
    reader = csv.reader(open(file, "r"), delimiter=',')

    # Read data and convert to float
    base_data = []
    for row in reader:
        data = []
        for cell in row:
            d = cell
            try:
                d = round(float(d), 4)
            except Exception:
                pass
            data.append(d)
        base_data.append(data)

    # update sampeling row
    base_data2 = []
    base_data2.append(base_data[0] + ['met', 'met_15min'] +
                      ['roomTempreture_15min', 'roomHumidity_15min'] +
                      ['resistance_15min', 'heartRate_15min'] +
                      ['skinTemperature_15min'] +
                      ['met_30min', 'roomTempreture_30min'] +
                      ['roomHumidity_30min', 'resistance_30min'] +
                      ['heartRate_30min', 'skinTemperature_30min'])

    print(base_data2)
    last_row = []
    roomTempreture = 0
    roomHumidity = 0
    locationType = 0
    clothingScore = 0
    clothing = 0

    for i in range(1, len(base_data)):
        try:
            base_data[i][49] = (base_data[i][49] - 32) * 5/9.0
        except:
            pass

    for i in range(1, len(base_data)):
        row = base_data[i]

        # process to chose the base_row
        data_type = 10
        try:
            data_type = int(row[1])
        except Exception:
            pass

        if len(last_row) <= 0:
            row.append(0.0)
            last_row = row
            base_data2.append(last_row)
        else:
            date_diff = 0
            date_diff = float(row[18]) - float(last_row[18])
            if row[17] - last_row[17] > 0 and date_diff > 1:
                row.append((row[17] - last_row[17]) / date_diff * 1000 * 60)
            else:
                row.append(1.00001)

            if data_type == 1:
                roomTempreture = row[49]
                roomHumidity = row[50]
                locationType = row[51]
                clothingScore = row[52]
                clothing = row[53]
            else:
                row[49] = roomTempreture
                row[50] = roomHumidity
                row[51] = locationType
                row[52] = clothingScore
                row[53] = clothing

            base_data2.append(row)

            # Add Average for the duration
            base_data2[-1] += calc_avg_Value_during(base_data2, i,
                                                    [54, 49, 50, 16, 3, 23],
                                                    15)
            base_data2[-1] += calc_avg_Value_during(base_data2, i,
                                                    [54, 49, 50, 16, 3, 23],
                                                    30)

            last_row = row

    fmt = '%m/%d/%Y %H:%M:%S'
    # d = datetime.datetime.strptime(base_data[l][0], fmt)

    d = pd.DataFrame(base_data2[1:], columns=base_data2[0])
    print(d.columns)
    d['hour'] = d['currentTime'].apply(
        lambda x: datetime.datetime.strptime(x, fmt).hour)

    d['vote2'] = d['vote'].apply(
        lambda x: 0 if x >= -1 and x <= 1 else 1 * np.sign(x))

    print(d.shape)
    d = d[(d['hour'] >= 8) & (d['hour'] <= 21)]
    if data_type_in_out == TYPE_1:
        d = d[(d['Data Type'] == TYPE_1)]
    elif data_type_in_out == TYPE_10:
        d = d[(d['Data Type'] == TYPE_10)]
    print(d.shape)

    file_name = ntpath.basename(file)
    d.to_csv(os.path.join(final_path, file_name))


if __name__ == "__main__":
    data_type_in_out = ALL_DATA

    if len(sys.argv) > 1:
        try:
            data_type_in_out = int(sys.argv[1])

            if data_type_in_out != TYPE_1 and data_type_in_out != TYPE_10:
                data_type_in_out = ALL_DATA
        except Exception:
            pass

    for file in glob.glob(os.path.join(base_path, "*.csv")):
        clean_file(file, data_type_in_out)
