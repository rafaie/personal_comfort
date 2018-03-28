import csv
import ntpath
import glob
import os

base_path = '../sample_data'
# base_path = '../comfort_data'
final_path = 'cleaned'


for file in glob.glob(os.path.join(base_path, "*.csv"))[:1]:
    print ("---------------------------------------------------------------")
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
            except Exception as e:
                pass
            data.append(d)
        base_data.append(data)

    # update sampeling row
    base_data2 = []
    base_data[0].append('met')
    base_data2.append(base_data[0])
    # base_data2.append(base_data[0])
    last_row = []
    roomTempreture = 0
    roomHumidity = 0
    locationType = 0
    clothingScore = 0
    clothing = 0
    for i in range(len(base_data[1:])):
        row = base_data[i+1]
        # process to chose the base_row
        data_type = 10
        try:
            data_type = int(row[1])
        except Exception as e:
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
            last_row = row

    # write in to csv
    file_name = ntpath.basename(file)
    with open(os.path.join(final_path, file_name), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='\'',
                            quoting=csv.QUOTE_MINIMAL)
        for row in base_data2:
            writer.writerow(row)

    # fmt = '%Y-%m-%d %H:%M:%S'
    # d1 = datetime.strptime('2010-01-01 17:31:22', fmt)
    # d2 = datetime.strptime('2010-01-01 17:35:22', fmt)
    # print (d2-d1).seconds/60
