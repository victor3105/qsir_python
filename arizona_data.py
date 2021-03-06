import numpy as np
import csv

# COVID-19 data for Arizona US
INFECTED = np.array([652.0, 758.0, 902.0, 1137.0, 1265.0, 1501.0, 1683.0, 1896.0, 2135.0, 2422.0, 2667.0, 2797.0,
                     2956.0, 2929.0, 3015.0, 3285.0, 3427.0, 3583.0, 3429.0, 3437.0, 3627.0, 3803.0, 4005.0, 3755.0,
                     3722.0, 3893.0, 3977.0, 4241.0, 4466.0, 4668.0, 4876.0, 5032.0, 5230.0, 5426.0, 5836.0, 6111.0,
                     6469.0, 6681.0, 6930.0, 7239.0, 7588.0, 7773.0, 8262.0, 7744.0, 7808.0, 7989.0, 8265.0, 8642.0,
                     8976.0, 9373.0, 9630.0, 9815.0, 9951.0, 10178.0, 10386.0, 10712.0, 10900.0, 11219.0, 11444.0,
                     11564.0, 11757.0, 12123.0, 12565.0, 13035.0, 13697.0, 14268.0])
RECOVERED = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 249.0,
                      385.0, 460.0, 539.0, 539.0, 994.0, 1155.0, 1155.0, 1265.0, 1282.0, 1313.0, 1345.0, 1383.0, 1418.0,
                      1450.0, 1475.0, 1499.0, 1528.0, 1565.0, 1597.0, 1632.0, 1671.0, 1693.0, 1722.0, 1747.0, 2684.0,
                      2775.0, 2852.0, 2909.0, 2979.0, 3074.0, 3145.0, 3357.0, 3450.0, 3570.0, 3693.0, 3773.0, 3872.0,
                      3949.0, 4033.0, 4132.0, 4204.0, 4297.0, 4361.0, 4452.0, 4551.0, 4657.0, 4761.0])
DEAD = np.array([13.0, 15.0, 17.0, 20.0, 24.0, 29.0, 32.0, 41.0, 52.0, 64.0, 65.0, 73.0, 80.0, 89.0, 97.0, 108.0, 115.0,
                 122.0, 131.0, 142.0, 150.0, 169.0, 180.0, 184.0, 191.0, 208.0, 231.0, 249.0, 266.0, 273.0, 275.0,
                 275.0, 275.0, 308.0, 320.0, 330.0, 330.0, 362.0, 362.0, 395.0, 426.0, 450.0, 517.0, 532.0, 536.0,
                 542.0, 562.0, 595.0, 624.0, 651.0, 679.0, 680.0, 687.0, 705.0, 747.0, 764.0, 775.0, 801.0, 801.0,
                 807.0, 810.0, 834.0, 860.0, 886.0, 904.0, 907.0])

MOBILITY = []
with open('az_mobility.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    for line in csv_reader:
        MOBILITY.append(line)

MOBILITY = MOBILITY[1][61:128]
MOBILITY = [float(x) for x in MOBILITY]

# COVID-19 spread parameters
ALPHA = 0.15
BETA = 0.013
GAMMA = 0.01
