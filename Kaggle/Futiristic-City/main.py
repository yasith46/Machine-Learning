import joblib
import pandas as pd

model = joblib.load('trained_model1.joblib')

#Getting Data
citysel = input('Please choose the city: \n 1. SolarisVille\n 2. AquaCity\n 3. Neuroburg\n 4. Ecoopolis\n 5. TechHaven\n 6. MetropolisX\nSelect ')
match citysel:
    case 1: city = 'SolarisVille'
    case 2: city = 'AquaCity'
    case 3: city = 'Neuroburg'
    case 4: city = 'Ecopolis'
    case 5: city = 'TechHaven'
    case 6: city = 'MetropolisX'

vehiclesel = input('Please choose the mode of transportation:\n 1. Drone\n 2. Flying Car\n 3. Autonomous Vehicle\n 4. Car\nSelect ')
match vehiclesel:
    case 1: vehicle = 'Drone'
    case 2: vehicle = 'Flying Car'
    case 3: vehicle = 'Autonomous Vehicle'
    case 4: vehicle = 'Car'

weathersel = input('Please choose the weather:\n 1. Snowy\n 2. Solar Flare\n 3. Clear\n 4. Rainy\n 5. Electromagnetic Storm\nSelect ')
match weathersel:
    case 1: weather = 'Snowy'
    case 2: weather = 'Solar Flare'
    case 3: weather = 'Clear'
    case 4: weather = 'Rainy'
    case 5: weather = 'Electromagnetic Storm'

econsel = input('Please choose the economic condition:\n 1. Stable\n 2. Recession\n 3. Booming\nSelect')
match econsel:
    case 1: econ = 'Stable'
    case 2: econ = 'Recession'
    case 3: econ = 'Booming'

datesel = input('Please choose the date:\n 1. Sunday\n 2. Monday\n 3. Tuesday\n 4. Wednesday\n 5. Thursday\n 6. Friday\n 7. Saturday\nSelect ')
match datesel:
    case 1: date = 'Sunday'
    case 2: date = 'Monday'
    case 3: date = 'Tuesday'
    case 4: date = 'Wednesday'
    case 5: date = 'Thursday'
    case 6: date = 'Friday'
    case 7: date = 'Saturday'

hod   = float(input('Input the hour of the day: '))
speed = float(input('Input the apee: '))

phsel = input('Is it the peak hour?: (Y/N) ')
match phsel:
    case 'Y': ph = 1
    case 'N': ph = 0 

resel = input('Is there a random event?: (Y/N) ')
match resel:
    case 'Y': re = 1
    case 'N': re = 0


#Converting to dataframe
query = [1, 1, 1, 1, 1, hod, speed, ph, re]
data = pd.DataFrame(query).T

data.columns = [city,vehicle, weather, econ, date,'Hour Of Day','Speed','Is Peak Hour','Random Event Occurred']

prediction = model.preds(data[])
print(prediction)
