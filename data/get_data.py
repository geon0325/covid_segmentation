from data import *

start_date = '3/1/20'

#countryList = ['us', 'brazil', 'india', 'russia', 'uk', 'france', 'italy', 'spain', 'turkey', 'germany', 'colombia', 'argentina', 'mexico', 'poland', 'iran', 'southafrica', 'ukraine', 'indonesia', 'peru', 'czech', 'netherlands', 'canada', 'chile', 'romania', 'israel', 'portugal', 'belgium', 'iraq', 'sweden', 'philippines', 'pakistan', 'switzerland', 'bangladesh', 'hungary', 'serbia', 'austria', 'jordan', 'morocco', 'japan', 'uae' ,'lebanon', 'saudiarabia', 'panama', 'slovakia', 'malaysia', 'belarus', 'ecuador', 'bulgaria', 'georgia', 'nepal', 'bolivia', 'croatia', 'dominicanrepublic', 'tunisia', 'azerbaijan', 'ireland', 'kazakhstan', 'greece', 'denmark', 'palestine', 'kuwait', 'costarica', 'moldova', 'lithuania', 'slovenia', 'egypt', 'guatemala', 'paraguay', 'armenia', 'honduras']
countryList = ['qatar']

country2population = {}
with open('population_by_country_2020.csv', 'r') as f:
    for line in f:
        term = line.split(',')
        country = term[0]
        population = term[1]
        
        if country == 'United States':
            country = 'Us'
        elif country == 'United Kingdom':
            country = 'Uk'
        elif country == 'South Africa':
            country = 'Southafrica'
        elif country == 'Saudi Arabia':
            country = 'Saudiarabia'
        elif country == 'Czech Republic (Czechia)':
            country = 'Czech'
        elif country == 'Dominican Republic':
            country = 'Dominicanrepublic'
        elif country == 'United Arab Emirates':
            country = 'Uae'
        elif country == 'Costa Rica':
            country = 'Costarica'
        elif country == 'State of Palestine':
            country = 'Palestine'

        country2population[country] = population

for country in countryList:
    _country = country[0].upper() + country[1:]
    print(_country)

    if country == 'us':
        data_df = load_ir_data_us(start_date)
    else:
        data_df = load_ir_data(_country, start_date)

    P = int(country2population[_country])

    f = open(country + '_sir.txt', 'w')
    for index, row in data_df.iterrows():
        if country == 'us':
            C = row['C']
            R = row['D']
        else:
            C = row['C']
            R = row['R'] + row['D']
        S = (P - C - R) / P
        f.write(str(S) + ' ' + str(C) + ' ' + str(R) + '\n')
    f.close()

    check = 0

    f = open(country + '_cr.txt', 'w')
    for index, row in data_df.iterrows():
        if country == 'us':
            C = row['C']
            R = row['D']
        else: 
            C = row['C']
            R = row['R'] + row['D']
        f.write(str(C) + ' ' + str(R) + '\n')
        check += (C + R)
    f.close()

