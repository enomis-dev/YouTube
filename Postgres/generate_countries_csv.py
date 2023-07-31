import requests
import sys
import pandas as pd

def get_world_capitals():
    url = "https://restcountries.com/v3.1/all"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        capitals_data = []

        for country_info in data:
            if "name" not in country_info.keys():
                continue
            country = country_info["name"]["common"]
            capital = country_info["capital"][0] if "capital" in country_info else "N/A"
            population = country_info["population"] if "population" in country_info else "N/A"
            # data cleaning to avoid encoding errors
            #if country == 'Moldova':
            #    capital = 'Chisinau'
            capitals_data.append((country, capital, population))

        return capitals_data

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data: {e}")
        return None


if __name__ == "__main__":
    capitals = get_world_capitals()
    if capitals:
        # save csv
        dataframe = pd.DataFrame(capitals, columns=['ID', 'Country', 'Capital', 'Population'])
        dataframe.to_csv('countries.csv')

        # Set the encoding to UTF-8 for printing Unicode characters
        sys.stdout.reconfigure(encoding='utf-8')

        # print data
        for i, (country, capital, population) in enumerate(capitals):
            print(f"{i}. Country: {country}, Capital: {capital}, Population: {population}")
