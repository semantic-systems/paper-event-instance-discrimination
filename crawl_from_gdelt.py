from typing import List, Dict
import matplotlib.pyplot as plt
from gdeltdoc import GdeltDoc, Filters
import numpy as np
import pandas as pd
import time
import urllib
from datetime import datetime, timedelta
from pathlib import Path
from itertools import chain
from scipy import ndimage
from scipy.signal import find_peaks, peak_widths, find_peaks_cwt
import json
from tqdm import tqdm


def get_gdelt_country():
    data = urllib.request.urlopen("http://data.gdeltproject.org/api/v2/guides/LOOKUP-COUNTRIES.TXT").read().decode().split('\r\n')
    gdelt_countries = {}
    for line in data:
        if line == "":
            continue
        alpha_2 = line.split("\t")[0]
        name = line.split("\t")[1]
        gdelt_countries[name] = alpha_2
    return gdelt_countries


def to_date(df, x):
    """
    takes the first Timestamp of the df as a start date
    and then converts a given relative date x (in days)
    back into a "normal" date

    Note how this only works since we resampled the df on a daily basis!
    """
    _start = df.loc[0, 'datetime']
    return pd.to_datetime(_start) + pd.to_timedelta(x, unit='D')


def filter_dates_within_range(df):
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    unique_dates = set()
    for _, row in df.iterrows():
        for date in pd.date_range(start=row['start_date'], end=row['end_date']):
            unique_dates.add(date)

    return sorted([d.strftime("%Y-%m-%d") for d in unique_dates])


def lpfilter(input_signal, win):
    # Low-pass linear Filter
    # (2*win)+1 is the size of the window that determines the values that influence
    # the filtered result, centred over the current measurement
    kernel = np.lib.pad(np.linspace(1, 3, win), (0, win - 1), 'reflect')
    kernel = np.divide(kernel, np.sum(kernel))  # normalise
    output_signal = ndimage.convolve(input_signal, kernel)
    return output_signal


def get_outburst_timeframe_per_country(keyword, start_date, end_date, country):
    gdelt_filters = Filters(
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        country=country
    )
    timelinevol = gdelt.timeline_search("timelinevol", gdelt_filters)
    timelinevol['smoothed_vi'] = lpfilter(timelinevol["Volume Intensity"], 5)
    fig = timelinevol.plot(x='datetime', y=["Volume Intensity", "smoothed_vi"])

    # now find the peaks
    idx, properties = find_peaks(timelinevol["smoothed_vi"], width=1, rel_height=0.9)

    l = properties["left_ips"]
    r = properties["right_ips"]
    p = properties["prominences"]
    w = properties["widths"]
    wh = properties["width_heights"]

    plt.plot(timelinevol["datetime"][idx], timelinevol["smoothed_vi"][idx], "x")
    plt.hlines(y=wh, xmin=to_date(timelinevol, l), xmax=to_date(timelinevol, r), color="C1")
    sd = []
    ed = []
    peaks = []
    for i, n in enumerate(l):
        sd.append(to_date(timelinevol, l[i]))
        ed.append(to_date(timelinevol, r[i]))
        peaks.append(i)
    df = pd.DataFrame.from_dict({'pearks': peaks, 'start_date': sd, 'end_date': ed})

    fig.figure.savefig(f"./data/gdelt_crawled/{keyword}/{alpha_to_name(country)}/volumne_intensity.png")
    df.to_csv(f"./data/gdelt_crawled/{keyword}/{alpha_to_name(country)}/peaks_timeframe.csv")
    return df


def get_countries_with_events(keyword, start_date, end_date):
    gdelt_filters = Filters(
        keyword=keyword,
        start_date=start_date,
        end_date=end_date
    )
    timelinesourcecountry = gdelt.timeline_search("timelinesourcecountry", gdelt_filters)
    countries_name = [c.split(" Volume Intensity")[0] for c in timelinesourcecountry.columns.values[1:]]
    countries_dict_of_search = {}
    for c in countries_name:
        if c == "Vietnam":
            c = "Vietnam, Democratic Republic of"
        countries_dict_of_search[c] = countries.get(c, None)
    unsupported_countries = {keyword: [name for name, alpha in countries_dict_of_search.items() if alpha is None]}
    countries_list = list(set(countries_dict_of_search.values()))
    return countries_list, unsupported_countries


def alpha_to_name(alpha):
    return alpha_to_name_map[alpha]


def validate_extracted_dates(unique_dates: List, country: str, keyword: str) -> List:
    validated_dates = []
    for d in unique_dates:
        end = datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)
        end = end.strftime("%Y-%m-%d")
        path = f"./data/gdelt_crawled/{keyword}/{alpha_to_name(country)}/{d}_{end}.csv"
        if not Path(path).exists():
            validated_dates.append(d)
    return validated_dates


gdelt_search_keywords = {"storm": ["storm", "hurricane", "tornado", "flood", "tsunami"],
                         "explosion": ["explosion"],
                         "wildfire": ["wildfire"],
                         "earthquake": ["earthquake"],
                         "pandemic": ["pandemic"],
                         "drought": ["drought"],
                         "human_caused_disaster": ["shooting"]
                         }

start_date = "2021-01-01"
end_date = "2023-09-01"
gdelt = GdeltDoc()
countries = get_gdelt_country()
alpha_to_name_map = {value: key for key, value in countries.items()}
unsupported_countries_store  = []

for keyword in list(chain(*gdelt_search_keywords.values())):
    eventful_countries, unsupported_countries = get_countries_with_events(keyword, start_date, end_date)
    eventful_countries = [c for c in eventful_countries if c is not None]
    unsupported_countries_store.append(unsupported_countries)

    for country in tqdm(eventful_countries):
        summary = []
        path = Path(f"./data/gdelt_crawled/{keyword}/{alpha_to_name(country)}/")
        if not path.exists():
            path.mkdir(parents=True, exist_ok=False)
        df_peak_per_country = get_outburst_timeframe_per_country(keyword, start_date, end_date, country)
        unique_date = filter_dates_within_range(df_peak_per_country)
        unique_date = validate_extracted_dates(unique_date, country, keyword)
        for d in tqdm(unique_date, leave=False):
            summary_per_day = {"all_language": 0, "english": 0}
            end = datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)
            end = end.strftime("%Y-%m-%d")
            try:
                gdelt_filters = Filters(
                    keyword=keyword,
                    start_date=d,
                    end_date=end,
                    country=country
                )
                resulting_articles = gdelt.article_search(gdelt_filters)
                summary_per_day["all_language"] = len(resulting_articles)
                time.sleep(5)
                if resulting_articles.empty:
                    # summary_per_day["english"] = 0
                    # summary.append({d: summary_per_day})
                    continue
                else:
                    english_df = resulting_articles.query('language == "English"')
                    if len(english_df) != 0:
                        path = f"./data/gdelt_crawled/{keyword}/{alpha_to_name(country)}/{d}_{end}.csv"
                        english_df.to_csv(f"./data/gdelt_crawled/{keyword}/{alpha_to_name(country)}/{d}_{end}.csv")
            except ValueError:
                continue
            except AttributeError:
                continue
            except ValueError:
                continue
                # summary_per_day["english"] = len(english_df)
                # summary.append({d: summary_per_day})

    #         with open(f"./data/gdelt_crawled/{keyword}/{alpha_to_name(country)}/summary.json", 'w') as fp:
    #             json.dump(summary, fp)
    # with open(f"./data/gdelt_crawled/{keyword}/unsupported_country_unsupported_country.json", 'w') as fp:
    #     json.dump(unsupported_countries_store, fp)



