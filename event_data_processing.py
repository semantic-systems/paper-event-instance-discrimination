import json
import glob
import pandas as pd
from collections import Counter
from pathlib import Path
from datetime import datetime, timedelta
from itertools import chain


root = Path("./data/gdelt_crawled/")


class NaturalDisasterWikidata():
    def __int__(self):
        self.P_TIME = {"P585": "point in time",
                       "P580": "start time",
                       "P523": "temporal range start",
                       "P3415": "start period"}
        self.events = self.get_wikidata_natural_disaster_instances()
        self.events_within_timeframe, self.events_out_of_time, self.events_with_invalid_time = self.categorize_events_by_date()
        print(f"{len(self.events_within_timeframe)} instances are between 2021-01-01 and 2023-09-01.")
        print(f"{len(self.events_out_of_time)} instances are NOT between 2021-01-01 and 2023-09-01.")
        print(f"{len(self.events_with_invalid_time)} instances have invalid time format.")

    def get_dates_from_valid_event(self):
        return list(set(map(self.get_point_in_time_value, self.events_within_timeframe)))

    def get_point_in_time_value(self, event: dict):
        date = None
        properties = list(self.P_TIME.keys())
        for property in properties:
            if date is None:
                point_in_time_statement_dict = event["claims"].get(property, None)
                date = point_in_time_statement_dict[0]["mainsnak"]["datavalue"]["value"]["time"] if point_in_time_statement_dict is not None else None
            else:
                return date

    @staticmethod
    def filter_events_with_date(date):
        try:
            event_date = datetime.strptime(date, "+%Y-%m-%dT%H:%M:%SZ")
            start_date = datetime.strptime("+2021-01-01T00:00:00Z", "+%Y-%m-%dT%H:%M:%SZ")
            end_date = datetime.strptime("+2023-09-01T00:00:00Z", "+%Y-%m-%dT%H:%M:%SZ")
            return start_date <= event_date <= end_date
        except ValueError:
            # try:
            #     print(date[:8])
            #     event_date = datetime.strptime(date[:8], "+%Y-%m")
            #     start_date = datetime.strptime("+2021-01", "+%Y-%m")
            #     end_date = datetime.strptime("+2023-09", "+%Y-%m")
            #     return start_date <= event_date <= end_date
            # except ValueError:
            #     print("value error", date)
            return None

    def categorize_events_by_date(self):
        events_within_timeframe = []
        timeframe = []
        events_out_of_time = []
        events_with_invalid_time = []
        for event in self.events:
            t = self.get_point_in_time_value(event)
            if t is not None:
                start_time = self.filter_events_with_date(t)
                if start_time is None:
                    events_out_of_time.append(event)
                else:
                    if start_time:
                        events_within_timeframe.append(event)
                        timeframe.append(t)
                    else:
                        events_with_invalid_time.append(event)
        return events_within_timeframe, events_out_of_time, events_with_invalid_time

    @staticmethod
    def get_wikidata_natural_disaster_instances():
        event_path = "./data/filtered_natural_disaster_entities_included_subclasses.json"
        with open(event_path, 'r') as f:
            events_from_wikidata = json.load(f)
            print(f"{len(events_from_wikidata)} natural disaster instances from wikidata found.")
        return events_from_wikidata


class NaturalDisasterGdelt(object):
    def __int__(self):
        self.root = root

    def aggregate_extracted_news(self):
        aggregated_news_all_events = []
        for event_type in Path(self.root).iterdir():
            aggregated_news_per_event_type = []
            if event_type.is_dir():
                event_type_str = str(event_type).split("/")[-1]
                for country in Path(event_type).iterdir():
                    csv = Path(country, "aggregated_news.csv")
                    if csv.exists():
                        df = pd.read_csv(csv, index_col=False, header=0)
                        if "gdelt_search_keyword" not in df.columns:
                            df["gdelt_search_keyword"] = event_type_str
                        aggregated_news_per_event_type.append(df)
                    else:
                        aggregated_news_per_country = []
                        for csvs in country.rglob("*.csv"):
                            if csvs.name not in ["peaks_timeframe.csv", "aggregated_news.csv"]:
                                start_date = csvs.stem.split("_")[0]
                                end_date = csvs.stem.split("_")[1].split(".")[0]
                                df = pd.read_csv(csvs, index_col=False, header=0)
                                df["start_date"] = start_date
                                df["end_date"] = end_date
                                aggregated_news_per_country.append(df)
                        if aggregated_news_per_country:
                            aggregated_df_per_country = pd.concat(aggregated_news_per_country, axis=0, ignore_index=True)
                            if "gdelt_search_keyword" not in aggregated_df_per_country.columns:
                                aggregated_df_per_country["gdelt_search_keyword"] = event_type_str
                            aggregated_df_per_country.to_csv(csv, index=False)
                            aggregated_news_per_event_type.append(aggregated_df_per_country)

            if aggregated_news_per_event_type:
                aggregated_df_per_event_type = pd.concat(aggregated_news_per_event_type, axis=0, ignore_index=True)
                aggregated_path = Path(event_type, "aggregated_news_all_country.csv")
                aggregated_df_per_event_type.to_csv(aggregated_path, index=False)
                print("aggregated_news_all_country.csv saved.")
                aggregated_news_all_events.append(aggregated_df_per_event_type)
        if aggregated_news_all_events:
            aggregated_df_all_events = pd.concat(aggregated_news_all_events, axis=0, ignore_index=True)
            aggregated_path = Path(self.root, "aggregated_news_all_events.csv")
            aggregated_df_all_events.to_csv(aggregated_path, index=False)
            print("aggregated_news_all_events.csv saved.")

    def get_news_timeframe(self):
        start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
        end_date = datetime.strptime("2023-09-01", "%Y-%m-%d")
        date_list = pd.date_range(start_date, end_date, freq='D')
        print(f"Creating list of dates starting from {start_date}, to {end_date}")
        # convert date into string
        date_list_str = date_list.strftime("%Y-%m-%d")
        dir_path = str(Path(self.root, "**/**/peaks_timeframe.csv").absolute())
        delta = timedelta(days=1)

        all_dates = {}
        for file in glob.glob(dir_path, recursive=True):
            disaster = file.split("/")[-3]
            country = file.split("/")[-2]
            if disaster not in all_dates:
                all_dates[disaster] = {}
            if country not in all_dates:
                all_dates[disaster][country] = []
            df = pd.read_csv(file)
            start_dates = list(map(lambda x: datetime.strptime(x[:10], "%Y-%m-%d"), df["start_date"].tolist()))
            end_dates = list(map(lambda x: datetime.strptime(x[:10], "%Y-%m-%d"), df["end_date"].tolist()))
            for i in range(len(start_dates)):
                start = start_dates[i]
                while start <= end_dates[i]:
                    if start.strftime("%Y-%m-%d") not in all_dates[disaster][country]:
                        all_dates[disaster][country].append(start.strftime("%Y-%m-%d"))
                    start += delta
            # print(f"{len(all_dates[disaster][country])} days crawled for {disaster} in {country}")
        with open(Path(root, "news_date_per_country.json"), "w") as outfile:
            json.dump(all_dates, outfile)
        return all_dates


if __name__ == "__main__":
    wikidata_events = NaturalDisasterWikidata()
    wikidata_events.__int__()
    wikidata_dates = wikidata_events.get_dates_from_valid_event()
    gdelt_news = NaturalDisasterGdelt()
    gdelt_news.__int__()
    gdelt_news.aggregate_extracted_news()
    gdelt_dates = gdelt_news.get_news_timeframe()
    wikidata_dates = [datetime.strptime(wikidata_date, "+%Y-%m-%dT%H:%M:%SZ") for wikidata_date in wikidata_dates]
    wikidata_dates = [date.strftime("%Y-%m-%d") for date in wikidata_dates]
    gdelt_dates = list(set(list(chain(*[gdelt_dates[disaster][country] for disaster in gdelt_dates.keys() for country in gdelt_dates[disaster].keys()]))))
    intersection_dates = sorted(list(set(gdelt_dates).intersection(wikidata_dates)))
    print(f"Number of intersection dates: {len(intersection_dates)}\nNumber of gdelt dates: {len(gdelt_dates)}\nNumber of wikidata dates: {len(wikidata_dates)}")
    leftover_dates = [date for date in wikidata_dates if date not in intersection_dates]
    print(f"Events in wikidata but not in gdelt search: {leftover_dates}")
