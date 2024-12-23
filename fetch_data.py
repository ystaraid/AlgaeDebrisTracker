import requests
import pandas as pd

# NOAA API 키
API_KEY = "aKGJXBivUfquUgxYLoWRvMboJeQHkKwk"

# API URL 및 파라미터
BASE_URL = "https://www.ncei.noaa.gov/access/services/search/v1/data"
params = {
    "dataset": "global-summary-of-the-day",  # 데이터셋 이름
    "startDate": "2022-01-01",  # 시작 날짜
    "endDate": "2022-12-31",  # 종료 날짜
    "stations": "GHCND:USW00094728",  # 스테이션 ID (NOAA 포털에서 확인)
    "dataTypes": "TEMP,PRCP",  # 데이터 유형
    "format": "json",  # 응답 형식
    "limit": 1000,
    "apiKey": API_KEY
}

# API 요청
response = requests.get(BASE_URL, params=params)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data["results"])
    df.to_csv("algae_data.csv", index=False)
    print("Algae data successfully downloaded and saved.")
else:
    print(f"Failed to fetch data: {response.status_code}")
