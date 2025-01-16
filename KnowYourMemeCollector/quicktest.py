from pytrends.request import TrendReq

# Create pytrends object
pytrends = TrendReq(hl='en-US', tz=360)

# Set up the keywords you want to search for
keywords = ['god']

# Get interest by region
pytrends.build_payload(kw_list=keywords,
                       geo='')
interest_by_region = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=True)

# Find the keyword with the highest interest value
#max_interest_keyword = max(keywords, key=lambda x: interest_by_region[x].max())
print(interest_by_region)

# Print the keyword with the highest interest value
