import os
import requests

url = "https://optifine.net/downloadx?f=preview_OptiFine_1.20.6_HD_U_I9_pre1.jar&x=9ac01c48559464f53dc8424a2932e8b0"
response = requests.get(url)
print(response.headers)
