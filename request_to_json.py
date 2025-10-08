import requests
import json

url = "https://www.daraz.com.np/camera-screen-guards/?ajax=true&from=suggest_normal&q=camera"
#json to hit the request url with headers and url 
headers = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Accept" : "application/json, text/plain, */*" ,
}

response = requests.get(url,headers=headers)

data = response.json()

filename = "camera-screen-guards.json"

with open(filename,'w') as file:
    json.dump(data,file,indent=4)
