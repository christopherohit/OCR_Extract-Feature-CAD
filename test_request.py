import requests



url ="http://localhost:2222/ocr-cad"
files = {'file': open('test.png', 'rb')}    
r = requests.post(url, files=files)
save_path =