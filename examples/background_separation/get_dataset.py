import urllib.request

if __name__ == "__main__":
    url = "https://data.kitware.com/api/v1/item/56f587a98d777f753209cb6d/download"
    urllib.request.urlretrieve(url, 'video_name.mp4') 
    