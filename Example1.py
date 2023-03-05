from PIL import Image
import pywhatkit
Image.open("download.jpg")

pywhatkit.image_to_ascii_art("download.jpg", "MyArt")
read_file= open("MyArt.txt","r")
print(read_file.read())