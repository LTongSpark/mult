# -*- coding: utf-8 -*-
from PIL import Image,ImageDraw,ImageFont

def add_num(img):
    draw = ImageDraw.Draw(img)
    myFont = ImageFont.truetype('D:\pychrom\mult\data\simhei.ttf',80)
    width,height = img.size
    draw.ellipse((width-200,0,width,200),fill="red",outline ="red")
    draw.text((width - 180,50), '刘通', font=myFont, fill="white")
    img.save('result.jpg','jpeg')

image = Image.open("C:/Users/Administrator/Pictures/1.jpg")
add_num(image)