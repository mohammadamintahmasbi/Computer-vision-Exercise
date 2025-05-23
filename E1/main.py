from PIL import Image, ImageDraw

image = Image.open('img/image.jpg')
m , n = 100, 100
r = 200
c = 180
color = (255, 255, 255)

draw = ImageDraw.Draw(image)

draw.rectangle([m, n, m + r, n + c], fill=color)

image.show()

image.save('image2.jpg')