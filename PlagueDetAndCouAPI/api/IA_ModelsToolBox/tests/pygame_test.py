import sys, pygame
pygame.init()
print(f"pygame: {pygame}")
size = width, height = 600, 400
black = 255,255,255
screen = pygame.display.set_mode(size)
print(f"screen: {screen}")
tux = pygame.image.load(r"/home/erickmfs/Pictures/i2.png")
x=0
y=0

while 1:
 for event in pygame.event.get():
  if event.type == pygame.QUIT: sys.exit()
 screen.fill(black)
 screen.blit(tux,(200,200))
 screen.blit(tux,(x,y))

 pygame.display.flip()
 x+=1
