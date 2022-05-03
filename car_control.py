import RPi.GPIO as gpio

FL_MOTOR = 0
FR_MOTOR = 1
RL_MOTOR = 2
RR_MOTOR = 3
# 17-Low 13 backwards
# 17-High 13 forwards
# 22-Low 13 backwards 24 forwards
# 22-High 13 backwards
# 23-Low 13 backwards
# 23-High Broken
# 24-Low Broken
# 24-High OK
def forward():
    init()
    # print('--backward--')
    gpio.output(17, gpio.LOW)
    gpio.output(22, gpio.HIGH)
    gpio.output(23, gpio.LOW)
    gpio.output(24, gpio.HIGH)
    
    
def backward():
    init()
    # print('--forward--')
    gpio.output(17, gpio.HIGH)
    gpio.output(22, gpio.LOW)
    gpio.output(23, gpio.HIGH)
    gpio.output(24, gpio.LOW)    


def left():
    init()
    # print('--left--')
    gpio.output(17, gpio.LOW)
    gpio.output(22, gpio.HIGH)
    gpio.output(23, gpio.HIGH)
    gpio.output(24, gpio.LOW)    


def right():
    init()
    # print('--right--')
    gpio.output(17, gpio.HIGH)    
    gpio.output(22, gpio.LOW)
    gpio.output(23, gpio.LOW)
    gpio.output(24, gpio.HIGH)



def init():
    gpio.setwarnings(False)

    gpio.setmode(gpio.BCM)
    gpio.setup(17, gpio.OUT)
    gpio.setup(22, gpio.OUT)
    gpio.setup(23, gpio.OUT)
    gpio.setup(24, gpio.OUT)
    
def clean_up_gpio():
    gpio.cleanup()