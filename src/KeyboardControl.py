import pyautogui
import pydirectinput as pdi
import time

def simulate_key_action(key):
    pdi.keyDown(key)
    time.sleep(1)
    pdi.keyUp(key)


def count_down(s_time):
    counter = s_time
    for i in range(s_time):
        print(counter)
        counter -= 1
        time.sleep(1)

count_down(5)
simulate_key_action('A')
