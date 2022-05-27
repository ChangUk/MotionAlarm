import cv2
import numpy as np
import telepot
import time
from telepot.loop import MessageLoop
import urllib.request
from playsound import playsound
import os

# INSTALLATION
# > pip install telepot
# > pip install playsound==1.2.2

IMAGE_FILE = "detected.jpg"
CURRENT_FILE = "current.jpg"
WINDOW_NAME = "Motion Detector"

TELEGRAM_TOKEN = "**********************************************"
THRESHOLD_MOTIONS = 10
SEND_LIMIT = 10

telegramBot = None
telegramChatId = None

def checkInternetConn(host="https://google.com"):
    try:
        urllib.request.urlopen(host, timeout=5)
        return True
    except:
        return False

def telegramHandler(msg):
    global telegramBot, telegramChatId, cap
    contentType, chatType, chatId = telepot.glance(msg)
    if contentType == "text":
        if msg["text"] == "start":
            telegramBot.sendMessage(chatId, "Motion Detector Starts!")
            telegramChatId = chatId
        elif msg["text"] == "stop":
            telegramBot.sendMessage(chatId, "Motion Detector Stopped!")
            telegramChatId = None
        elif msg["text"] == "check":
            if checkInternetConn(): telegramBot.sendMessage(chatId, "Working properly!")
            else: telegramChatId = None
        elif msg["text"] == "photo":
            try:
                _, curFrame = cap.read()
                cv2.imwrite(CURRENT_FILE, curFrame)
                telegramBot.sendPhoto(chatId, photo=open(CURRENT_FILE, "rb"))
            except: telegramBot.sendMessage(chatId, "Unable to take a photo!")
        elif msg["text"] == "warn":
            telegramBot.sendMessage(chatId, "Beep! Beep! Beep!")
            playsound("beep.mp3")
        elif msg["text"] == "shutdown":
            os.system("shutdown /s /t 1")
        elif msg["text"] == "help":
            telegramBot.sendMessage(chatId, "Commands: start, stop, check, photo, warn, shutdown")
        else:
            telegramBot.sendMessage(chatId, "Invalid command!")

cap = None
curFrame = None
prevFrame = None

# Initialzation
try:
    telegramBot = telepot.Bot(TELEGRAM_TOKEN)
    telegramBot.getMe()
    MessageLoop(telegramBot, telegramHandler).run_as_thread()

    cap = cv2.VideoCapture()
    cap.open(0, cv2.CAP_DSHOW)
except Exception as err:
    print(str(err))
    exit(1)

# Motion Detection
sent = 0
while True:
    try:
        ret, curFrame = cap.read()

        grayed = cv2.cvtColor(curFrame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=grayed, ksize=(5, 5), sigmaX=0)

        if prevFrame is None:
            prevFrame = blurred
            continue

        diffFrame = cv2.absdiff(src1=prevFrame, src2=blurred)
        prevFrame = blurred

        dilated = cv2.dilate(diffFrame, np.ones((5, 5)), 1)
        thresholdFrame = cv2.threshold(src=dilated, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(image=thresholdFrame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > THRESHOLD_MOTIONS and telegramChatId is not None:
            # Save current photo as a file
            cv2.imwrite(IMAGE_FILE, curFrame)

            try:
                if sent < SEND_LIMIT:
                    telegramBot.sendPhoto(telegramChatId, photo=open(IMAGE_FILE, "rb"))
                    sent += 1
                if sent == SEND_LIMIT:
                    telegramBot.sendMessage(telegramChatId, "Motion Detector Stopped!")
                    telegramChatId = None
                    sent = 0
            except:
                pass

        cv2.imshow(WINDOW_NAME, curFrame)
        if cv2.waitKey(1) == 27: break
        time.sleep(1)
    except Exception as err:
        print(str(err))
        break

cap.release()
cv2.destroyAllWindows()
