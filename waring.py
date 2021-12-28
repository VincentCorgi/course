import smtplib
from email.mime.text import MIMEText

def waring_message():
    gmail_user = 'tonytony3tonytony3@gmail.com'
    gmail_password = 'b918273645' # your gmail password

    msg = MIMEText('someone fall !!!')
    msg['Subject'] = 'Test'
    msg['From'] = gmail_user
    msg['To'] = 'tonytony3tonytony3@gmail.com'

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(gmail_user, gmail_password)
    server.send_message(msg)
    server.quit()
    print('Email sent!')
    #https://accounts.google.com/DisplayUnlockCaptcha