#Import and instantiate socket library
import socket


#Image to send
image = "Sad2.jpg"

#Port to receive data from
recvPort = 1024

#Host(the server) and port 
host = '3.16.151.132' #Originally 127.0.0.1
sendPort = 1234

#Create socket and connect
socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.connect((host,sendPort))
print 'Successfully connected to: %s' % host

#Send input to server, server sends same input back
#Echo
# while True:
    # data = input()    #Waits for input
    # socket.sendall(data.encode())
    # given = socket.recv(1024)    #Server sends back data to this port?
    # toprint = given.decode()
    # print("The server responds with:",toprint)

try:

    #Open image to evaluate
    testImage = open(image, 'rb')
    bytes = testImage.read()
    size = len(bytes)

    #Send image size to server
    socket.sendall("SIZE %s" % size)
    print 'Sent: SIZE %s' % size
    
    #Receive answer to image size, 4096 is size of response
    answer = socket.recv(4096)
    answer = answer.decode("utf-8")
    print 'Answer to image size: %s' % answer
    
    #Send image to server
    if answer == 'GOT SIZE':
        
        #Send image bit data to server
        print 'About to send image bytes to server'
        socket.sendall(bytes)
        print 'Sent image bytes to server'

        #Check response to image data
        answer = socket.recv(4096)
        print 'Answer to image data %s' % answer
        
        #Start image evaluation
        if answer == 'GOT IMAGE' :
            #Send command to get result/json file
            socket.sendall("RESULT")

            #Receive json file, 409600 to be safe(?)
            jsonData = socket.recv(4096)

            #Recreate json file
            emotionEval = open("emotionEval.json", 'wb')
            emotionEval.write(jsonData)
            emotionEval.close()
        
    testImage.close()

# try:
    # #Send command to get result/json file
    # socket.sendall("RESULT")
    
    # #Receive json file
    # jsonData = socket.recv(409600)
    
    # #Recreate json file
    # emotionEval = open("emotionEval.json", 'wb')
    # emotionEval.write(jsonData)
    # emotionEval.close()

finally:
    socket.close()
