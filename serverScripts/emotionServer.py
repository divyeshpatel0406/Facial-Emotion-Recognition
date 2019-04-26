#source py3env/bin/activate

# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE for details.
import transfer
import download
import io
import subprocess
from twisted.internet import reactor, protocol
from twisted.logger import Logger, textFileLogObserver

class Echo(protocol.Protocol):
    """This is just about the simplest possible protocol"""

    #Executes when client connects to server
    def connectionMade(self):
        print("Client has connected.")
        client_IP = self.transport.getPeer()
        print("Your IP is ", client_IP)
        self.factory.logs = []

    #Executes when data is received
    def dataReceived(self, data):
        # "As soon as any data is received, write it back."
        # client = "Client: " + data.decode("utf-8") #Used by logger
        # resp = "I have recieved: " + data.decode("utf-8") #To write out data received
        # serverResp = "Server: " + resp  #Used by logger
        # print(resp) #To command prompt
        # self.transport.write(data)    #Sends back data to client
        # self.factory.logs.append(client)
        # self.factory.logs.append(serverResp)
        
        #turns data to text command if possible
        print("Attempting to decode byte data to text")
        try:
            textData = data.decode("utf-8")
            success = 1
            print("Data received is command")
        except:
            success = 0
            print("Data received is image")
        
        #Possible commands server can receive
        sizeCheck = 'SIZE'
        jsonCheck = 'RESULT'
        
        #Command SIZE creates new file for image to be stored in
        if success==1 and data.startswith(sizeCheck.encode("utf-8")):
            #Save image size, print command from client
            print(textData)
            textSplit = textData.split()
            sizeFile = open("imageSize.txt", 'w')
            sizeFile.write(textSplit[1])
            sizeFile.close()
            print("Created new file for image size")
            
            #Let client know size has been received
            response = "GOT SIZE"
            print("About to send response 'GOT SIZE' to client")
            self.transport.write(response.encode("utf-8"))
            print("Sent response to client")
            
            print("About to create new file for image")
            receivedImage = open("receivedImage.jpg", 'wb')
            receivedImage.close()
            print("New file created for image")
        
        #Command RESULT evaluates image and sends resulting JSON file to client
        elif success==1 and data.startswith(jsonCheck.encode("utf-8")):
            #Print command from client
            print(textData)
            
            print("About to evaluate image")
            #Run _ on receivedImage.jpg
            subprocess.run(["python3", "extraCustomTest.py"])
            print("Evaluated image, created json file")
            
            #Open json file created
            print("About to open json file")
            jsonFile = open("emotionValidation.json", 'rb')
            #jsonFile = open("fer_70.json", 'rb')
            jsonBytes = jsonFile.read()
            jsonSize = len(jsonBytes)
            print("Opened json file, size is")
            print(jsonSize)
            
            #Send json bit data to client
            print("About to send json bytes to client")
            #self.transport.write(jsonSize.encode("utf-8")
            self.transport.write(jsonBytes)
            print("Sent json bytes to client")
            
        #Image data is received to write to image file
        else :
            # if newPicture == 1 :
                # #Write the first bits of data
                
                # print("opened file to store image, about to WRITE to file")
                # receivedImage.write(data)
                # print("wrote to file, about to close file")
                # receivedImage.close()
                # print("closed file, about to send response that image was received")
                # newPicture = 0
            
            #Append bits of data
            print("About to append to image file")
            receivedImage = open("receivedImage.jpg", 'ab')
            receivedImage.write(data)
            receivedImage.close()
            print("Appended to file and closed it")
              
            receivedImage = open("receivedImage.jpg", 'rb')
            bytes = receivedImage.read()
            receivedImage.close()
            currentSize = len(bytes)
            print("Size after adding data", currentSize)
            
            sizeFile = open("imageSize.txt", 'r')
            sizeText = sizeFile.read()
            sizeFile.close()
            finalSize = int(sizeText)
            print("Final size of image", finalSize)
            
            if finalSize == currentSize :
                print("About to send 'GOT IMAGE' to client") 
                response = "GOT IMAGE"
                self.transport.write(response.encode("utf-8"))
                print("Sent message that file was received")
            

    #Executes when client disconnects
    def connectionLost(self, reason):
        print("Connection Lost.")
        file = open("log.txt", "a")
        for text in self.factory.logs:
            file.write(text + "\n")
        file.write("\n")
        file.close()
        transfer.transfer() # transfers the file to the S3 bucket



def main():
    """This runs the protocol on port 1234"""
    factory = protocol.ServerFactory()
    factory.protocol = Echo
    reactor.listenTCP(1234,factory)
    reactor.run()
    download.download() # download the log.txt from the S3
    
# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()
