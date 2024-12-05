import csv
import wave
import requests
import json
from urllib.request import urlretrieve
headers = {"X-API-TOKEN": "NsG8rxlorrYg3TD48VBS4SXPYJrBogcsYoEdKN8M"}
with open('data/survey.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    entries = [row for row in csv_reader if row['Zip File_Name'] == '' ]
    for row in entries[2:3]:
        infiles = [row['wav' + str(i) + '_Url'] for i in range(1,6)]
        print(infiles)
        outfile = row['ResponseId']+'.wav'
        data= []
        for infile in infiles[0:1]:
            try:
                res = requests.get(infile, headers=headers, allow_redirects=True)
                temp = open("data/new_wav.wav", 'wb').write(res._content) 

                w = wave.open("data/new_wav.wav", 'rb')
                data.append( [w.getparams(), w.readframes(w.getnframes())] )
                w.close()
                output = wave.open(outfile, 'wb')
                output.setparams(data[0][0])
                for i in range(len(data)):
                    output.writeframes(data[i][1])
                output.close()
            except Exception as e:
                print(e)
                continue