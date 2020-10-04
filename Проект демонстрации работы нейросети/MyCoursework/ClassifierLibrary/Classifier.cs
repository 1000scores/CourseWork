using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using System.Net;

namespace ClassifierLibrary
{
    public class Classifier
    {
        public string GetAnswer(string input)
        {
            try
            {
                WebRequest request = WebRequest.Create("http://denis.hiweb.ru/cgi-bin/Coursework/SpamClassifier.py");

                var data = Encoding.ASCII.GetBytes($"input={input}");
                request.ContentType = "application/x-www-form-urlencoded";
                request.ContentLength = data.Length;
                request.Method = "POST";

                using (var stream = request.GetRequestStream())
                {
                    stream.Write(data, 0, data.Length);
                }


                WebResponse response = request.GetResponse();

                using (Stream dataStream = response.GetResponseStream())
                {
                    // Open the stream using a StreamReader for easy access.
                    StreamReader reader = new StreamReader(dataStream);
                    // Read the content.
                    string responseFromServer = reader.ReadToEnd();
                    // Display the content.
                    Console.WriteLine(responseFromServer);
                    string response2 = "";
                    for(int i = 0; i < responseFromServer.Length; i++)
                    {
                        if(responseFromServer[i] == '\n')
                        {
                            response2 += Environment.NewLine;
                        } else
                        {
                            response2 += responseFromServer[i];
                        }
                    }
                    return response2;
                }
            } catch(Exception ex)
            {
                return $"Ошибка при обращении к серверу: {ex.Message}";
            }
        }
    }
}
