import os
import openai
import time
import logging
import random
import datetime
import ffmpeg
import io
import sys
import requests as req
from threading import Thread
from abc import ABC, abstractmethod
from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import Dict, Tuple



logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)

IP = "0.0.0.0"
PORT = os.getenv("PORT",443)

API_KEY = os.environ.get("OPENAI_KEY")
print(API_KEY)
OPEN_AI_CLIENT = openai.OpenAI(api_key=API_KEY)

TRANSCRIBE_MODEL = "whisper-1"
CHAT_MODEL = "gpt-5.1"

MAX_JOB_AGE = 30*60 # half hour

@dataclass
class ActiveJob:
    code: str
    thread: Thread
    start_time: float
    result: any = None
    over: bool = False


class BatchSerivce(ABC):
    def __init__(self):
        self.active_calls: int = 0
        self.jobs: Dict[str,ActiveJob] = {}
    
    @abstractmethod
    def run_job(self, inp: any, code: str): pass
    
    def assigne_job(self, inp) -> str:
        self.clean_old_jobs()
        code = str(random.randint(10**9,10**10))
        def job_runner():
            try:
                res = self.run_job(inp,code)
                job = self.jobs[code]
                job.result = res
            except Exception as e:
                logging.error(f"error in job thread {e}")
                return None
            finally:
                self.terminate_job(code)
        runner = Thread(target=job_runner).start()
        self.jobs[code] = ActiveJob(code,runner,time.time(),None)
        self.active_calls += 1
        return code

    def get_job_update(self, code) -> Tuple[any, bool]:
        if (not code in self.jobs): raise RuntimeError("try to acsses job that doesn't exist")
        job: ActiveJob = self.jobs[code]
        if (job.result is not None): return job.result, True
        else: return None, False
    
    def get_job(self, code: str) -> ActiveJob:
        if (not code in self.jobs): raise RuntimeError("try to acsses job that doesn't exist")
        return self.jobs[code]
        
    def terminate_job(self, code):
        logging.info(f"terminate job [{code}]")
        job = self.jobs[code]
        if (job.over): 
            logging.warning("try to terminate unactive job"); return
        job.over = True
        self.active_calls -= 1
    
    def clean_old_jobs(self):
        current_time = time.time()
        for code,job in self.jobs.items():
            age = current_time - job.start_time
            if (age > MAX_JOB_AGE):
                logging.info(f"remove the job [{code}] because of age")
                self.jobs.pop(code)
        

class TranscribeService(BatchSerivce):
    def __init__(self):  super().__init__()
    
    def run_job(self, inp, code):
        logging.info(f"got transcribe request with {inp} [{code}]")
        file_url = inp["file_url"]
        file_req = req.get(file_url)
        input_size =  int(file_req.headers.get("Content-Length", 0))
        logging.info(f"got file from size {input_size}")
        input_stream = io.BytesIO(file_req.content)
        output_stream = io.BytesIO()
       
        process = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='mp3')
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        # Write MP4 bytes to stdin and get the MP3 output
        stdout, stderr = process.communicate(input=input_stream.read())
        output_stream.write(stdout)
        output_stream.seek(0)
        output_stream.name = "audio.mp3"
        
        logging.info(f"file audio size: {output_stream.getbuffer().nbytes}")
        transcribetion = OPEN_AI_CLIENT.audio.transcriptions.create(model=TRANSCRIBE_MODEL,file=output_stream,response_format='text')
        
        job = self.jobs[code]
        logging.info(f"done transcribtion after [{round(time.time()-job.start_time,2)}]")  
        return transcribetion

class ChatService(BatchSerivce):
    def __init__(self): super().__init__()

    def run_job(self, inp, code):
        logging.info(inp)
        prompt = inp["prompt"]
        completion = OPEN_AI_CLIENT.chat.completions.create(model=CHAT_MODEL,messages=[{"role":"user","content":prompt}])
        #logging.info(completion)
        answer = completion.choices[0].message.content
        
        job = self.jobs[code]
        logging.info(f"got gpt answer after {round(time.time()-job.start_time,2)}s")
        return answer

TRAN_SERVICE = TranscribeService()
CHAT_SERVICE = ChatService()

def send_response(table):
    resp = jsonify(table)
    resp.headers.add("Access-Control-Allow-Origin","*")
    resp.headers.add("Content-Type", "application/json")
    resp.headers.add("Access-Control-Allow-Headers","Content-Type")
    return resp 

def job_to_json(job: ActiveJob) -> Dict:
    return {"over":job.over, "have_result": (job.result is not None), "res":job.result}

@app.get("/test")
def test():
    logging.info("call test")
    return send_response({"message":f"hello this is a test\ntime: {datetime.datetime.now()}\nactive tran {TRAN_SERVICE.active_calls}\n active chat {CHAT_SERVICE.active_calls}"})

@app.post('/transcribe')
def transcribe():
    try:
        code = TRAN_SERVICE.assigne_job(request.json)
        return send_response({"code":code})
    except Exception as e:
        logging.error(f"transcribe assigne fail with {e}")
        return send_response({"message":f"transcribe assigne error {e}"})
    
@app.post("/transcribe_update")
def transcribe_update():
    try:
        code = request.json["code"]
        job = TRAN_SERVICE.get_job(code)
        return send_response(job_to_json(job))
    except Exception as e:
        logging.error(f"transcribe update fail with {e}")
        return send_response({"message":f"transcribe update error {e}"})

@app.post('/chat')
def chat():
    try:
        code = CHAT_SERVICE.assigne_job(request.json)
        return send_response({"code":code})
    except Exception as e:
        logging.error(f"chat assigne fail with {e}")
        return send_response({"message":f"chat assigne error {e}"})
    
@app.post("/chat_update")
def chat_update():
    try:
        code = request.json["code"]
        job = CHAT_SERVICE.get_job(code)
        return send_response(job_to_json(job))
    except Exception as e:
        logging.error(f"chat update fail with {e}")
        return send_response({"message":f"chat update error {e}"})

if __name__ == '__main__':
    logging.info(f"Starting server on https://{IP}:{PORT}")
    app.run(host=IP, port=PORT)
    #app.run(host=IP, port=PORT, ssl_context=('cert.pem', 'key.pem'))