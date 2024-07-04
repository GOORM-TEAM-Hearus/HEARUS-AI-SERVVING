## Change Python Dev Env
```bash
source ./venvs/hearus/Scripts/activate
```

## Start AI Serving Server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```