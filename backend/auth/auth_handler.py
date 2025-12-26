import time
from typing import Dict
import jwt

SECRET_KEY = "SECRET123"   # change this later
ALGORITHM = "HS256"

def token_response(token: str):
    return {
        "access_token": token
    }

def signJWT(user_id: str) -> Dict[str, str]:
    payload = {
        "user_id": user_id,
        "expires": time.time() + 3600
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token_response(token)

def decodeJWT(token: str) -> dict:
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded if decoded["expires"] >= time.time() else None
    except:
        return {}
