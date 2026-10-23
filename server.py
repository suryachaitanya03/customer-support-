#!/usr/bin/env python3
"""Local Customer Support Copilot web server."""
from __future__ import annotations
import json, os, re, secrets, threading, time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from pipeline import ingest, parse_document, retrieve, answer_draft, evidence_notices

ROOT=Path(__file__).parent; DOCS=ROOT/"knowledge"; DOCS.mkdir(exist_ok=True)
LOCK=threading.RLock(); SESSIONS={}
try: USERS=json.loads(os.getenv("SUPPORT_USERS","{}"))
except json.JSONDecodeError: USERS={}
if not isinstance(USERS,dict): USERS={}
APP_ENV=os.getenv("APP_ENV","development").lower()
HOST=os.getenv("HOST","127.0.0.1")
COOKIE_SECURE=APP_ENV=="production"
if APP_ENV=="production" and (
    not USERS or any(not isinstance(account,dict) or account.get("role") not in {"agent","specialist","admin"} or len(str(account.get("password","")))<16 for account in USERS.values())
 ):
    raise RuntimeError("Production SUPPORT_USERS must contain valid roles and passwords of at least 16 characters.")
FAILS={}  # ip -> recent failed-login timestamps
MAX_FAILS,WINDOW=5,300
ALLOWED_EXT={".txt",".md",".json"}
FILTER_FIELDS={"product","plan","region","issue_type","type"}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Do not log query text, request bodies, cookies, or user-supplied document names.
        print(f"[{self.log_date_time_string()}] {self.command} {urlparse(self.path).path} {args[1] if len(args)>1 else ''}")
    def send_json(self,status,data,extra=None):
        body=json.dumps(data,ensure_ascii=False).encode(); self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff"); self.send_header("Referrer-Policy","no-referrer")
        if extra:
            for k,v in extra.items(): self.send_header(k,v)
        self.end_headers(); self.wfile.write(body)
    def read_json(self):
        n=int(self.headers.get("Content-Length","0"))
        if n>1_000_000: raise ValueError("Request exceeds the 1 MB limit.")
        try: d=json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as e: raise ValueError("Request body must be valid JSON.") from e
        if not isinstance(d,dict): raise ValueError("Request body must be an object.")
        return d
    def identity(self):
        if not USERS: return {"username":"Local agent","role":"agent"}
        cookie=SimpleCookie(self.headers.get("Cookie","")); morsel=cookie.get("support_session")
        return SESSIONS.get(morsel.value) if morsel else None
    def authorized(self):
        identity=self.identity()
        if not identity: self.send_json(401,{"error":"Sign in to access the support workspace."}); return None
        return identity
    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/healthz": return self.send_json(200,{"status":"ok"})
        if path=="/":
            body=(ROOT/"index.html").read_bytes(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(body); return
        if path=="/api/session":
            who=self.identity(); self.send_json(200,{"authenticated":bool(who),"login_required":bool(USERS),"user":who}); return
        who=self.authorized()
        if not who:return
        if path=="/api/knowledge":
            with LOCK: docs,errors=ingest(DOCS)
            visible=[d for d in docs if who["role"] in d.get("roles",["agent"])]
            files={d["source"]:d for d in visible}
            self.send_json(200,{"sources":[{"source":n,"title":d["title"],"type":d["type"],"product":d["product"],"plan":d["plan"],"region":d["region"],"issue_type":d["issue_type"],"updated":d["updated"],"effective_date":d["effective_date"],"authority":d["authority"],"source_url":d.get("source_url","")} for n,d in sorted(files.items())],"chunks":len(visible),"errors":errors,"role":who["role"],"configuration":{"generation":bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL")),"embeddings":bool(os.getenv("EMBEDDING_API_KEY") or (os.getenv("OPENAI_API_KEY") and os.getenv("EMBEDDING_MODEL"))),"embedding_model":os.getenv("EMBEDDING_MODEL") or "keyword-only","vector_store":os.getenv("VECTOR_STORE","memory"),"reranker":os.getenv("RERANKER_PROVIDER","hybrid")}}); return
        self.send_json(404,{"error":"Not found"})
    def do_POST(self):
        path=urlparse(self.path).path
        try: data=self.read_json()
        except ValueError as e: self.send_json(400,{"error":str(e)}); return
        if path=="/api/login":
            ip=self.headers.get("X-Forwarded-For",self.client_address[0]).split(",")[0].strip(); now=time.time()
            FAILS[ip]=[t for t in FAILS.get(ip,[]) if now-t<WINDOW]
            if len(FAILS[ip])>=MAX_FAILS: self.send_json(429,{"error":"Too many failed sign-in attempts. Try again in a few minutes."}); return
            username=str(data.get("username","")); password=str(data.get("password","")); entry=USERS.get(username)
            if not entry or not isinstance(entry,dict) or not secrets.compare_digest(str(entry.get("password","")),password): FAILS[ip].append(now); self.send_json(401,{"error":"Username or password is incorrect."}); return
            role=entry.get("role","agent")
            if role not in {"agent","specialist","admin"}: self.send_json(500,{"error":"Configured account has an invalid role."}); return
            token=secrets.token_urlsafe(32); SESSIONS[token]={"username":username,"role":role}
            secure="; Secure" if COOKIE_SECURE else ""
            self.send_json(200,{"authenticated":True,"user":SESSIONS[token]}, {"Set-Cookie":f"support_session={token}; HttpOnly; SameSite=Strict; Path=/{secure}"}); return
        if path=="/api/logout":
            cookie=SimpleCookie(self.headers.get("Cookie","")); m=cookie.get("support_session")
            if m: SESSIONS.pop(m.value,None)
            secure="; Secure" if COOKIE_SECURE else ""
            self.send_json(200,{"ok":True},{"Set-Cookie":f"support_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0{secure}"}); return
        who=self.authorized()
        if not who:return
        try:
            if path=="/api/ask":
                question=str(data.get("question","" )).strip()
                if not question or len(question)>4000: return self.send_json(400,{"error":"Enter a customer question up to 4,000 characters."})
                filters={k:str(data.get("filters",{}).get(k,"All")) for k in FILTER_FIELDS} if isinstance(data.get("filters",{}),dict) else {}
                with LOCK: docs,errors=ingest(DOCS)
                visible=[d for d in docs if who["role"] in d.get("roles",["agent"])]
                evidence=retrieve(question,visible,filters,who["role"])
                draft=answer_draft(question,evidence)
                notices=evidence_notices(evidence,draft.get("confidence","low"))
                if errors: notices.append({"level":"warning","text":"Some knowledge sources could not be indexed. Review the source list."})
                if draft.get("notice") and draft.get("confidence")!="low": notices.append({"level":"info","text":draft["notice"]})
                public=[]
                for d in evidence:
                    public.append({k:d.get(k) for k in ("chunk_id","source","source_url","title","type","product","plan","region","issue_type","updated","effective_date","authority","section","passage","score")})
                self.send_json(200,{"draft":draft,"evidence":public,"notices":notices,"filters":filters}); return
            if path=="/api/upload":
                if who["role"] not in {"specialist","admin"}: return self.send_json(403,{"error":"Only a specialist or admin can upload knowledge sources."})
                name=Path(str(data.get("name",""))).name; suffix=Path(name).suffix.lower(); content=str(data.get("content",""))
                if suffix not in ALLOWED_EXT: return self.send_json(400,{"error":"Unsupported file. Upload .txt, .md, or .json sources."})
                if not content.strip(): return self.send_json(400,{"error":"This document is empty."})
                if len(content.encode())>500_000: return self.send_json(400,{"error":"File exceeds the 500 KB limit."})
                target=DOCS/name
                try:
                    target.write_text(content,encoding="utf-8"); parse_document(target)
                except Exception as e:
                    target.unlink(missing_ok=True); return self.send_json(400,{"error":f"Could not parse document: {type(e).__name__}. Check its format and metadata."})
                return self.send_json(200,{"ok":True})
            if path=="/api/delete":
                if who["role"]!="admin": return self.send_json(403,{"error":"Only an admin can remove knowledge sources."})
                name=Path(str(data.get("name",""))).name; target=DOCS/name
                if target.exists() and target.suffix.lower() in ALLOWED_EXT: target.unlink()
                return self.send_json(200,{"ok":True})
            self.send_json(404,{"error":"Not found"})
        except RuntimeError as e: self.send_json(502,{"error":str(e)})
        except Exception as e: self.send_json(500,{"error":f"Request failed ({type(e).__name__}). Check the server configuration."})

if __name__=="__main__":
    host=HOST; port=int(os.getenv("PORT","8000"))
    print(f"Customer Support Copilot ready at http://{host}:{port}")
    if not USERS: print("Local demo mode: agent role; set SUPPORT_USERS JSON to require sign-in.")
    ThreadingHTTPServer((host,port),Handler).serve_forever()
